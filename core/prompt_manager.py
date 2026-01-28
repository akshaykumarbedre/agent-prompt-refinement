"""Prompt version manager with API integration for get/update prompts."""
import json
import httpx
from pathlib import Path
from typing import Optional
from rich.console import Console

from .models import PromptVersion
from config import settings

console = Console()


class PromptManager:
    """Manages prompt versions with local storage and remote API sync."""
    
    def __init__(self):
        self.prompts_dir = settings.PROMPTS_DIR
        self.prompts_dir.mkdir(parents=True, exist_ok=True)
    
    # ============== Local Storage Methods ==============
    
    def _get_version_path(self, version_number: int) -> Path:
        """Get file path for a version number."""
        return self.prompts_dir / f"v{version_number:03d}.json"
    
    def _get_next_version_number(self) -> int:
        """Get the next available version number."""
        existing = list(self.prompts_dir.glob("v*.json"))
        if not existing:
            return 1
        numbers = [int(p.stem[1:]) for p in existing]
        return max(numbers) + 1
    
    def save_version(self, prompt: PromptVersion) -> PromptVersion:
        """Save a prompt version to local storage."""
        path = self._get_version_path(prompt.version_number)
        with open(path, "w") as f:
            json.dump(prompt.model_dump(), f, indent=2)
        return prompt
    
    def get_version(self, version_number: int) -> Optional[PromptVersion]:
        """Get a specific prompt version from local storage."""
        path = self._get_version_path(version_number)
        if not path.exists():
            return None
        with open(path, "r") as f:
            data = json.load(f)
        return PromptVersion(**data)
    
    def get_latest_version(self) -> Optional[PromptVersion]:
        """Get the latest prompt version."""
        existing = list(self.prompts_dir.glob("v*.json"))
        if not existing:
            return None
        latest = max(existing, key=lambda p: int(p.stem[1:]))
        with open(latest, "r") as f:
            data = json.load(f)
        return PromptVersion(**data)
    
    def list_versions(self) -> list[PromptVersion]:
        """List all prompt versions."""
        versions = []
        for path in sorted(self.prompts_dir.glob("v*.json")):
            with open(path, "r") as f:
                data = json.load(f)
            versions.append(PromptVersion(**data))
        return versions
    
    def get_lineage(self, version_number: int) -> list[PromptVersion]:
        """Get the lineage (ancestry) of a prompt version."""
        lineage = []
        current = self.get_version(version_number)
        while current:
            lineage.append(current)
            if current.parent_version_id:
                # Find parent by ID
                for v in self.list_versions():
                    if v.id == current.parent_version_id:
                        current = v
                        break
                else:
                    current = None
            else:
                current = None
        return lineage
    
    def create_version(
        self,
        system_prompt: str,
        instruction_template: str = "",
        few_shot_examples: list[dict] = None,
        parent_version: Optional[PromptVersion] = None,
        optimization_notes: str = "",
    ) -> PromptVersion:
        """Create a new prompt version."""
        version_number = self._get_next_version_number()
        prompt = PromptVersion(
            version_number=version_number,
            system_prompt=system_prompt,
            instruction_template=instruction_template,
            few_shot_examples=few_shot_examples or [],
            parent_version_id=parent_version.id if parent_version else None,
            optimization_notes=optimization_notes,
        )
        return self.save_version(prompt)
    
    # ============== Remote API Methods ==============
    
    async def fetch_prompt_from_api(self, prompt_id: Optional[str] = None) -> dict:
        """
        Fetch current prompt from the remote API.
        
        API Expected Response:
        {
            "prompt_id": "abc123",
            "system_prompt": "You are a helpful assistant...",
            "instruction_template": "...",
            "few_shot_examples": [...],
            "metadata": {...}
        }
        """
        async with httpx.AsyncClient() as client:
            headers = {}
            if settings.PROMPT_API_KEY:
                headers["Authorization"] = f"Bearer {settings.PROMPT_API_KEY}"
            
            params = {}
            if prompt_id:
                params["prompt_id"] = prompt_id
            
            response = await client.get(
                settings.PROMPT_API_GET_URL,
                headers=headers,
                params=params,
                timeout=30.0
            )
            response.raise_for_status()
            return response.json()
    
    async def update_prompt_via_api(
        self,
        prompt_id: str,
        system_prompt: str,
        instruction_template: str = "",
        few_shot_examples: list[dict] = None,
        optimization_notes: str = "",
    ) -> dict:
        """
        Update prompt via the remote API.
        
        API Expected Request:
        {
            "prompt_id": "abc123",
            "system_prompt": "Updated prompt...",
            "instruction_template": "...",
            "few_shot_examples": [...],
            "optimization_notes": "Fixed formatting issues"
        }
        """
        async with httpx.AsyncClient() as client:
            headers = {"Content-Type": "application/json"}
            if settings.PROMPT_API_KEY:
                headers["Authorization"] = f"Bearer {settings.PROMPT_API_KEY}"
            
            payload = {
                "prompt_id": prompt_id,
                "system_prompt": system_prompt,
                "instruction_template": instruction_template,
                "few_shot_examples": few_shot_examples or [],
                "optimization_notes": optimization_notes,
            }
            
            response = await client.post(
                settings.PROMPT_API_UPDATE_URL,
                headers=headers,
                json=payload,
                timeout=30.0
            )
            response.raise_for_status()
            return response.json()
    
    async def sync_from_api(self, prompt_id: Optional[str] = None) -> PromptVersion:
        """Fetch prompt from API and save as new local version."""
        api_data = await self.fetch_prompt_from_api(prompt_id)
        
        # Create new version from API data
        version = self.create_version(
            system_prompt=api_data.get("system_prompt", ""),
            instruction_template=api_data.get("instruction_template", ""),
            few_shot_examples=api_data.get("few_shot_examples", []),
            optimization_notes=f"Synced from API: {api_data.get('prompt_id', 'unknown')}",
        )
        
        console.print(f"[green]✓[/green] Synced prompt from API as version {version.version_number}")
        return version
    
    async def push_to_api(self, version_number: int, prompt_id: str) -> dict:
        """Push a local version to the remote API."""
        version = self.get_version(version_number)
        if not version:
            raise ValueError(f"Version {version_number} not found")
        
        result = await self.update_prompt_via_api(
            prompt_id=prompt_id,
            system_prompt=version.system_prompt,
            instruction_template=version.instruction_template,
            few_shot_examples=version.few_shot_examples,
            optimization_notes=version.optimization_notes,
        )
        
        console.print(f"[green]✓[/green] Pushed version {version_number} to API")
        return result
