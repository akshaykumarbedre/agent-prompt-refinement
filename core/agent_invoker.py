"""Agent invoker for calling the agent API to get responses."""
import time
import httpx
from typing import Optional
from rich.console import Console

from .models import TestCase, PromptVersion
from config import settings

console = Console()


class AgentInvoker:
    """Invokes the agent API with test cases and prompts."""
    
    def __init__(self):
        self.api_url = settings.AGENT_API_URL
        self.api_key = settings.AGENT_API_KEY
    
    def _build_headers(self) -> dict:
        """Build request headers."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
    
    def _apply_template(self, template: str, variables: dict) -> str:
        """Apply variables to a template string."""
        result = template
        for key, value in variables.items():
            result = result.replace(f"{{{{{key}}}}}", str(value))
        return result
    
    async def invoke(
        self,
        test_case: TestCase,
        prompt_version: Optional[PromptVersion] = None,
    ) -> dict:
        """
        Invoke the agent API with a test case.
        
        API Expected Request:
        {
            "input": "User question/input text",
            "system_prompt": "Optional system prompt override",
            "context": {...}  # Optional additional context
        }
        
        API Expected Response:
        {
            "output": "Agent's response",
            "tokens_used": 123,
            "metadata": {...}
        }
        
        Returns:
            dict with keys: output, latency_ms, tokens_used, success, error
        """
        start_time = time.time()
        
        try:
            # Build request payload
            payload = {
                "input": test_case.input_text,
            }
            
            # Add prompt if provided
            if prompt_version:
                payload["system_prompt"] = prompt_version.system_prompt
                
                # Apply instruction template if exists
                if prompt_version.instruction_template:
                    payload["instruction"] = self._apply_template(
                        prompt_version.instruction_template,
                        {"user_input": test_case.input_text}
                    )
                
                # Add few-shot examples if present
                if prompt_version.few_shot_examples:
                    payload["examples"] = prompt_version.few_shot_examples
            
            # Add test case metadata as context
            if test_case.metadata:
                payload["context"] = test_case.metadata
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.api_url,
                    headers=self._build_headers(),
                    json=payload,
                    timeout=60.0
                )
                response.raise_for_status()
                result = response.json()
            
            latency_ms = int((time.time() - start_time) * 1000)
            
            return {
                "output": result.get("output", result.get("response", "")),
                "latency_ms": latency_ms,
                "tokens_used": result.get("tokens_used", result.get("usage", {}).get("total_tokens", 0)),
                "success": True,
                "error": None,
                "raw_response": result,
            }
            
        except httpx.HTTPStatusError as e:
            latency_ms = int((time.time() - start_time) * 1000)
            return {
                "output": "",
                "latency_ms": latency_ms,
                "tokens_used": 0,
                "success": False,
                "error": f"HTTP {e.response.status_code}: {e.response.text}",
                "raw_response": None,
            }
        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            return {
                "output": "",
                "latency_ms": latency_ms,
                "tokens_used": 0,
                "success": False,
                "error": str(e),
                "raw_response": None,
            }
    
    async def invoke_batch(
        self,
        test_cases: list[TestCase],
        prompt_version: Optional[PromptVersion] = None,
        progress_callback=None,
    ) -> list[dict]:
        """Invoke agent for multiple test cases sequentially."""
        results = []
        
        for i, test_case in enumerate(test_cases):
            result = await self.invoke(test_case, prompt_version)
            result["test_case_id"] = test_case.id
            results.append(result)
            
            if progress_callback:
                progress_callback(i + 1, len(test_cases), test_case.id, result["success"])
        
        return results
    
    def invoke_sync(
        self,
        test_case: TestCase,
        prompt_version: Optional[PromptVersion] = None,
    ) -> dict:
        """Synchronous version of invoke for simpler use cases."""
        import asyncio
        return asyncio.run(self.invoke(test_case, prompt_version))
