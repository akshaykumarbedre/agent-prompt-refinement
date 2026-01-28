"""Agent invoker for calling the agent API to get responses."""
import time
import httpx
from typing import Optional
from openai import OpenAI
from rich.console import Console

from .models import TestCase, PromptVersion
from config import settings

console = Console()


class AgentInvoker:
    """Invokes the agent API with test cases and prompts."""
    
    def __init__(self, use_mock: bool = False):
        self.api_url = settings.AGENT_API_URL
        self.api_key = settings.AGENT_API_KEY
        
        # Determine if we should use mock mode
        is_placeholder_key = self.api_key in ["", "your-agent-api-key"]
        self.use_mock = use_mock or self.api_url == "mock" or is_placeholder_key
        
        # Always create OpenAI client for mock mode fallback
        self.openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
    
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
        # Use mock mode if configured or API unavailable
        if self.use_mock or not self.api_key or self.api_key == "your-agent-api-key":
            return await self._invoke_mock(test_case, prompt_version)
        
        return await self._invoke_api(test_case, prompt_version)
    
    async def _invoke_mock(
        self,
        test_case: TestCase,
        prompt_version: Optional[PromptVersion] = None,
    ) -> dict:
        """Mock agent using OpenAI directly for testing."""
        start_time = time.time()
        
        try:
            system_prompt = "You are a helpful customer service assistant."
            if prompt_version:
                system_prompt = prompt_version.system_prompt
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": test_case.input_text},
            ]
            
            # Add few-shot examples if present
            if prompt_version and prompt_version.few_shot_examples:
                example_messages = []
                for ex in prompt_version.few_shot_examples:
                    example_messages.append({"role": "user", "content": ex.get("input", "")})
                    example_messages.append({"role": "assistant", "content": ex.get("output", "")})
                messages = [messages[0]] + example_messages + [messages[1]]
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.7,
                max_tokens=500,
            )
            
            latency_ms = int((time.time() - start_time) * 1000)
            output = response.choices[0].message.content
            tokens_used = response.usage.total_tokens if response.usage else 0
            
            return {
                "output": output,
                "latency_ms": latency_ms,
                "tokens_used": tokens_used,
                "success": True,
                "error": None,
                "raw_response": {"model": "gpt-4o-mini", "mock": True},
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
    
    async def _invoke_api(
        self,
        test_case: TestCase,
        prompt_version: Optional[PromptVersion] = None,
    ) -> dict:
        """Invoke the actual agent API."""
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
