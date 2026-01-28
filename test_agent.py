"""Quick test script for the agent invoker."""
import asyncio
from core import AgentInvoker, TestCaseManager, PromptManager

async def test():
    invoker = AgentInvoker()
    tm = TestCaseManager()
    pm = PromptManager()
    
    cases = tm.load_test_cases()
    prompt = pm.get_version(1)
    
    print(f"Using mock mode: {invoker.use_mock}")
    print(f"API Key set: {bool(invoker.api_key)}")
    print(f"Testing with: {cases[0].input_text}")
    
    result = await invoker.invoke(cases[0], prompt)
    print(f"Success: {result['success']}")
    print(f"Output: {result['output'][:200] if result['output'] else 'EMPTY'}")
    print(f"Error: {result['error']}")

asyncio.run(test())
