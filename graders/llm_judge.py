"""GPT-4 LLM-as-Judge grader for evaluating agent outputs."""
import json
from openai import OpenAI
from rich.console import Console

from core.models import GraderResult
from config import settings

console = Console()


class LLMJudge:
    """Uses GPT-4 to evaluate agent outputs against expected outputs."""
    
    JUDGE_SYSTEM_PROMPT = """You are an expert evaluator assessing AI agent responses.
Your task is to compare the ACTUAL OUTPUT from an agent against the EXPECTED OUTPUT.

Evaluate on three dimensions (each scored 0-10):

1. **Correctness** (0-10): Does the actual output convey the same information/meaning as expected?
   - 10: Perfect match in meaning
   - 7-9: Correct with minor differences
   - 4-6: Partially correct
   - 1-3: Mostly incorrect
   - 0: Completely wrong or opposite

2. **Format Compliance** (0-10): Is the output in the expected format?
   - 10: Perfect format match
   - 7-9: Good format with minor issues
   - 4-6: Format issues but readable
   - 1-3: Major format problems
   - 0: Completely wrong format

3. **Behavior Alignment** (0-10): Does the output follow expected behavior patterns?
   - Consider: tone, verbosity, structure, following instructions
   - 10: Perfect alignment
   - 7-9: Good with minor deviations
   - 4-6: Some behavioral issues
   - 1-3: Major behavioral problems
   - 0: Completely misaligned

Respond ONLY with a JSON object in this exact format:
{
    "correctness": <0-10>,
    "format_compliance": <0-10>,
    "behavior_alignment": <0-10>,
    "aggregate_score": <average of above three>,
    "passed": <true if aggregate >= 7>,
    "failure_category": "<one of: none, invalid_format, factual_error, behavior_drift, edge_case, verbosity, other>",
    "feedback": "<Brief explanation of the evaluation>"
}"""

    JUDGE_USER_TEMPLATE = """Evaluate the following agent output:

**INPUT/QUESTION:**
{input_text}

**EXPECTED OUTPUT:**
{expected_output}

**ACTUAL OUTPUT:**
{actual_output}

Provide your evaluation as JSON."""

    def __init__(self):
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = settings.JUDGE_MODEL
        self.pass_threshold = settings.PASS_SCORE_THRESHOLD
    
    def evaluate(
        self,
        input_text: str,
        expected_output: str,
        actual_output: str,
    ) -> GraderResult:
        """
        Evaluate an agent output using GPT-4 as judge.
        
        Args:
            input_text: The original question/input
            expected_output: The ground truth expected output
            actual_output: The agent's actual response
            
        Returns:
            GraderResult with scores and feedback
        """
        try:
            # Build the evaluation prompt
            user_message = self.JUDGE_USER_TEMPLATE.format(
                input_text=input_text,
                expected_output=expected_output,
                actual_output=actual_output,
            )
            
            # Call GPT-4
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.1,  # Low temperature for consistent evaluation
                response_format={"type": "json_object"},
            )
            
            # Parse the response
            result_text = response.choices[0].message.content
            result_data = json.loads(result_text)
            
            return GraderResult(
                grader_name="gpt4_judge",
                grader_type="llm_judge",
                scores={
                    "correctness": result_data.get("correctness", 0),
                    "format_compliance": result_data.get("format_compliance", 0),
                    "behavior_alignment": result_data.get("behavior_alignment", 0),
                },
                aggregate_score=result_data.get("aggregate_score", 0),
                passed=result_data.get("passed", False),
                details={
                    "failure_category": result_data.get("failure_category", "other"),
                    "raw_response": result_data,
                },
                feedback=result_data.get("feedback", ""),
            )
            
        except json.JSONDecodeError as e:
            console.print(f"[red]Error parsing judge response: {e}[/red]")
            return GraderResult(
                grader_name="gpt4_judge",
                grader_type="llm_judge",
                scores={"correctness": 0, "format_compliance": 0, "behavior_alignment": 0},
                aggregate_score=0,
                passed=False,
                details={"error": str(e), "failure_category": "other"},
                feedback=f"Failed to parse judge response: {e}",
            )
        except Exception as e:
            console.print(f"[red]Error calling GPT-4 judge: {e}[/red]")
            return GraderResult(
                grader_name="gpt4_judge",
                grader_type="llm_judge",
                scores={"correctness": 0, "format_compliance": 0, "behavior_alignment": 0},
                aggregate_score=0,
                passed=False,
                details={"error": str(e), "failure_category": "other"},
                feedback=f"Judge evaluation failed: {e}",
            )
    
    def evaluate_batch(
        self,
        evaluations: list[dict],
        progress_callback=None,
    ) -> list[GraderResult]:
        """
        Evaluate multiple outputs.
        
        Args:
            evaluations: List of dicts with input_text, expected_output, actual_output
            progress_callback: Optional callback(current, total, test_id, passed)
            
        Returns:
            List of GraderResults
        """
        results = []
        
        for i, eval_data in enumerate(evaluations):
            result = self.evaluate(
                input_text=eval_data["input_text"],
                expected_output=eval_data["expected_output"],
                actual_output=eval_data["actual_output"],
            )
            results.append(result)
            
            if progress_callback:
                progress_callback(i + 1, len(evaluations), eval_data.get("test_id", ""), result.passed)
        
        return results
