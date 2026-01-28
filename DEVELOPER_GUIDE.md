# Developer Guide: Extending the Prompt Refinement Workflow

This guide provides detailed instructions for developers who want to modify or extend the workflow functionality.

## Table of Contents

1. [Code Architecture Deep Dive](#code-architecture-deep-dive)
2. [Adding New Features](#adding-new-features)
3. [Modifying Existing Components](#modifying-existing-components)
4. [Testing Your Changes](#testing-your-changes)
5. [Common Modification Patterns](#common-modification-patterns)

---

## Code Architecture Deep Dive

### Data Flow Diagram

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Test Cases    │     │  Prompt Version │     │   Agent API     │
│   (JSONL)       │     │     (JSON)      │     │   or Mock       │
└────────┬────────┘     └────────┬────────┘     └────────┬────────┘
         │                       │                       │
         ▼                       ▼                       ▼
    ┌────────────────────────────────────────────────────────┐
    │                  IterationController                    │
    │  ┌──────────────────────────────────────────────────┐  │
    │  │ run_evaluation()                                  │  │
    │  │   → AgentInvoker.invoke() for each test case     │  │
    │  │   → LLMJudge.evaluate() for each response        │  │
    │  │   → Aggregate results into EvalRunSummary        │  │
    │  └──────────────────────────────────────────────────┘  │
    │                          ▼                              │
    │  ┌──────────────────────────────────────────────────┐  │
    │  │ FailureAnalyzer.analyze_failures()                │  │
    │  │   → Categorize each failure                       │  │
    │  │   → Generate fix recommendations                  │  │
    │  └──────────────────────────────────────────────────┘  │
    │                          ▼                              │
    │  ┌──────────────────────────────────────────────────┐  │
    │  │ HumanReviewer.review_all_failures()               │  │
    │  │   → Interactive CLI for each failure              │  │
    │  │   → Collect annotations and refinement feedback   │  │
    │  └──────────────────────────────────────────────────┘  │
    │                          ▼                              │
    │  ┌──────────────────────────────────────────────────┐  │
    │  │ PromptOptimizer.create_optimized_version()        │  │
    │  │   → Generate improved prompt via GPT-4            │  │
    │  │   → Save with parent linkage                      │  │
    │  └──────────────────────────────────────────────────┘  │
    └────────────────────────────────────────────────────────┘
                               ▼
                    ┌─────────────────┐
                    │  New Prompt     │
                    │  Version (JSON) │
                    └─────────────────┘
```

### Module Dependencies

```
main.py (CLI)
    ├── config/settings.py
    ├── core/
    │   ├── models.py (Pydantic models - no dependencies)
    │   ├── prompt_manager.py (depends on: models, settings)
    │   ├── test_case_manager.py (depends on: models, settings)
    │   ├── agent_invoker.py (depends on: models, settings)
    │   └── iteration_controller.py (depends on: ALL core, graders, refinement, human_feedback)
    ├── graders/
    │   └── llm_judge.py (depends on: models, settings)
    ├── refinement/
    │   ├── failure_analyzer.py (depends on: models)
    │   └── prompt_optimizer.py (depends on: models, settings, prompt_manager)
    └── human_feedback/
        └── cli_reviewer.py (depends on: models, settings)
```

### Key Pydantic Models

```python
# core/models.py

class TestCase(BaseModel):
    id: str = Field(default_factory=generate_id)
    input_text: str                    # The question/input
    expected_output: str               # Ground truth answer
    tags: list[str] = []               # For filtering
    difficulty: str = "medium"         # easy/medium/hard
    metadata: dict = {}                # Custom fields

class PromptVersion(BaseModel):
    id: str = Field(default_factory=generate_id)
    version_number: int
    system_prompt: str                 # System message
    instruction_template: str          # Template with {input}
    few_shot_examples: list[dict] = [] # Optional examples
    parent_version_id: Optional[str]   # For lineage tracking
    pass_rate: Optional[float]         # From evaluation
    optimization_notes: str = ""       # What changed
    created_at: datetime

class GraderResult(BaseModel):
    scores: dict[str, float]           # {"correctness": 8.5, ...}
    aggregate_score: float             # Weighted average
    passed: bool                       # >= threshold
    feedback: str                      # LLM explanation
    details: dict = {}                 # Extra metadata

class EvaluationResult(BaseModel):
    test_case_id: str
    prompt_version_id: str
    input_text: str
    expected_output: str
    actual_output: str                 # What agent returned
    grader_result: GraderResult
    passed: bool
    latency_ms: int
    tokens_used: int

class HumanAnnotation(BaseModel):
    test_case_id: str
    run_id: str
    rating: str                        # good/bad/skip
    category: Optional[str]            # failure category
    notes: str = ""                    # Quick notes
    refinement_instruction: str = ""   # Detailed: how to fix
    ideal_output: str = ""             # Detailed: correct answer
    created_at: datetime

class RefinementFeedback(BaseModel):
    run_id: str
    general_instructions: str          # Overall guidance
    priority_issues: list[str]         # Top issues to fix
    style_guidance: str = ""           # Tone/format preferences
    constraints: list[str] = []        # Hard rules
    created_at: datetime
```

---

## Adding New Features

### Pattern 1: Adding a New CLI Command

**Example: Add a command to export evaluation results**

```python
# In main.py

@cli.command("export")
@click.option("--run-id", "-r", required=True, help="Evaluation run ID")
@click.option("--format", "-f", type=click.Choice(["json", "csv", "md"]), default="json")
@click.option("--output", "-o", required=True, help="Output file path")
def export_results(run_id, format, output):
    """Export evaluation results to a file."""
    controller = IterationController()
    
    # Load the run
    summary = controller.load_run_summary(run_id)
    if not summary:
        console.print(f"[red]Run {run_id} not found[/red]")
        return
    
    # Export based on format
    if format == "json":
        with open(output, "w") as f:
            json.dump(summary.model_dump(), f, indent=2, default=str)
    elif format == "csv":
        _export_csv(summary, output)
    elif format == "md":
        _export_markdown(summary, output)
    
    console.print(f"[green]✓[/green] Exported to {output}")

def _export_csv(summary, output):
    import csv
    with open(output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["test_case_id", "passed", "score", "input", "expected", "actual"])
        for r in summary.results:
            writer.writerow([
                r.test_case_id,
                r.passed,
                r.grader_result.aggregate_score,
                r.input_text[:100],
                r.expected_output[:100],
                r.actual_output[:100],
            ])

def _export_markdown(summary, output):
    with open(output, "w") as f:
        f.write(f"# Evaluation Run: {summary.run_id}\n\n")
        f.write(f"**Pass Rate:** {summary.pass_rate:.1%}\n")
        f.write(f"**Total Cases:** {summary.total_cases}\n\n")
        f.write("## Results\n\n")
        for r in summary.results:
            status = "✅" if r.passed else "❌"
            f.write(f"### {status} {r.test_case_id}\n")
            f.write(f"**Score:** {r.grader_result.aggregate_score:.1f}/10\n\n")
```

### Pattern 2: Adding a New Grading Criterion

**Example: Add "conciseness" as a grading criterion**

```python
# In graders/llm_judge.py

class LLMJudge:
    GRADING_CRITERIA = {
        "correctness": {
            "weight": 0.35,
            "description": "How factually accurate is the response compared to expected output?",
        },
        "format_compliance": {
            "weight": 0.20,
            "description": "Does the response follow the expected format and structure?",
        },
        "behavior_alignment": {
            "weight": 0.25,
            "description": "Does the response match the intended agent behavior and tone?",
        },
        # NEW CRITERION
        "conciseness": {
            "weight": 0.20,
            "description": "Is the response appropriately concise without unnecessary verbosity?",
        },
    }
    
    # Update the system prompt to include the new criterion
    JUDGE_SYSTEM_PROMPT = """You are an expert evaluator...
    
    Evaluate on these criteria (0-10 each):
    1. correctness: ...
    2. format_compliance: ...
    3. behavior_alignment: ...
    4. conciseness: Is the response appropriately brief? Penalize excessive wordiness.
    
    Return JSON:
    {
        "scores": {"correctness": X, "format_compliance": X, "behavior_alignment": X, "conciseness": X},
        "feedback": "...",
        "failure_category": "..."
    }
    """
```

### Pattern 3: Adding a New Failure Category

**Example: Add "hallucination" category**

```python
# In refinement/failure_analyzer.py

class FailureAnalyzer:
    FAILURE_CATEGORIES = {
        "invalid_format": {
            "description": "Response doesn't match expected format/structure",
            "fix_strategy": "add_format_instructions",
            "keywords": ["format", "structure", "missing", "wrong format"],
        },
        "factual_error": {
            "description": "Response contains incorrect information",
            "fix_strategy": "add_context_grounding",
            "keywords": ["incorrect", "wrong", "error", "inaccurate"],
        },
        # ... existing categories ...
        
        # NEW CATEGORY
        "hallucination": {
            "description": "Response invents information not supported by context",
            "fix_strategy": "add_grounding_constraints",
            "keywords": ["made up", "invented", "fabricated", "not mentioned", "hallucinated"],
        },
    }
    
    # Add new fix strategy
    FIX_STRATEGIES = {
        # ... existing strategies ...
        
        "add_grounding_constraints": {
            "description": "Add explicit instructions to only use provided context",
            "prompt_modification": "Add: 'Only use information explicitly provided. If unsure, say so.'",
        },
    }
```

### Pattern 4: Adding a New Agent API Integration

**Example: Support for a different agent framework**

```python
# In core/agent_invoker.py

class AgentInvoker:
    def __init__(self):
        self.api_url = settings.AGENT_API_URL
        self.api_key = settings.AGENT_API_KEY
        
        # NEW: Support for LangChain agents
        self.langchain_endpoint = os.getenv("LANGCHAIN_ENDPOINT", "")
        self.agent_type = self._detect_agent_type()
    
    def _detect_agent_type(self) -> str:
        if self.langchain_endpoint:
            return "langchain"
        elif self.api_key and self.api_key not in ["", "your-agent-api-key"]:
            return "custom"
        else:
            return "mock"
    
    async def invoke(self, test_case: TestCase, prompt_version: PromptVersion) -> dict:
        if self.agent_type == "mock":
            return await self._invoke_mock(test_case, prompt_version)
        elif self.agent_type == "langchain":
            return await self._invoke_langchain(test_case, prompt_version)
        else:
            return await self._invoke_custom(test_case, prompt_version)
    
    async def _invoke_langchain(self, test_case: TestCase, prompt_version: PromptVersion) -> dict:
        """Call a LangChain agent endpoint."""
        import time
        start = time.time()
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.langchain_endpoint,
                json={
                    "input": test_case.input_text,
                    "config": {
                        "system_prompt": prompt_version.system_prompt,
                    },
                },
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=60.0,
            )
            response.raise_for_status()
            data = response.json()
        
        latency = int((time.time() - start) * 1000)
        
        return {
            "output": data.get("output", ""),
            "latency_ms": latency,
            "tokens_used": data.get("usage", {}).get("total_tokens", 0),
        }
```

### Pattern 5: Adding Custom Metrics/Analytics

**Example: Track token usage over time**

```python
# Create new file: analytics/usage_tracker.py

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import json
from typing import Optional
from core.models import EvalRunSummary


@dataclass
class UsageMetrics:
    run_id: str
    timestamp: datetime
    total_tokens: int
    avg_latency_ms: float
    pass_rate: float
    prompt_version: int


class UsageTracker:
    """Track resource usage across evaluation runs."""
    
    def __init__(self, data_dir: Path = Path("data/analytics")):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file = self.data_dir / "usage_metrics.jsonl"
    
    def record_run(self, summary: EvalRunSummary, prompt_version: int):
        """Record metrics from an evaluation run."""
        total_tokens = sum(r.tokens_used for r in summary.results)
        avg_latency = sum(r.latency_ms for r in summary.results) / len(summary.results)
        
        metrics = UsageMetrics(
            run_id=summary.run_id,
            timestamp=datetime.now(),
            total_tokens=total_tokens,
            avg_latency_ms=avg_latency,
            pass_rate=summary.pass_rate,
            prompt_version=prompt_version,
        )
        
        with open(self.metrics_file, "a") as f:
            f.write(json.dumps({
                "run_id": metrics.run_id,
                "timestamp": metrics.timestamp.isoformat(),
                "total_tokens": metrics.total_tokens,
                "avg_latency_ms": metrics.avg_latency_ms,
                "pass_rate": metrics.pass_rate,
                "prompt_version": metrics.prompt_version,
            }) + "\n")
    
    def get_usage_report(self) -> dict:
        """Generate usage report."""
        if not self.metrics_file.exists():
            return {"error": "No data"}
        
        records = []
        with open(self.metrics_file, "r") as f:
            for line in f:
                records.append(json.loads(line))
        
        return {
            "total_runs": len(records),
            "total_tokens_used": sum(r["total_tokens"] for r in records),
            "avg_pass_rate": sum(r["pass_rate"] for r in records) / len(records),
            "avg_latency_ms": sum(r["avg_latency_ms"] for r in records) / len(records),
        }
```

Then add to CLI:

```python
# In main.py

@cli.command("usage")
def show_usage():
    """Show resource usage analytics."""
    from analytics.usage_tracker import UsageTracker
    tracker = UsageTracker()
    report = tracker.get_usage_report()
    
    table = Table(title="Usage Report")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="yellow")
    
    for key, value in report.items():
        if isinstance(value, float):
            table.add_row(key, f"{value:.2f}")
        else:
            table.add_row(key, str(value))
    
    console.print(table)
```

---

## Modifying Existing Components

### Modifying the LLM Judge Prompt

The judge uses a specific prompt template. To modify evaluation behavior:

```python
# In graders/llm_judge.py

class LLMJudge:
    # Modify this prompt to change how evaluation works
    JUDGE_SYSTEM_PROMPT = """You are an expert evaluator for AI agent responses.
    
    EVALUATION CONTEXT:
    - You're evaluating a conversational AI agent
    - Focus on helpfulness and accuracy
    - Be strict about factual correctness
    
    SCORING GUIDELINES:
    - 9-10: Excellent, exceeds expectations
    - 7-8: Good, meets requirements
    - 5-6: Acceptable, minor issues
    - 3-4: Poor, significant issues
    - 1-2: Unacceptable, major failures
    
    CRITERIA (score each 0-10):
    1. correctness: Factual accuracy vs expected output
    2. format_compliance: Follows expected structure
    3. behavior_alignment: Matches intended behavior
    
    OUTPUT FORMAT (JSON):
    {
        "scores": {"correctness": X, "format_compliance": X, "behavior_alignment": X},
        "feedback": "Detailed explanation of the evaluation",
        "failure_category": "none|invalid_format|factual_error|behavior_drift|edge_case|verbosity|other"
    }
    """
```

### Modifying the Optimizer Prompt

To change how prompts are optimized:

```python
# In refinement/prompt_optimizer.py

class PromptOptimizer:
    OPTIMIZER_SYSTEM_PROMPT = """You are an expert prompt engineer specializing in AI agent optimization.
    
    YOUR TASK:
    Analyze the failures and improve the prompt to address them.
    
    PRINCIPLES:
    1. Be specific - vague instructions lead to inconsistent behavior
    2. Add examples when behavior is unclear
    3. Use structured output formats when possible
    4. Address the most common failure patterns first
    5. Preserve what works - don't over-optimize
    
    OUTPUT FORMAT (JSON):
    {
        "improved_system_prompt": "...",
        "instruction_template": "...",
        "few_shot_examples": [...],
        "changes_made": ["list of changes"],
        "optimization_notes": "explanation"
    }
    """
```

### Modifying Convergence Logic

To change when the loop stops:

```python
# In core/iteration_controller.py

def should_continue(self, history: list[EvalRunSummary]) -> tuple[bool, str]:
    """Customize convergence criteria."""
    if not history:
        return True, "Starting first iteration"
    
    latest = history[-1]
    
    # Custom: Stop if average score is very high
    if latest.avg_score >= 9.0:
        return False, f"Excellent average score ({latest.avg_score:.1f})"
    
    # Custom: Stop if no failures in critical categories
    critical = ["factual_error", "behavior_drift"]
    critical_failures = sum(
        latest.failure_breakdown.get(cat, 0) for cat in critical
    )
    if critical_failures == 0 and latest.pass_rate >= 0.8:
        return False, "No critical failures remaining"
    
    # Original checks...
    if len(history) >= self.max_iterations:
        return False, f"Reached maximum iterations"
    
    if latest.pass_rate >= self.convergence_threshold:
        return False, f"Reached target pass rate"
    
    return True, "Continuing refinement"
```

---

## Testing Your Changes

### Manual Testing

```bash
# Test specific component
python -c "from graders import LLMJudge; j = LLMJudge(); print(j.GRADING_CRITERIA)"

# Test CLI command
python main.py --help
python main.py test list

# Run a quick evaluation
python main.py eval run --prompt-version 1
```

### Creating Unit Tests

```python
# Create tests/test_analyzer.py

import pytest
from refinement.failure_analyzer import FailureAnalyzer
from core.models import EvaluationResult, GraderResult


def test_categorize_format_failure():
    analyzer = FailureAnalyzer()
    
    result = EvaluationResult(
        test_case_id="tc_001",
        prompt_version_id="pv_001",
        input_text="Format as JSON",
        expected_output='{"key": "value"}',
        actual_output="key: value",  # Wrong format
        grader_result=GraderResult(
            scores={"correctness": 8, "format_compliance": 2, "behavior_alignment": 7},
            aggregate_score=5.5,
            passed=False,
            feedback="Wrong format",
            details={"failure_category": "invalid_format"},
        ),
        passed=False,
        latency_ms=100,
        tokens_used=50,
    )
    
    analysis = analyzer.analyze_failures([result])
    assert "invalid_format" in analysis["failure_breakdown"]


def test_fix_recommendations():
    analyzer = FailureAnalyzer()
    # ... test that recommendations are generated correctly
```

Run tests:
```bash
pytest tests/ -v
```

---

## Common Modification Patterns

### Quick Reference: Where to Modify

| I want to... | Modify this file |
|--------------|------------------|
| Add a CLI command | `main.py` |
| Change grading criteria | `graders/llm_judge.py` |
| Add failure category | `refinement/failure_analyzer.py` |
| Change optimization logic | `refinement/prompt_optimizer.py` |
| Add review mode | `human_feedback/cli_reviewer.py` |
| Change agent API | `core/agent_invoker.py` |
| Modify convergence | `core/iteration_controller.py` |
| Add data model | `core/models.py` |
| Change settings | `config/settings.py` |

### Quick Reference: Common Imports

```python
# Models
from core.models import (
    TestCase, PromptVersion, EvaluationResult,
    EvalRunSummary, GraderResult, HumanAnnotation,
    RefinementFeedback
)

# Managers
from core.prompt_manager import PromptManager
from core.test_case_manager import TestCaseManager
from core.agent_invoker import AgentInvoker
from core.iteration_controller import IterationController

# Graders
from graders import LLMJudge

# Refinement
from refinement import FailureAnalyzer, PromptOptimizer

# Human Feedback
from human_feedback import HumanReviewer

# Config
from config import settings

# Rich (for CLI output)
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress
console = Console()
```

---

## Need Help?

1. Check the [README.md](README.md) for usage documentation
2. Review the existing code patterns in similar components
3. Run with verbose output to debug issues
4. Check `data/results/` for raw evaluation data

Happy extending! 🚀
