# Agent Prompt Refinement Workflow - Copilot Instructions

## Architecture Overview

This is an **iterative prompt engineering tool** that refines AI agent prompts through a feedback loop: **Measure → Analyze → Human Review → Improve → Repeat**.

### Core Components
- **`main.py`** - Click-based CLI entrypoint with command groups: `test`, `prompt`, `eval`, `review`, `optimize`, `loop`
- **`core/`** - Domain models (Pydantic), prompt versioning, test case storage, agent invocation, iteration orchestration
- **`graders/`** - LLM-as-Judge using GPT-4 to score outputs on correctness/format/behavior (0-10 scale)
- **`refinement/`** - Failure analysis (categorization) and GPT-4-powered prompt optimization
- **`human_feedback/`** - Interactive CLI for reviewing failures with rich NLP feedback collection

### Data Flow
1. Test cases stored as JSONL in `data/test_cases/` → Agent invoked via API or mock mode
2. Responses graded by `LLMJudge` → Results saved to `data/results/run_<id>.json`
3. Failures analyzed into categories → Human annotations saved to `data/results/annotations/`
4. `PromptOptimizer` generates improved prompt → New version saved to `data/prompts/versions/v00N.json`

## Key Patterns

### Pydantic Models (core/models.py)
All data structures use Pydantic v2 with `model_dump()` for serialization:
```python
class TestCase(BaseModel):
    id: str = Field(default_factory=generate_id)
    input_text: str
    expected_output: str
    tags: list[str] = Field(default_factory=list)
```

### JSONL Storage Pattern
Test cases and annotations use line-delimited JSON with `{"item": {...}}` wrapper for test cases:
```jsonl
{"item": {"id": "tc_001", "input_text": "...", "expected_output": "...", "tags": [...], "difficulty": "medium"}}
```

### Prompt Versions
Versioned prompts in `data/prompts/versions/v00N.json` with lineage tracking via `parent_version_id`. Key fields: `system_prompt`, `instruction_template`, `few_shot_examples`, `pass_rate`.

### Failure Categories
Six categories defined in `refinement/failure_analyzer.py`, each with a fix strategy:
- `invalid_format` → `add_format_instructions`
- `factual_error` → `add_context_grounding`
- `behavior_drift` → `strengthen_instructions`
- `edge_case` → `add_edge_case_handling`
- `verbosity` → `add_length_constraints`
- `other` → `manual_review`

## Developer Workflow

### Running the Tool
```bash
# Install dependencies
pip install -r requirements.txt

# Configure .env with OPENAI_API_KEY (required) and agent API settings
cp .env.example .env

# Add test cases
python main.py test add --input "question" --expected "answer"

# Run evaluation
python main.py eval run --prompt-version 1

# Review failures interactively
python main.py review --run-id <8-char-id>

# Full automated loop (skips human review)
python main.py loop start --skip-review
```

### Mock Mode
When `AGENT_API_KEY` is empty or placeholder, `AgentInvoker` automatically uses OpenAI directly as a mock agent—no external API needed for testing.

### Configuration
Settings loaded from environment via `config/settings.py`:
- `JUDGE_MODEL`: Defaults to `gpt-4o-mini` (requires JSON mode support)
- `PASS_SCORE_THRESHOLD`: Default 7.0—aggregate score must meet this to pass
- `CONVERGENCE_THRESHOLD`: Stop loop at 95% pass rate
- `MAX_ITERATIONS`: Default 10 iterations max

## Conventions

### ID Generation
8-character UUIDs via `str(uuid.uuid4())[:8]` used for test cases, runs, prompts.

### Rich Console Output
All CLI output uses `rich` library—tables, panels, progress bars. Import as:
```python
from rich.console import Console
console = Console()
```

### Async Pattern
Agent invocation is async (`async def invoke()`). The iteration controller runs via `asyncio.run()`.

### Error Handling in Graders
`LLMJudge.evaluate()` catches exceptions and returns a failed `GraderResult` with zero scores rather than raising—ensures evaluation runs complete.
