# Agent Prompt Refinement Workflow

An automated prompt engineering tool that refines AI agent prompts through an iterative feedback loop: **Measure → Analyze → Human Review → Improve → Repeat**.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [CLI Commands Reference](#cli-commands-reference)
- [Configuration](#configuration)
- [Data Formats](#data-formats)
- [Extending the Workflow](#extending-the-workflow)
- [Troubleshooting](#troubleshooting)

---

## Overview

This workflow automates the process of improving AI agent prompts by:

1. **Running evaluations** against a test dataset
2. **Grading responses** using GPT-4 as an LLM judge
3. **Analyzing failures** to identify patterns
4. **Collecting human feedback** (quick ratings or detailed NLP)
5. **Generating optimized prompts** with lineage tracking
6. **Repeating** until convergence threshold is met

### Key Features

| Feature | Description |
|---------|-------------|
| **LLM-as-Judge** | GPT-4 evaluates correctness, format, and behavior (0-10 scale) |
| **Prompt Versioning** | All versions tracked with parent lineage |
| **Failure Categorization** | Automatic categorization into 6 failure types |
| **Human Feedback Modes** | Quick (g/b/s), Standard, or Detailed (NLP) review |
| **Mock Agent Mode** | Test without external API using OpenAI directly |
| **Convergence Detection** | Auto-stops at target pass rate or plateau |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    REFINEMENT LOOP                              │
│                                                                 │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │ MEASURE  │───▶│ ANALYZE  │───▶│  REVIEW  │───▶│ IMPROVE  │  │
│  │          │    │          │    │          │    │          │  │
│  │ Run eval │    │ Identify │    │ Human    │    │ Generate │  │
│  │ with     │    │ failure  │    │ feedback │    │ optimized│  │
│  │ GPT-4    │    │ patterns │    │ (optional│    │ prompt   │  │
│  │ judge    │    │          │    │          │    │          │  │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘  │
│       ▲                                               │         │
│       └───────────────────────────────────────────────┘         │
│                        (repeat until convergence)               │
└─────────────────────────────────────────────────────────────────┘
```

### Directory Structure

```
workflow/
├── main.py                    # CLI entrypoint
├── config/
│   └── settings.py            # Environment configuration
├── core/
│   ├── models.py              # Pydantic data models
│   ├── prompt_manager.py      # Prompt CRUD + versioning
│   ├── test_case_manager.py   # Test case storage
│   ├── agent_invoker.py       # Agent API + mock mode
│   └── iteration_controller.py # Main loop orchestration
├── graders/
│   └── llm_judge.py           # GPT-4 evaluation
├── refinement/
│   ├── failure_analyzer.py    # Failure categorization
│   └── prompt_optimizer.py    # GPT-4 prompt generation
├── human_feedback/
│   └── cli_reviewer.py        # Interactive review CLI
└── data/
    ├── test_cases/            # JSONL test datasets
    ├── prompts/versions/      # JSON prompt versions
    └── results/               # Evaluation run results
```

### Core Components

| Component | File | Purpose |
|-----------|------|---------|
| **IterationController** | `core/iteration_controller.py` | Orchestrates the full refinement loop |
| **AgentInvoker** | `core/agent_invoker.py` | Calls agent API or uses mock mode |
| **LLMJudge** | `graders/llm_judge.py` | GPT-4 grading with 3 criteria |
| **FailureAnalyzer** | `refinement/failure_analyzer.py` | Categorizes failures, suggests fixes |
| **PromptOptimizer** | `refinement/prompt_optimizer.py` | Generates improved prompts |
| **HumanReviewer** | `human_feedback/cli_reviewer.py` | Interactive feedback collection |
| **PromptManager** | `core/prompt_manager.py` | Version control for prompts |
| **TestCaseManager** | `core/test_case_manager.py` | Test case CRUD operations |

---

## Installation

### Prerequisites

- Python 3.10+
- OpenAI API key

### Setup

```bash
# Clone or navigate to the workflow directory
cd workflow

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys
```

### Environment Variables

```bash
# Required
OPENAI_API_KEY=sk-your-openai-key

# Optional - Agent API (leave empty for mock mode)
AGENT_API_URL=http://localhost:8000/agent/invoke
AGENT_API_KEY=your-agent-api-key

# Optional - Remote prompt sync
PROMPT_API_GET_URL=http://localhost:8000/prompts/{id}
PROMPT_API_UPDATE_URL=http://localhost:8000/prompts/{id}

# Tuning (optional)
JUDGE_MODEL=gpt-4o-mini
PASS_SCORE_THRESHOLD=7.0
CONVERGENCE_THRESHOLD=0.95
MAX_ITERATIONS=10
```

---

## Quick Start

### 1. Add Test Cases

```bash
# Add individual test case
python main.py test add \
  --input "What is the capital of France?" \
  --expected "Paris"

# Or import from JSONL file
python main.py test import --file my_tests.jsonl
```

### 2. Create Initial Prompt

```bash
python main.py prompt create \
  --system "You are a helpful assistant." \
  --template "Answer the following question: {input}"
```

### 3. Run Evaluation

```bash
python main.py eval run --prompt-version 1
```

### 4. Review Failures

```bash
# Quick review (good/bad/skip)
python main.py review --run-id <run-id> --quick

# Detailed review with NLP feedback
python main.py review --run-id <run-id> --detailed
```

### 5. Generate Optimized Prompt

```bash
python main.py optimize --from-version 1 --run-id <run-id>
```

### 6. Or Run Full Automated Loop

```bash
# With human review
python main.py loop start

# Skip human review (fully automated)
python main.py loop start --skip-review
```

---

## CLI Commands Reference

### Test Case Management

```bash
# Add a test case
python main.py test add \
  --input "Question text" \
  --expected "Expected answer" \
  --tags "tag1,tag2" \
  --difficulty "medium"

# List all test cases
python main.py test list

# Import from JSONL file
python main.py test import --file path/to/tests.jsonl
```

### Prompt Management

```bash
# Create new prompt version
python main.py prompt create \
  --system "System prompt text" \
  --template "Instruction template with {input}"

# List all versions
python main.py prompt list

# Show specific version
python main.py prompt show --version 1

# View version lineage
python main.py prompt lineage --version 3
```

### Evaluation

```bash
# Run evaluation
python main.py eval run \
  --prompt-version 1 \
  --dataset my_dataset

# Show evaluation results
python main.py eval show --run-id <8-char-id>
```

### Human Review

```bash
# Quick review (g=good, b=bad, s=skip)
python main.py review --run-id <id> --quick

# Standard review (rating + optional notes)
python main.py review --run-id <id>

# Detailed review with full NLP feedback
python main.py review --run-id <id> --detailed
```

#### Detailed Review Mode

In detailed mode, for each failure you can provide:
- **Detailed critique**: Multi-line NLP explanation of what went wrong
- **Ideal output**: What the response should have been
- **Refinement instruction**: Specific guidance for fixing this type of error

After reviewing all failures, you provide overall refinement feedback:
- **General instructions**: High-level guidance for the optimizer
- **Priority issues**: Comma-separated list of top issues to fix
- **Style guidance**: Desired tone, format, verbosity
- **Constraints**: Hard rules the prompt must follow

### Optimization

```bash
# Generate optimized prompt from failures
python main.py optimize \
  --from-version 1 \
  --run-id <id>

# Optimize and push to remote API
python main.py optimize \
  --from-version 1 \
  --run-id <id> \
  --push remote-prompt-id
```

### Full Refinement Loop

```bash
# Start loop with defaults
python main.py loop start

# With options
python main.py loop start \
  --prompt-version 1 \
  --dataset my_tests \
  --skip-review \
  --push remote-prompt-id
```

---

## Configuration

### Settings Reference (`config/settings.py`)

| Setting | Default | Description |
|---------|---------|-------------|
| `JUDGE_MODEL` | `gpt-4o-mini` | Model for LLM judge (must support JSON mode) |
| `OPTIMIZER_MODEL` | `gpt-4o-mini` | Model for prompt optimization |
| `PASS_SCORE_THRESHOLD` | `7.0` | Minimum aggregate score to pass (0-10) |
| `CONVERGENCE_THRESHOLD` | `0.95` | Target pass rate to stop loop |
| `MAX_ITERATIONS` | `10` | Maximum refinement iterations |
| `IMPROVEMENT_MIN` | `0.02` | Minimum improvement to continue |
| `PATIENCE` | `3` | Iterations without improvement before stopping |

### Grading Criteria

The LLM judge evaluates on three criteria (each 0-10):

| Criterion | Weight | Description |
|-----------|--------|-------------|
| `correctness` | 40% | Factual accuracy vs expected output |
| `format_compliance` | 30% | Follows expected format/structure |
| `behavior_alignment` | 30% | Matches intended agent behavior |

**Aggregate Score** = weighted average of all three criteria

---

## Data Formats

### Test Case (JSONL)

```jsonl
{"item": {"id": "tc_001", "input_text": "What is 2+2?", "expected_output": "4", "tags": ["math"], "difficulty": "easy"}}
{"item": {"id": "tc_002", "input_text": "Explain gravity", "expected_output": "Gravity is...", "tags": ["science"], "difficulty": "medium"}}
```

### Prompt Version (JSON)

```json
{
  "id": "pv_abc123",
  "version_number": 1,
  "system_prompt": "You are a helpful assistant.",
  "instruction_template": "Answer: {input}",
  "few_shot_examples": [],
  "parent_version_id": null,
  "pass_rate": null,
  "optimization_notes": "Initial version",
  "created_at": "2025-01-28T10:00:00"
}
```

### Evaluation Result (JSON)

```json
{
  "run_id": "abc12345",
  "prompt_version_id": "pv_abc123",
  "total_cases": 10,
  "passed": 8,
  "failed": 2,
  "pass_rate": 0.8,
  "avg_score": 7.5,
  "failure_breakdown": {"factual_error": 1, "format": 1},
  "results": [...]
}
```

### Human Annotation (JSONL)

```jsonl
{"test_case_id": "tc_001", "run_id": "abc12345", "rating": "bad", "category": "factual_error", "notes": "Wrong answer", "refinement_instruction": "Add calculation steps", "ideal_output": "2+2=4"}
```

### Refinement Feedback (JSON)

```json
{
  "run_id": "abc12345",
  "general_instructions": "Focus on accuracy over brevity",
  "priority_issues": ["factual errors", "missing context"],
  "style_guidance": "Use bullet points for lists",
  "constraints": ["Never mention competitors", "Keep under 200 words"],
  "created_at": "2025-01-28T10:00:00"
}
```

---

## Extending the Workflow

### Adding a New Failure Category

Edit `refinement/failure_analyzer.py`:

```python
class FailureAnalyzer:
    FAILURE_CATEGORIES = {
        "invalid_format": {...},
        "factual_error": {...},
        # Add your new category:
        "hallucination": {
            "description": "Agent made up information not in context",
            "fix_strategy": "add_grounding_constraints",
            "keywords": ["invented", "fabricated", "not found"],
        },
    }
```

### Adding a New Grading Criterion

Edit `graders/llm_judge.py`:

```python
class LLMJudge:
    GRADING_CRITERIA = {
        "correctness": {"weight": 0.35, "description": "..."},
        "format_compliance": {"weight": 0.25, "description": "..."},
        "behavior_alignment": {"weight": 0.25, "description": "..."},
        # Add your new criterion:
        "safety": {
            "weight": 0.15,
            "description": "Response avoids harmful content",
        },
    }
```

### Adding a New CLI Command

Edit `main.py`:

```python
@cli.command("my-command")
@click.option("--option1", required=True, help="Description")
def my_command(option1):
    """Command description shown in --help."""
    # Implementation
    console.print(f"Running with {option1}")
```

### Customizing the Optimization Prompt

Edit `refinement/prompt_optimizer.py`:

```python
class PromptOptimizer:
    OPTIMIZER_SYSTEM_PROMPT = """
    You are an expert prompt engineer.
    # Add your custom instructions here
    """
    
    OPTIMIZER_USER_TEMPLATE = """
    # Customize the template sent to GPT-4
    """
```

### Adding a New Agent API Integration

Edit `core/agent_invoker.py`:

```python
class AgentInvoker:
    async def invoke(self, test_case, prompt_version):
        if self.use_mock:
            return await self._invoke_mock(test_case, prompt_version)
        elif self.api_type == "custom":
            return await self._invoke_custom_api(test_case, prompt_version)
        else:
            return await self._invoke_default(test_case, prompt_version)
    
    async def _invoke_custom_api(self, test_case, prompt_version):
        # Your custom API logic
        pass
```

### Adding New Review Modes

Edit `human_feedback/cli_reviewer.py`:

```python
class HumanReviewer:
    def my_custom_review(self, eval_results, run_id):
        """Custom review mode."""
        for result in eval_results:
            if not result.passed:
                # Your custom review logic
                pass
```

Then add to `main.py`:

```python
@click.option("--my-mode", is_flag=True, help="My custom review mode")
def review(run_id, quick, detailed, my_mode):
    if my_mode:
        annotations = reviewer.my_custom_review(failures, run_id)
```

---

## Troubleshooting

### Common Issues

#### "GPT-4 doesn't support response_format: json_object"

**Solution**: Use `gpt-4o-mini` instead of `gpt-4` in settings:

```bash
JUDGE_MODEL=gpt-4o-mini
```

#### "Agent API not responding"

**Solution**: The workflow automatically uses mock mode when the agent API is unavailable. Check your `.env`:

```bash
# Leave empty or use placeholder for mock mode
AGENT_API_KEY=
```

#### "No test cases found"

**Solution**: Add test cases first:

```bash
python main.py test add --input "Test question" --expected "Expected answer"
```

#### "Prompt version not found"

**Solution**: Create an initial prompt:

```bash
python main.py prompt create --system "You are helpful." --template "{input}"
```

### Debug Mode

For verbose output, check the evaluation results directly:

```bash
# View raw results
cat data/results/run_<id>.json | python -m json.tool

# View annotations
cat data/results/annotations/<id>.jsonl
```

### Reset Data

To start fresh:

```bash
# Clear all data
rm -rf data/test_cases/*
rm -rf data/prompts/versions/*
rm -rf data/results/*
```

---

## API Reference

### Models (`core/models.py`)

| Model | Purpose |
|-------|---------|
| `TestCase` | Input question + expected output |
| `PromptVersion` | Versioned prompt with lineage |
| `EvaluationResult` | Single test case result |
| `EvalRunSummary` | Aggregated run statistics |
| `GraderResult` | LLM judge scores |
| `HumanAnnotation` | Human feedback on failure |
| `RefinementFeedback` | Overall refinement instructions |

### Key Methods

```python
# Run evaluation
controller = IterationController()
summary = await controller.run_evaluation(prompt, test_cases)

# Analyze failures
analyzer = FailureAnalyzer()
analysis = analyzer.analyze_failures(results)

# Generate optimized prompt
optimizer = PromptOptimizer()
new_version = optimizer.create_optimized_version(
    current_version=prompt,
    failure_analysis=analysis,
    human_annotations=annotations,
    refinement_feedback=feedback,
)

# Run full loop
await controller.run_loop(
    initial_prompt_version=1,
    skip_human_review=False,
)
```

---

## License

MIT License - See LICENSE file for details.

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

For major changes, please open an issue first to discuss the proposed changes.
