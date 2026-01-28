# Agent Prompt Refinement Workflow

An automated workflow for iteratively refining AI agent prompts through evaluation, analysis, and optimization.

## Overview

This tool implements a systematic approach to prompt engineering:

1. **MEASURE** - Run test cases against your agent and grade outputs with GPT-4
2. **ANALYZE** - Identify failure patterns and categorize issues
3. **HUMAN REVIEW** - Collect feedback on failures (optional)
4. **IMPROVE** - Generate optimized prompts using GPT-4
5. **REPEAT** - Iterate until convergence

## Features

- 📝 **Test Case Management** - JSONL-based storage for questions and expected outputs
- 🔄 **Version Control** - Track prompt versions with full lineage
- 🤖 **LLM-as-Judge** - GPT-4 evaluates agent outputs on correctness, format, and behavior
- 👤 **Human Feedback** - Interactive CLI for reviewing failures
- ⚡ **Auto-Optimization** - GPT-4 generates improved prompts based on failure analysis
- 🎯 **Convergence Detection** - Automatic stopping when target pass rate is reached
- 🔌 **API Integration** - Sync prompts with your agent's API

## Installation

```bash
# Clone the repository
git clone https://github.com/akshaykumarbedre/agent-prompt-refinement.git
cd agent-prompt-refinement

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys
```

## Configuration

Copy `.env.example` to `.env` and configure:

```env
# Required: OpenAI API key for GPT-4 Judge
OPENAI_API_KEY=your-openai-api-key

# Your agent's API endpoint
AGENT_API_URL=http://localhost:8000/api/agent/invoke
AGENT_API_KEY=your-agent-api-key

# Optional: Remote prompt management API
PROMPT_API_GET_URL=http://localhost:8000/api/prompt/get
PROMPT_API_UPDATE_URL=http://localhost:8000/api/prompt/update
```

## Quick Start

### 1. Add Test Cases

```bash
# Add individual test cases
python main.py test add --input "What is 2+2?" --expected "4"

# Or import from JSONL file
python main.py test import test_cases.jsonl
```

### 2. Create Initial Prompt

```bash
python main.py prompt create --system-prompt "You are a helpful assistant that answers questions accurately and concisely."
```

### 3. Run Evaluation

```bash
python main.py eval run --prompt-version 1
```

### 4. Review Failures (Optional)

```bash
python main.py review --run-id <run-id>
```

### 5. Run Full Refinement Loop

```bash
# Automated mode (no human review)
python main.py loop start --skip-review

# With human review
python main.py loop start
```

## CLI Commands

### Test Case Management
```bash
python main.py test add      # Add a test case
python main.py test list     # List all test cases
python main.py test import   # Import from JSONL
python main.py test export   # Export to JSONL
python main.py test stats    # Show dataset statistics
```

### Prompt Management
```bash
python main.py prompt create   # Create new version
python main.py prompt list     # List all versions
python main.py prompt show     # Show version details
python main.py prompt lineage  # Show version ancestry
python main.py prompt sync     # Sync from remote API
python main.py prompt push     # Push to remote API
```

### Evaluation
```bash
python main.py eval run    # Run evaluation
python main.py eval show   # Show run results
```

### Optimization
```bash
python main.py review      # Review failures
python main.py optimize    # Generate improved prompt
```

### Refinement Loop
```bash
python main.py loop start   # Start refinement loop
python main.py loop config  # Show loop settings
```

## Project Structure

```
├── main.py                 # CLI entry point
├── config/
│   └── settings.py         # Configuration management
├── core/
│   ├── models.py           # Data models (Pydantic)
│   ├── prompt_manager.py   # Prompt version management
│   ├── test_case_manager.py # Test case storage
│   ├── agent_invoker.py    # Agent API client
│   └── iteration_controller.py # Main loop orchestrator
├── graders/
│   └── llm_judge.py        # GPT-4 evaluation
├── refinement/
│   ├── failure_analyzer.py # Failure categorization
│   └── prompt_optimizer.py # Prompt improvement
├── human_feedback/
│   └── cli_reviewer.py     # Interactive review CLI
└── utils/
```

## Failure Categories

The system automatically categorizes failures:

| Category | Description | Fix Strategy |
|----------|-------------|--------------|
| `invalid_format` | Output format issues | Add format instructions |
| `factual_error` | Incorrect information | Add context grounding |
| `behavior_drift` | Inconsistent behavior | Strengthen instructions |
| `edge_case` | Unusual input failures | Add edge case handling |
| `verbosity` | Length issues | Add length constraints |

## API Integration

The workflow can sync with your agent's prompt management API:

```bash
# Fetch current prompt from API
python main.py prompt sync --prompt-id my-agent-prompt

# Push optimized prompt to API
python main.py prompt push 5 --prompt-id my-agent-prompt
```

Expected API format:
- GET `/api/prompt/get` - Returns `{prompt_id, system_prompt, ...}`
- POST `/api/prompt/update` - Accepts `{prompt_id, system_prompt, ...}`

## License

MIT License
