"""Data models for the prompt refinement workflow."""
from datetime import datetime
from typing import Any, Optional
from pydantic import BaseModel, Field
import uuid


def generate_id() -> str:
    """Generate a unique ID."""
    return str(uuid.uuid4())[:8]


class TestCase(BaseModel):
    """Individual test case with input and expected output."""
    id: str = Field(default_factory=generate_id)
    input_text: str  # The question/prompt to send to agent
    expected_output: str  # Ground truth answer
    metadata: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    difficulty: str = "medium"  # easy, medium, hard
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    source: str = "manual"  # manual, synthetic, production


class TestDataset(BaseModel):
    """Collection of test cases."""
    name: str
    version: str = "1.0"
    cases: list[TestCase] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class PromptVersion(BaseModel):
    """Versioned prompt with metadata and lineage tracking."""
    id: str = Field(default_factory=generate_id)
    version_number: int
    system_prompt: str
    instruction_template: str = ""
    few_shot_examples: list[dict[str, str]] = Field(default_factory=list)
    variables: list[str] = Field(default_factory=list)
    model: str = "gpt-4"
    temperature: float = 0.7
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    parent_version_id: Optional[str] = None  # For lineage tracking
    optimization_notes: str = ""  # Why this version was created
    pass_rate: Optional[float] = None  # Last evaluated pass rate


class GraderResult(BaseModel):
    """Result from a single grader evaluation."""
    grader_name: str
    grader_type: str  # llm_judge, string_check, etc.
    scores: dict[str, float] = Field(default_factory=dict)  # correctness, format, behavior
    aggregate_score: float
    passed: bool
    details: dict[str, Any] = Field(default_factory=dict)
    feedback: str = ""  # Detailed feedback from judge


class EvaluationResult(BaseModel):
    """Complete evaluation result for one test case."""
    id: str = Field(default_factory=generate_id)
    test_case_id: str
    prompt_version_id: str
    input_text: str
    expected_output: str
    actual_output: str
    grader_result: GraderResult
    passed: bool
    latency_ms: int = 0
    tokens_used: int = 0
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    failure_category: Optional[str] = None  # Set after analysis


class EvalRunSummary(BaseModel):
    """Summary of an entire evaluation run."""
    run_id: str = Field(default_factory=generate_id)
    prompt_version_id: str
    total_cases: int
    passed: int
    failed: int
    pass_rate: float
    avg_score: float
    failure_breakdown: dict[str, int] = Field(default_factory=dict)  # category -> count
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    results: list[EvaluationResult] = Field(default_factory=list)


class HumanAnnotation(BaseModel):
    """Human feedback on a model output."""
    id: str = Field(default_factory=generate_id)
    eval_result_id: str
    rating: str  # good, bad, neutral
    critique: str = ""  # Detailed feedback
    failure_category: str = ""  # Categorized failure type
    suggested_fix: str = ""  # Human's suggested improvement
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class FailureCategory(BaseModel):
    """Categorized failure mode with fix strategy."""
    category_id: str
    name: str  # e.g., formatting_error, factual_incorrect
    description: str
    fix_strategy: str  # Which fix strategy to apply
    example_failures: list[str] = Field(default_factory=list)
    frequency: int = 0  # How often this occurs
