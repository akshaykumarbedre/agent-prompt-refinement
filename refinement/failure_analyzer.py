"""Failure analyzer for categorizing and understanding evaluation failures."""
from collections import Counter
from typing import Optional
from rich.console import Console

from core.models import EvaluationResult, FailureCategory, HumanAnnotation

console = Console()


class FailureAnalyzer:
    """Analyzes evaluation failures to identify patterns and fix strategies."""
    
    # Failure categories with descriptions and fix strategies
    FAILURE_CATEGORIES = {
        "invalid_format": FailureCategory(
            category_id="invalid_format",
            name="Invalid Format",
            description="Output doesn't match expected format (JSON, list, structured data, etc.)",
            fix_strategy="add_format_instructions",
        ),
        "factual_error": FailureCategory(
            category_id="factual_error",
            name="Factual Error",
            description="Output contains incorrect facts or information",
            fix_strategy="add_context_grounding",
        ),
        "behavior_drift": FailureCategory(
            category_id="behavior_drift",
            name="Behavior Drift",
            description="Model doesn't follow instructions consistently, wrong tone or style",
            fix_strategy="strengthen_instructions",
        ),
        "edge_case": FailureCategory(
            category_id="edge_case",
            name="Edge Case Failure",
            description="Fails on unusual or edge-case inputs",
            fix_strategy="add_edge_case_handling",
        ),
        "verbosity": FailureCategory(
            category_id="verbosity",
            name="Verbosity Issue",
            description="Output is too long, too short, or not appropriately concise",
            fix_strategy="add_length_constraints",
        ),
        "other": FailureCategory(
            category_id="other",
            name="Other",
            description="Failures that don't fit other categories",
            fix_strategy="manual_review",
        ),
        "none": FailureCategory(
            category_id="none",
            name="No Failure",
            description="Test passed, no failure",
            fix_strategy="none",
        ),
    }
    
    def __init__(self):
        pass
    
    def categorize_failure(self, eval_result: EvaluationResult) -> str:
        """
        Categorize a single failure based on grader feedback.
        
        Returns the failure category ID.
        """
        if eval_result.passed:
            return "none"
        
        # Get failure category from grader if available
        grader_category = eval_result.grader_result.details.get("failure_category", "other")
        
        if grader_category in self.FAILURE_CATEGORIES:
            return grader_category
        
        # Fallback: analyze scores to determine category
        scores = eval_result.grader_result.scores
        
        # Format issues
        if scores.get("format_compliance", 10) < 5:
            return "invalid_format"
        
        # Correctness issues
        if scores.get("correctness", 10) < 5:
            return "factual_error"
        
        # Behavior issues
        if scores.get("behavior_alignment", 10) < 5:
            return "behavior_drift"
        
        return "other"
    
    def analyze_failures(
        self,
        eval_results: list[EvaluationResult],
        human_annotations: Optional[list[HumanAnnotation]] = None,
    ) -> dict:
        """
        Analyze all failures to identify patterns and priorities.
        
        Returns:
            dict with:
                - failure_breakdown: category -> count
                - prioritized_categories: list of categories by frequency
                - examples_by_category: category -> list of example failures
                - fix_recommendations: list of recommended fixes
        """
        failures = [r for r in eval_results if not r.passed]
        
        if not failures:
            return {
                "failure_breakdown": {},
                "prioritized_categories": [],
                "examples_by_category": {},
                "fix_recommendations": [],
                "total_failures": 0,
                "total_cases": len(eval_results),
            }
        
        # Categorize each failure
        categories = []
        examples_by_category = {}
        
        for failure in failures:
            category = self.categorize_failure(failure)
            failure.failure_category = category
            categories.append(category)
            
            # Store examples (up to 3 per category)
            if category not in examples_by_category:
                examples_by_category[category] = []
            if len(examples_by_category[category]) < 3:
                examples_by_category[category].append({
                    "test_case_id": failure.test_case_id,
                    "input": failure.input_text[:200],
                    "expected": failure.expected_output[:200],
                    "actual": failure.actual_output[:200],
                    "feedback": failure.grader_result.feedback,
                })
        
        # Incorporate human annotations if provided
        if human_annotations:
            annotation_map = {a.eval_result_id: a for a in human_annotations}
            for failure in failures:
                if failure.id in annotation_map:
                    annotation = annotation_map[failure.id]
                    if annotation.failure_category:
                        # Override with human category
                        old_cat = failure.failure_category
                        failure.failure_category = annotation.failure_category
                        # Update counts
                        categories = [annotation.failure_category if c == old_cat else c for c in categories]
        
        # Count frequencies
        failure_breakdown = dict(Counter(categories))
        
        # Prioritize by frequency
        prioritized = sorted(failure_breakdown.keys(), key=lambda x: failure_breakdown[x], reverse=True)
        
        # Generate fix recommendations
        fix_recommendations = []
        for category in prioritized:
            if category not in ["none", "other"]:
                cat_info = self.FAILURE_CATEGORIES.get(category)
                if cat_info:
                    fix_recommendations.append({
                        "category": category,
                        "count": failure_breakdown[category],
                        "fix_strategy": cat_info.fix_strategy,
                        "description": cat_info.description,
                    })
        
        return {
            "failure_breakdown": failure_breakdown,
            "prioritized_categories": prioritized,
            "examples_by_category": examples_by_category,
            "fix_recommendations": fix_recommendations,
            "total_failures": len(failures),
            "total_cases": len(eval_results),
            "failure_rate": len(failures) / len(eval_results) if eval_results else 0,
        }
    
    def get_fix_strategy(self, category: str) -> str:
        """Get the recommended fix strategy for a failure category."""
        cat_info = self.FAILURE_CATEGORIES.get(category)
        if cat_info:
            return cat_info.fix_strategy
        return "manual_review"
    
    def summarize_for_optimizer(self, analysis: dict) -> str:
        """
        Generate a human-readable summary for the prompt optimizer.
        """
        if not analysis["fix_recommendations"]:
            return "No significant failures to address."
        
        lines = [
            f"Analysis of {analysis['total_failures']}/{analysis['total_cases']} failed cases:",
            "",
        ]
        
        for rec in analysis["fix_recommendations"]:
            lines.append(f"- {rec['category']}: {rec['count']} failures")
            lines.append(f"  Issue: {rec['description']}")
            lines.append(f"  Recommended fix: {rec['fix_strategy']}")
            lines.append("")
        
        return "\n".join(lines)
