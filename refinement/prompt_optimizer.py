"""Prompt optimizer that refines prompts based on failure analysis."""
import json
from typing import Optional
from openai import OpenAI
from rich.console import Console

from core.models import PromptVersion, EvaluationResult, HumanAnnotation, RefinementFeedback
from core.prompt_manager import PromptManager
from config import settings

console = Console()


class PromptOptimizer:
    """Optimizes prompts based on failure analysis and human feedback."""
    
    OPTIMIZER_SYSTEM_PROMPT = """You are an expert prompt engineer. Your task is to improve an AI agent's system prompt based on evaluation failures and human feedback.

You will receive:
1. The current system prompt
2. A failure analysis summary
3. Example failures with feedback
4. Human annotations with detailed NLP critiques
5. Overall refinement instructions from the human reviewer

Your job is to:
1. Understand the root causes of failures from human critiques
2. Follow the human's refinement instructions carefully
3. Address priority fixes first
4. Modify the prompt to address these issues
5. Preserve what's working well
6. Apply specific fix strategies

Fix Strategy Guidelines:
- **add_format_instructions**: Add explicit format requirements with examples
- **add_context_grounding**: Add instructions to stay factual and cite sources
- **strengthen_instructions**: Make instructions more explicit and emphatic
- **add_edge_case_handling**: Add handling for unusual inputs
- **add_length_constraints**: Add word/sentence count limits

IMPORTANT: Pay close attention to the human's specific refinement instructions and critiques.
They know the domain better than the automated analysis.

Respond ONLY with a JSON object:
{
    "improved_system_prompt": "<the full improved system prompt>",
    "instruction_template": "<optional instruction template with {{user_input}} placeholder>",
    "few_shot_examples": [{"input": "...", "output": "..."}, ...],
    "changes_made": ["<list of specific changes made>"],
    "optimization_notes": "<brief summary of improvements>"
}"""

    OPTIMIZER_USER_TEMPLATE = """Improve this prompt based on the failure analysis and human feedback:

## CURRENT SYSTEM PROMPT:
{current_prompt}

## CURRENT INSTRUCTION TEMPLATE:
{instruction_template}

## CURRENT FEW-SHOT EXAMPLES:
{few_shot_examples}

## FAILURE ANALYSIS:
{failure_analysis}

## EXAMPLE FAILURES:
{example_failures}

## HUMAN ANNOTATIONS (NLP Feedback):
{human_feedback}

## HUMAN REFINEMENT INSTRUCTIONS:
{refinement_instructions}

Generate an improved prompt that addresses these issues, prioritizing the human's specific feedback."""

    def __init__(self):
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = settings.JUDGE_MODEL
        self.prompt_manager = PromptManager()
    
    def _format_examples(self, examples_by_category: dict) -> str:
        """Format failure examples for the optimizer."""
        if not examples_by_category:
            return "No failure examples available."
        
        lines = []
        for category, examples in examples_by_category.items():
            if category == "none":
                continue
            lines.append(f"\n### {category.upper()} failures:")
            for i, ex in enumerate(examples, 1):
                lines.append(f"\nExample {i}:")
                lines.append(f"  Input: {ex['input']}")
                lines.append(f"  Expected: {ex['expected']}")
                lines.append(f"  Actual: {ex['actual']}")
                lines.append(f"  Feedback: {ex['feedback']}")
        
        return "\n".join(lines) if lines else "No failure examples available."
    
    def _format_human_feedback(self, annotations: Optional[list[HumanAnnotation]]) -> str:
        """Format human annotations for the optimizer with detailed NLP feedback."""
        if not annotations:
            return "No human feedback available."
        
        lines = []
        for i, ann in enumerate(annotations, 1):
            lines.append(f"\n### Annotation {i} (Rating: {ann.rating}):")
            
            if ann.critique:
                lines.append(f"  Critique: {ann.critique}")
            
            if ann.refinement_instruction:
                lines.append(f"  Refinement instruction: {ann.refinement_instruction}")
            
            if ann.ideal_output:
                lines.append(f"  Ideal output should be: {ann.ideal_output}")
            
            if ann.suggested_fix:
                lines.append(f"  Suggested prompt fix: {ann.suggested_fix}")
            
            if ann.failure_category:
                lines.append(f"  Category: {ann.failure_category}")
        
        return "\n".join(lines) if lines else "No human feedback to address."
    
    def _format_refinement_instructions(self, feedback: Optional[RefinementFeedback]) -> str:
        """Format overall refinement feedback for the optimizer."""
        if not feedback:
            return "No overall refinement instructions provided."
        
        lines = []
        
        if feedback.general_instructions:
            lines.append(f"GENERAL INSTRUCTIONS: {feedback.general_instructions}")
        
        if feedback.priority_fixes:
            lines.append("\nPRIORITY FIXES (in order of importance):")
            for i, fix in enumerate(feedback.priority_fixes, 1):
                lines.append(f"  {i}. {fix}")
        
        if feedback.tone_feedback:
            lines.append(f"\nTONE/STYLE: {feedback.tone_feedback}")
        
        if feedback.format_feedback:
            lines.append(f"\nFORMAT: {feedback.format_feedback}")
        
        if feedback.content_feedback:
            lines.append(f"\nCONTENT ACCURACY: {feedback.content_feedback}")
        
        if feedback.additional_context:
            lines.append(f"\nADDITIONAL CONTEXT: {feedback.additional_context}")
        
        return "\n".join(lines) if lines else "No overall refinement instructions provided."
    
    def generate_improved_prompt(
        self,
        current_version: PromptVersion,
        failure_analysis: dict,
        human_annotations: Optional[list[HumanAnnotation]] = None,
        refinement_feedback: Optional[RefinementFeedback] = None,
    ) -> dict:
        """
        Generate an improved prompt based on failure analysis and human feedback.
        
        Returns:
            dict with improved_system_prompt, instruction_template, 
            few_shot_examples, changes_made, optimization_notes
        """
        try:
            # Build the optimization request
            user_message = self.OPTIMIZER_USER_TEMPLATE.format(
                current_prompt=current_version.system_prompt,
                instruction_template=current_version.instruction_template or "None",
                few_shot_examples=json.dumps(current_version.few_shot_examples, indent=2) if current_version.few_shot_examples else "None",
                failure_analysis=self._summarize_analysis(failure_analysis),
                example_failures=self._format_examples(failure_analysis.get("examples_by_category", {})),
                human_feedback=self._format_human_feedback(human_annotations),
                refinement_instructions=self._format_refinement_instructions(refinement_feedback),
            )
            
            # Call GPT-4 for optimization
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.OPTIMIZER_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.7,
                response_format={"type": "json_object"},
            )
            
            result_text = response.choices[0].message.content
            result_data = json.loads(result_text)
            
            return result_data
            
        except Exception as e:
            console.print(f"[red]Error generating improved prompt: {e}[/red]")
            return {
                "improved_system_prompt": current_version.system_prompt,
                "instruction_template": current_version.instruction_template,
                "few_shot_examples": current_version.few_shot_examples,
                "changes_made": [],
                "optimization_notes": f"Optimization failed: {e}",
            }
    
    def _summarize_analysis(self, analysis: dict) -> str:
        """Create a summary of the failure analysis."""
        if not analysis.get("fix_recommendations"):
            return "No significant failures to address."
        
        lines = [
            f"Failure rate: {analysis.get('failure_rate', 0):.1%}",
            f"Total failures: {analysis.get('total_failures', 0)}/{analysis.get('total_cases', 0)}",
            "",
            "Issues by priority:",
        ]
        
        for rec in analysis.get("fix_recommendations", []):
            lines.append(f"- {rec['category']}: {rec['count']} cases")
            lines.append(f"  Strategy: {rec['fix_strategy']}")
        
        return "\n".join(lines)
    
    def create_optimized_version(
        self,
        current_version: PromptVersion,
        failure_analysis: dict,
        human_annotations: Optional[list[HumanAnnotation]] = None,
        refinement_feedback: Optional[RefinementFeedback] = None,
    ) -> PromptVersion:
        """
        Create a new optimized prompt version and save it.
        
        Args:
            current_version: The current prompt version to improve
            failure_analysis: Analysis of failures from the analyzer
            human_annotations: Individual case annotations
            refinement_feedback: Overall refinement instructions and priorities
        
        Returns:
            The new PromptVersion
        """
        # Generate improvement
        improvement = self.generate_improved_prompt(
            current_version=current_version,
            failure_analysis=failure_analysis,
            human_annotations=human_annotations,
            refinement_feedback=refinement_feedback,
        )
        
        # Create new version with parent linkage
        new_version = self.prompt_manager.create_version(
            system_prompt=improvement.get("improved_system_prompt", current_version.system_prompt),
            instruction_template=improvement.get("instruction_template", current_version.instruction_template),
            few_shot_examples=improvement.get("few_shot_examples", current_version.few_shot_examples),
            parent_version=current_version,
            optimization_notes=improvement.get("optimization_notes", "Automated optimization"),
        )
        
        console.print(f"[green]✓[/green] Created optimized prompt version {new_version.version_number}")
        
        # Log changes
        changes = improvement.get("changes_made", [])
        if changes:
            console.print("[dim]Changes made:[/dim]")
            for change in changes:
                console.print(f"  [dim]- {change}[/dim]")
        
        return new_version
    
    async def push_optimized_prompt(
        self,
        version: PromptVersion,
        remote_prompt_id: str,
    ) -> dict:
        """
        Push the optimized prompt to the remote API.
        
        Uses the PROMPT_API_UPDATE_URL to update the remote prompt.
        """
        return await self.prompt_manager.push_to_api(
            version_number=version.version_number,
            prompt_id=remote_prompt_id,
        )
    
    # ============== Manual Fix Strategies ==============
    
    def apply_format_fix(self, prompt: str, format_description: str) -> str:
        """Add format instructions to a prompt."""
        format_section = f"""

## OUTPUT FORMAT
You MUST respond in the following format:
{format_description}

Do not deviate from this format under any circumstances."""
        
        return prompt + format_section
    
    def apply_length_fix(self, prompt: str, min_words: int = None, max_words: int = None) -> str:
        """Add length constraints to a prompt."""
        constraints = []
        if min_words:
            constraints.append(f"at least {min_words} words")
        if max_words:
            constraints.append(f"no more than {max_words} words")
        
        length_section = f"""

## LENGTH REQUIREMENTS
Your response should be {' and '.join(constraints)}. Be concise but complete."""
        
        return prompt + length_section
    
    def apply_instruction_strengthening(self, prompt: str, key_instructions: list[str]) -> str:
        """Strengthen key instructions in the prompt."""
        emphasis_section = """

## CRITICAL REQUIREMENTS
The following instructions are MANDATORY and must be followed exactly:
"""
        for i, instruction in enumerate(key_instructions, 1):
            emphasis_section += f"\n{i}. {instruction}"
        
        return prompt + emphasis_section
