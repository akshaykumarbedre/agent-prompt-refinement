"""Interactive CLI for human review of all failures."""
import json
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich.markdown import Markdown
from rich.syntax import Syntax

from core.models import EvaluationResult, HumanAnnotation, EvalRunSummary, RefinementFeedback
from config import settings

console = Console()


class HumanReviewer:
    """Interactive CLI for reviewing all failures and providing feedback."""
    
    FAILURE_CATEGORIES = [
        "invalid_format",
        "factual_error", 
        "behavior_drift",
        "edge_case",
        "verbosity",
        "other",
    ]
    
    def __init__(self):
        self.annotations_dir = settings.RESULTS_DIR / "annotations"
        self.annotations_dir.mkdir(parents=True, exist_ok=True)
    
    def _save_annotation(self, annotation: HumanAnnotation, run_id: str):
        """Save an annotation to disk."""
        file_path = self.annotations_dir / f"{run_id}_annotations.jsonl"
        with open(file_path, "a") as f:
            f.write(json.dumps(annotation.model_dump()) + "\n")
    
    def load_annotations(self, run_id: str) -> list[HumanAnnotation]:
        """Load all annotations for a run."""
        file_path = self.annotations_dir / f"{run_id}_annotations.jsonl"
        if not file_path.exists():
            return []
        
        annotations = []
        with open(file_path, "r") as f:
            for line in f:
                if line.strip():
                    annotations.append(HumanAnnotation(**json.loads(line)))
        return annotations
    
    def display_failure(self, result: EvaluationResult, index: int, total: int):
        """Display a single failure for review."""
        console.clear()
        
        # Header
        console.print(Panel(
            f"[bold]Reviewing Failure {index}/{total}[/bold]  |  Test ID: {result.test_case_id}",
            style="blue"
        ))
        
        # Scores table
        scores_table = Table(title="GPT-4 Judge Scores", show_header=True)
        scores_table.add_column("Metric", style="cyan")
        scores_table.add_column("Score", style="yellow")
        scores_table.add_column("Status", style="red")
        
        for metric, score in result.grader_result.scores.items():
            status = "✓" if score >= 7 else "✗"
            color = "green" if score >= 7 else "red"
            scores_table.add_row(
                metric.replace("_", " ").title(),
                f"{score:.1f}/10",
                f"[{color}]{status}[/{color}]"
            )
        scores_table.add_row(
            "[bold]Aggregate[/bold]",
            f"[bold]{result.grader_result.aggregate_score:.1f}/10[/bold]",
            f"[red]FAILED[/red]"
        )
        console.print(scores_table)
        console.print()
        
        # Input
        console.print(Panel(
            result.input_text,
            title="[cyan]INPUT / QUESTION[/cyan]",
            border_style="cyan"
        ))
        
        # Expected output
        console.print(Panel(
            result.expected_output,
            title="[green]EXPECTED OUTPUT[/green]",
            border_style="green"
        ))
        
        # Actual output
        console.print(Panel(
            result.actual_output,
            title="[red]ACTUAL OUTPUT[/red]",
            border_style="red"
        ))
        
        # Judge feedback
        if result.grader_result.feedback:
            console.print(Panel(
                result.grader_result.feedback,
                title="[yellow]JUDGE FEEDBACK[/yellow]",
                border_style="yellow"
            ))
        
        # Auto-detected category
        auto_category = result.grader_result.details.get("failure_category", "other")
        console.print(f"\n[dim]Auto-detected category: {auto_category}[/dim]")
    
    def collect_feedback(self, result: EvaluationResult) -> HumanAnnotation:
        """Collect human feedback for a failure."""
        console.print("\n[bold]─── Your Feedback ───[/bold]\n")
        
        # Rating
        rating = Prompt.ask(
            "Rating",
            choices=["good", "bad", "neutral"],
            default="bad"
        )
        
        # Category selection
        console.print("\n[dim]Failure categories:[/dim]")
        for i, cat in enumerate(self.FAILURE_CATEGORIES, 1):
            console.print(f"  {i}. {cat}")
        
        auto_category = result.grader_result.details.get("failure_category", "other")
        auto_idx = self.FAILURE_CATEGORIES.index(auto_category) + 1 if auto_category in self.FAILURE_CATEGORIES else 6
        
        cat_input = Prompt.ask(
            "Category (number or name)",
            default=str(auto_idx)
        )
        
        # Parse category input
        try:
            cat_idx = int(cat_input) - 1
            failure_category = self.FAILURE_CATEGORIES[cat_idx]
        except (ValueError, IndexError):
            if cat_input in self.FAILURE_CATEGORIES:
                failure_category = cat_input
            else:
                failure_category = "other"
        
        # Critique
        critique = Prompt.ask(
            "What's wrong with this output? (critique)",
            default=""
        )
        
        # Suggested fix
        suggested_fix = Prompt.ask(
            "Suggested fix for the prompt",
            default=""
        )
        
        return HumanAnnotation(
            eval_result_id=result.id,
            rating=rating,
            failure_category=failure_category,
            critique=critique,
            suggested_fix=suggested_fix,
        )
    
    def review_all_failures(
        self,
        eval_results: list[EvaluationResult],
        run_id: str,
    ) -> list[HumanAnnotation]:
        """
        Interactive review of ALL failures.
        
        Returns:
            List of HumanAnnotation objects
        """
        failures = [r for r in eval_results if not r.passed]
        
        if not failures:
            console.print("[green]✓ No failures to review![/green]")
            return []
        
        console.print(f"\n[bold]Found {len(failures)} failures to review[/bold]")
        console.print("[dim]Press Ctrl+C at any time to stop and save progress[/dim]\n")
        
        if not Confirm.ask("Start reviewing failures?"):
            return []
        
        annotations = []
        
        try:
            for i, failure in enumerate(failures, 1):
                # Display the failure
                self.display_failure(failure, i, len(failures))
                
                # Collect feedback
                annotation = self.collect_feedback(failure)
                annotations.append(annotation)
                
                # Save immediately
                self._save_annotation(annotation, run_id)
                
                console.print(f"\n[green]✓ Saved annotation for {failure.test_case_id}[/green]")
                
                # Continue prompt
                if i < len(failures):
                    if not Confirm.ask("\nContinue to next failure?", default=True):
                        console.print(f"\n[yellow]Stopped after {i}/{len(failures)} reviews[/yellow]")
                        break
        
        except KeyboardInterrupt:
            console.print(f"\n\n[yellow]Review interrupted. Saved {len(annotations)} annotations.[/yellow]")
        
        # Summary
        self._display_review_summary(annotations)
        
        return annotations
    
    def _display_review_summary(self, annotations: list[HumanAnnotation]):
        """Display a summary of the review session."""
        if not annotations:
            return
        
        console.print("\n" + "─" * 50)
        console.print("[bold]Review Session Summary[/bold]\n")
        
        # Rating breakdown
        ratings = {}
        categories = {}
        
        for ann in annotations:
            ratings[ann.rating] = ratings.get(ann.rating, 0) + 1
            categories[ann.failure_category] = categories.get(ann.failure_category, 0) + 1
        
        # Ratings table
        table = Table(title="Ratings")
        table.add_column("Rating", style="cyan")
        table.add_column("Count", style="yellow")
        for rating, count in sorted(ratings.items()):
            table.add_row(rating, str(count))
        console.print(table)
        
        # Categories table
        table = Table(title="Failure Categories")
        table.add_column("Category", style="cyan")
        table.add_column("Count", style="yellow")
        for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
            table.add_row(cat, str(count))
        console.print(table)
        
        # Suggestions with content
        suggestions = [a.suggested_fix for a in annotations if a.suggested_fix]
        if suggestions:
            console.print("\n[bold]Suggested Fixes:[/bold]")
            for i, suggestion in enumerate(suggestions, 1):
                console.print(f"  {i}. {suggestion}")
    
    def quick_review(
        self,
        eval_results: list[EvaluationResult],
        run_id: str,
    ) -> list[HumanAnnotation]:
        """
        Quick review mode - just rate good/bad for each failure.
        """
        failures = [r for r in eval_results if not r.passed]
        
        if not failures:
            console.print("[green]✓ No failures to review![/green]")
            return []
        
        console.print(f"\n[bold]Quick Review Mode: {len(failures)} failures[/bold]")
        console.print("[dim]Rate each output as good/bad/skip. Press 'q' to quit.[/dim]\n")
        
        annotations = []
        
        for i, failure in enumerate(failures, 1):
            console.print(f"\n[bold]─── {i}/{len(failures)} ───[/bold]")
            console.print(f"[cyan]Input:[/cyan] {failure.input_text[:100]}...")
            console.print(f"[green]Expected:[/green] {failure.expected_output[:100]}...")
            console.print(f"[red]Actual:[/red] {failure.actual_output[:100]}...")
            
            rating = Prompt.ask(
                "Rate",
                choices=["g", "b", "s", "q"],
                default="b"
            )
            
            if rating == "q":
                break
            
            if rating != "s":  # Skip
                annotation = HumanAnnotation(
                    eval_result_id=failure.id,
                    rating="good" if rating == "g" else "bad",
                    failure_category=failure.grader_result.details.get("failure_category", "other"),
                )
                annotations.append(annotation)
                self._save_annotation(annotation, run_id)
        
        console.print(f"\n[green]✓ Completed {len(annotations)} quick reviews[/green]")
        return annotations

    def detailed_review(
        self,
        eval_results: list[EvaluationResult],
        run_id: str,
    ) -> tuple[list[HumanAnnotation], Optional[RefinementFeedback]]:
        """
        Detailed review mode with NLP feedback for each failure.
        Allows natural language critique, ideal output, and refinement instructions.
        
        Returns:
            Tuple of (annotations, refinement_feedback)
        """
        failures = [r for r in eval_results if not r.passed]
        
        if not failures:
            console.print("[green]✓ No failures to review![/green]")
            return [], None
        
        console.print(f"\n[bold]Detailed Review Mode: {len(failures)} failures[/bold]")
        console.print("[dim]Provide detailed NLP feedback for each failure.[/dim]")
        console.print("[dim]Your feedback will be used to refine the prompt.[/dim]\n")
        
        if not Confirm.ask("Start detailed review?"):
            return [], None
        
        annotations = []
        
        try:
            for i, failure in enumerate(failures, 1):
                # Display the failure
                self.display_failure(failure, i, len(failures))
                
                # Collect detailed feedback
                annotation = self._collect_detailed_feedback(failure)
                annotations.append(annotation)
                
                # Save immediately
                self._save_annotation(annotation, run_id)
                
                console.print(f"\n[green]✓ Saved detailed feedback for {failure.test_case_id}[/green]")
                
                # Continue prompt
                if i < len(failures):
                    if not Confirm.ask("\nContinue to next failure?", default=True):
                        console.print(f"\n[yellow]Stopped after {i}/{len(failures)} reviews[/yellow]")
                        break
        
        except KeyboardInterrupt:
            console.print(f"\n\n[yellow]Review interrupted. Saved {len(annotations)} annotations.[/yellow]")
        
        # Collect overall refinement feedback
        console.print("\n" + "═" * 50)
        console.print("[bold]Overall Refinement Feedback[/bold]")
        console.print("[dim]Provide general instructions for the prompt optimizer.[/dim]\n")
        
        refinement_feedback = self._collect_refinement_feedback(run_id, annotations)
        
        # Summary
        self._display_detailed_review_summary(annotations, refinement_feedback)
        
        return annotations, refinement_feedback
    
    def _collect_detailed_feedback(self, result: EvaluationResult) -> HumanAnnotation:
        """Collect detailed NLP feedback for a single failure."""
        console.print("\n[bold]─── Your Detailed Feedback ───[/bold]\n")
        
        # Rating
        console.print("[dim]Rate the output quality:[/dim]")
        rating = Prompt.ask(
            "Rating (good/bad/neutral)",
            choices=["good", "bad", "neutral"],
            default="bad"
        )
        
        # Category selection
        console.print("\n[dim]Failure categories:[/dim]")
        for i, cat in enumerate(self.FAILURE_CATEGORIES, 1):
            console.print(f"  {i}. {cat}")
        
        auto_category = result.grader_result.details.get("failure_category", "other")
        auto_idx = self.FAILURE_CATEGORIES.index(auto_category) + 1 if auto_category in self.FAILURE_CATEGORIES else 6
        
        cat_input = Prompt.ask(
            "Category (number or name)",
            default=str(auto_idx)
        )
        
        try:
            cat_idx = int(cat_input) - 1
            failure_category = self.FAILURE_CATEGORIES[cat_idx]
        except (ValueError, IndexError):
            failure_category = cat_input if cat_input in self.FAILURE_CATEGORIES else "other"
        
        # Detailed NLP critique
        console.print("\n[bold cyan]Describe what's wrong with this output (NLP feedback):[/bold cyan]")
        console.print("[dim]Be specific about the issues - this will help the optimizer understand the problem.[/dim]")
        critique = Prompt.ask("Critique", default="")
        
        # Ideal output (optional)
        console.print("\n[bold green]What should the ideal output be? (optional)[/bold green]")
        console.print("[dim]Leave empty to skip, or provide what you expected.[/dim]")
        ideal_output = Prompt.ask("Ideal output", default="")
        
        # Refinement instruction
        console.print("\n[bold yellow]How should the prompt be changed to fix this?[/bold yellow]")
        console.print("[dim]e.g., 'Add instruction to mention return policy time limits'[/dim]")
        refinement_instruction = Prompt.ask("Refinement instruction", default="")
        
        # Suggested fix (more specific)
        console.print("\n[bold magenta]Any specific text to add to the prompt?[/bold magenta]")
        console.print("[dim]e.g., 'Always mention the 30-day return window'[/dim]")
        suggested_fix = Prompt.ask("Suggested prompt addition", default="")
        
        return HumanAnnotation(
            eval_result_id=result.id,
            rating=rating,
            failure_category=failure_category,
            critique=critique,
            ideal_output=ideal_output,
            refinement_instruction=refinement_instruction,
            suggested_fix=suggested_fix,
        )
    
    def _collect_refinement_feedback(
        self, 
        run_id: str, 
        annotations: list[HumanAnnotation]
    ) -> RefinementFeedback:
        """Collect overall refinement feedback after reviewing all failures."""
        
        if not Confirm.ask("Provide overall refinement feedback?", default=True):
            return RefinementFeedback(run_id=run_id)
        
        console.print("\n[bold]General Instructions for Prompt Refinement:[/bold]")
        console.print("[dim]What overall changes should be made to improve the prompt?[/dim]")
        general_instructions = Prompt.ask("General instructions", default="")
        
        console.print("\n[bold]Tone/Style Feedback:[/bold]")
        console.print("[dim]Should the responses be more formal, friendly, concise, detailed?[/dim]")
        tone_feedback = Prompt.ask("Tone feedback", default="")
        
        console.print("\n[bold]Format Feedback:[/bold]")
        console.print("[dim]Any changes to how responses should be structured?[/dim]")
        format_feedback = Prompt.ask("Format feedback", default="")
        
        console.print("\n[bold]Content Accuracy Feedback:[/bold]")
        console.print("[dim]What factual information is missing or incorrect?[/dim]")
        content_feedback = Prompt.ask("Content feedback", default="")
        
        console.print("\n[bold]Priority Fixes (comma-separated):[/bold]")
        console.print("[dim]List the most important things to fix, in order of priority.[/dim]")
        priority_input = Prompt.ask("Priority fixes", default="")
        priority_fixes = [p.strip() for p in priority_input.split(",") if p.strip()]
        
        console.print("\n[bold]Additional Context:[/bold]")
        console.print("[dim]Any other information that would help improve the prompt?[/dim]")
        additional_context = Prompt.ask("Additional context", default="")
        
        feedback = RefinementFeedback(
            run_id=run_id,
            general_instructions=general_instructions,
            tone_feedback=tone_feedback,
            format_feedback=format_feedback,
            content_feedback=content_feedback,
            priority_fixes=priority_fixes,
            additional_context=additional_context,
        )
        
        # Save refinement feedback
        self._save_refinement_feedback(feedback, run_id)
        
        return feedback
    
    def _save_refinement_feedback(self, feedback: RefinementFeedback, run_id: str):
        """Save refinement feedback to disk."""
        file_path = self.annotations_dir / f"{run_id}_refinement_feedback.json"
        with open(file_path, "w") as f:
            json.dump(feedback.model_dump(), f, indent=2)
        console.print(f"\n[green]✓ Saved refinement feedback[/green]")
    
    def load_refinement_feedback(self, run_id: str) -> Optional[RefinementFeedback]:
        """Load refinement feedback for a run."""
        file_path = self.annotations_dir / f"{run_id}_refinement_feedback.json"
        if not file_path.exists():
            return None
        with open(file_path, "r") as f:
            return RefinementFeedback(**json.load(f))
    
    def _display_detailed_review_summary(
        self, 
        annotations: list[HumanAnnotation],
        refinement_feedback: Optional[RefinementFeedback]
    ):
        """Display summary of detailed review session."""
        if not annotations:
            return
        
        console.print("\n" + "═" * 50)
        console.print("[bold]Detailed Review Summary[/bold]\n")
        
        # Rating breakdown
        ratings = {}
        categories = {}
        
        for ann in annotations:
            ratings[ann.rating] = ratings.get(ann.rating, 0) + 1
            categories[ann.failure_category] = categories.get(ann.failure_category, 0) + 1
        
        # Ratings table
        table = Table(title="Ratings")
        table.add_column("Rating", style="cyan")
        table.add_column("Count", style="yellow")
        for rating, count in sorted(ratings.items()):
            table.add_row(rating, str(count))
        console.print(table)
        
        # Categories table
        table = Table(title="Failure Categories")
        table.add_column("Category", style="cyan")
        table.add_column("Count", style="yellow")
        for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
            table.add_row(cat, str(count))
        console.print(table)
        
        # Critiques
        critiques = [a.critique for a in annotations if a.critique]
        if critiques:
            console.print("\n[bold]Critiques Provided:[/bold]")
            for i, critique in enumerate(critiques, 1):
                console.print(f"  {i}. {critique[:100]}{'...' if len(critique) > 100 else ''}")
        
        # Refinement instructions
        instructions = [a.refinement_instruction for a in annotations if a.refinement_instruction]
        if instructions:
            console.print("\n[bold]Refinement Instructions:[/bold]")
            for i, inst in enumerate(instructions, 1):
                console.print(f"  {i}. {inst}")
        
        # Overall feedback
        if refinement_feedback and refinement_feedback.general_instructions:
            console.print("\n[bold]Overall Refinement Direction:[/bold]")
            console.print(f"  {refinement_feedback.general_instructions}")
        
        if refinement_feedback and refinement_feedback.priority_fixes:
            console.print("\n[bold]Priority Fixes:[/bold]")
            for i, fix in enumerate(refinement_feedback.priority_fixes, 1):
                console.print(f"  {i}. {fix}")