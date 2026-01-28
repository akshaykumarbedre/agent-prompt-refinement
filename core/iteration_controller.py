"""Iteration controller for the Analyze → Improve → Measure loop."""
import json
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table

from .models import (
    TestCase, PromptVersion, EvaluationResult, 
    EvalRunSummary, GraderResult, HumanAnnotation, RefinementFeedback
)
from .prompt_manager import PromptManager
from .test_case_manager import TestCaseManager
from .agent_invoker import AgentInvoker
from graders import LLMJudge
from refinement import FailureAnalyzer, PromptOptimizer
from human_feedback import HumanReviewer
from config import settings

console = Console()


class IterationController:
    """
    Orchestrates the prompt refinement loop:
    
    1. MEASURE: Run evaluation with current prompt
    2. ANALYZE: Identify failure patterns
    3. HUMAN REVIEW: Collect feedback on all failures
    4. IMPROVE: Generate optimized prompt
    5. Repeat until convergence
    """
    
    def __init__(self):
        self.prompt_manager = PromptManager()
        self.test_case_manager = TestCaseManager()
        self.agent_invoker = AgentInvoker()
        self.llm_judge = LLMJudge()
        self.failure_analyzer = FailureAnalyzer()
        self.prompt_optimizer = PromptOptimizer()
        self.human_reviewer = HumanReviewer()
        
        self.results_dir = settings.RESULTS_DIR
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Convergence settings
        self.max_iterations = settings.MAX_ITERATIONS
        self.convergence_threshold = settings.CONVERGENCE_THRESHOLD
        self.improvement_min = settings.IMPROVEMENT_MIN
        self.patience = settings.PATIENCE
    
    def _save_run_summary(self, summary: EvalRunSummary):
        """Save evaluation run summary to disk."""
        file_path = self.results_dir / f"run_{summary.run_id}.json"
        with open(file_path, "w") as f:
            json.dump(summary.model_dump(), f, indent=2)
    
    def load_run_summary(self, run_id: str) -> Optional[EvalRunSummary]:
        """Load a run summary from disk."""
        file_path = self.results_dir / f"run_{run_id}.json"
        if not file_path.exists():
            return None
        with open(file_path, "r") as f:
            return EvalRunSummary(**json.load(f))
    
    async def run_evaluation(
        self,
        prompt_version: PromptVersion,
        test_cases: list[TestCase],
    ) -> EvalRunSummary:
        """
        Run a full evaluation cycle.
        
        1. Invoke agent for each test case
        2. Grade each response with GPT-4 judge
        3. Aggregate results
        """
        console.print(f"\n[bold]Running Evaluation[/bold]")
        console.print(f"Prompt Version: v{prompt_version.version_number}")
        console.print(f"Test Cases: {len(test_cases)}")
        
        eval_results = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
        ) as progress:
            # Phase 1: Invoke agent
            task1 = progress.add_task("[cyan]Invoking agent...", total=len(test_cases))
            
            agent_responses = []
            for test_case in test_cases:
                response = await self.agent_invoker.invoke(test_case, prompt_version)
                agent_responses.append({
                    "test_case": test_case,
                    "response": response,
                })
                progress.update(task1, advance=1)
            
            # Phase 2: Grade responses
            task2 = progress.add_task("[yellow]Grading with GPT-4...", total=len(test_cases))
            
            for item in agent_responses:
                test_case = item["test_case"]
                response = item["response"]
                
                # Grade the response
                grader_result = self.llm_judge.evaluate(
                    input_text=test_case.input_text,
                    expected_output=test_case.expected_output,
                    actual_output=response["output"],
                )
                
                # Create evaluation result
                eval_result = EvaluationResult(
                    test_case_id=test_case.id,
                    prompt_version_id=prompt_version.id,
                    input_text=test_case.input_text,
                    expected_output=test_case.expected_output,
                    actual_output=response["output"],
                    grader_result=grader_result,
                    passed=grader_result.passed,
                    latency_ms=response["latency_ms"],
                    tokens_used=response["tokens_used"],
                )
                eval_results.append(eval_result)
                progress.update(task2, advance=1)
        
        # Aggregate results
        passed = sum(1 for r in eval_results if r.passed)
        failed = len(eval_results) - passed
        pass_rate = passed / len(eval_results) if eval_results else 0
        avg_score = sum(r.grader_result.aggregate_score for r in eval_results) / len(eval_results) if eval_results else 0
        
        # Count failure categories
        failure_breakdown = {}
        for r in eval_results:
            if not r.passed:
                cat = r.grader_result.details.get("failure_category", "other")
                failure_breakdown[cat] = failure_breakdown.get(cat, 0) + 1
        
        summary = EvalRunSummary(
            prompt_version_id=prompt_version.id,
            total_cases=len(eval_results),
            passed=passed,
            failed=failed,
            pass_rate=pass_rate,
            avg_score=avg_score,
            failure_breakdown=failure_breakdown,
            results=eval_results,
        )
        
        # Save summary
        self._save_run_summary(summary)
        
        # Display results
        self._display_eval_summary(summary)
        
        return summary
    
    def _display_eval_summary(self, summary: EvalRunSummary):
        """Display evaluation summary."""
        console.print("\n[bold]─── Evaluation Results ───[/bold]\n")
        
        # Results table
        table = Table()
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="yellow")
        
        pass_color = "green" if summary.pass_rate >= self.convergence_threshold else "red"
        
        table.add_row("Total Cases", str(summary.total_cases))
        table.add_row("Passed", f"[green]{summary.passed}[/green]")
        table.add_row("Failed", f"[red]{summary.failed}[/red]")
        table.add_row("Pass Rate", f"[{pass_color}]{summary.pass_rate:.1%}[/{pass_color}]")
        table.add_row("Avg Score", f"{summary.avg_score:.2f}/10")
        
        console.print(table)
        
        # Failure breakdown
        if summary.failure_breakdown:
            console.print("\n[bold]Failure Breakdown:[/bold]")
            for cat, count in sorted(summary.failure_breakdown.items(), key=lambda x: x[1], reverse=True):
                console.print(f"  • {cat}: {count}")
    
    def should_continue(self, history: list[EvalRunSummary]) -> tuple[bool, str]:
        """
        Determine if the refinement loop should continue.
        
        Returns:
            (should_continue, reason)
        """
        if not history:
            return True, "Starting first iteration"
        
        latest = history[-1]
        
        # Check max iterations
        if len(history) >= self.max_iterations:
            return False, f"Reached maximum iterations ({self.max_iterations})"
        
        # Check convergence threshold
        if latest.pass_rate >= self.convergence_threshold:
            return False, f"Reached target pass rate ({latest.pass_rate:.1%} >= {self.convergence_threshold:.1%})"
        
        # Check improvement plateau
        if len(history) >= self.patience:
            old_rate = history[-self.patience].pass_rate
            improvement = latest.pass_rate - old_rate
            if improvement < self.improvement_min:
                return False, f"Improvement plateau (only {improvement:.1%} in last {self.patience} iterations)"
        
        return True, "Continuing refinement"
    
    async def run_iteration(
        self,
        current_prompt: PromptVersion,
        test_cases: list[TestCase],
        skip_human_review: bool = False,
    ) -> tuple[PromptVersion, EvalRunSummary]:
        """
        Execute one iteration of the refinement loop.
        
        Returns:
            (new_prompt_version, eval_summary)
        """
        console.print(f"\n[bold blue]═══ Iteration with Prompt v{current_prompt.version_number} ═══[/bold blue]\n")
        
        # 1. MEASURE: Run evaluation
        console.print("[bold]Phase 1: MEASURE[/bold]")
        summary = await self.run_evaluation(current_prompt, test_cases)
        
        # If all passed, no need to continue
        if summary.failed == 0:
            console.print("[green]✓ All tests passed! No refinement needed.[/green]")
            return current_prompt, summary
        
        # 2. ANALYZE: Identify failure patterns
        console.print("\n[bold]Phase 2: ANALYZE[/bold]")
        analysis = self.failure_analyzer.analyze_failures(summary.results)
        
        console.print(f"Identified {len(analysis['fix_recommendations'])} failure patterns")
        for rec in analysis['fix_recommendations']:
            console.print(f"  • {rec['category']}: {rec['count']} cases → {rec['fix_strategy']}")
        
        # 3. HUMAN REVIEW: Review all failures
        annotations = []
        refinement_feedback = None
        if not skip_human_review:
            console.print("\n[bold]Phase 3: HUMAN REVIEW[/bold]")
            annotations = self.human_reviewer.review_all_failures(
                eval_results=summary.results,
                run_id=summary.run_id,
            )
            # Check for refinement feedback file
            refinement_feedback = self.human_reviewer.load_refinement_feedback(summary.run_id)
        
        # 4. IMPROVE: Generate optimized prompt
        console.print("\n[bold]Phase 4: IMPROVE[/bold]")
        new_prompt = self.prompt_optimizer.create_optimized_version(
            current_version=current_prompt,
            failure_analysis=analysis,
            human_annotations=annotations if annotations else None,
            refinement_feedback=refinement_feedback,
        )
        
        return new_prompt, summary
    
    async def run_loop(
        self,
        initial_prompt_version: int = None,
        dataset_name: str = None,
        skip_human_review: bool = False,
        remote_prompt_id: str = None,
    ):
        """
        Run the full refinement loop until convergence.
        
        Args:
            initial_prompt_version: Starting prompt version (default: latest)
            dataset_name: Test dataset to use (default: default dataset)
            skip_human_review: Skip human review phase (for automated testing)
            remote_prompt_id: If provided, push final prompt to remote API
        """
        console.print("\n[bold magenta]╔══════════════════════════════════════════╗[/bold magenta]")
        console.print("[bold magenta]║   PROMPT REFINEMENT WORKFLOW STARTED     ║[/bold magenta]")
        console.print("[bold magenta]╚══════════════════════════════════════════╝[/bold magenta]\n")
        
        # Load initial prompt
        if initial_prompt_version:
            current_prompt = self.prompt_manager.get_version(initial_prompt_version)
            if not current_prompt:
                console.print(f"[red]Error: Prompt version {initial_prompt_version} not found[/red]")
                return
        else:
            current_prompt = self.prompt_manager.get_latest_version()
            if not current_prompt:
                console.print("[red]Error: No prompt versions found. Create one first.[/red]")
                return
        
        # Load test cases
        test_cases = self.test_case_manager.load_test_cases(dataset_name)
        if not test_cases:
            console.print("[red]Error: No test cases found. Add some first.[/red]")
            return
        
        console.print(f"Starting with prompt v{current_prompt.version_number}")
        console.print(f"Test dataset: {len(test_cases)} cases")
        console.print(f"Convergence target: {self.convergence_threshold:.0%} pass rate")
        console.print(f"Max iterations: {self.max_iterations}")
        
        # Run the loop
        history = []
        iteration = 0
        
        while True:
            iteration += 1
            console.print(f"\n[bold]{'═' * 50}[/bold]")
            console.print(f"[bold]ITERATION {iteration}[/bold]")
            console.print(f"[bold]{'═' * 50}[/bold]")
            
            # Run one iteration
            new_prompt, summary = await self.run_iteration(
                current_prompt=current_prompt,
                test_cases=test_cases,
                skip_human_review=skip_human_review,
            )
            history.append(summary)
            
            # Check convergence
            should_continue, reason = self.should_continue(history)
            
            if not should_continue:
                console.print(f"\n[bold green]✓ Refinement complete: {reason}[/bold green]")
                break
            
            console.print(f"\n[dim]{reason}[/dim]")
            current_prompt = new_prompt
        
        # Final summary
        self._display_loop_summary(history)
        
        # Push to remote if requested
        if remote_prompt_id and history:
            best_prompt = self._get_best_prompt(history)
            if best_prompt:
                console.print(f"\n[bold]Pushing best prompt to remote API...[/bold]")
                await self.prompt_optimizer.push_optimized_prompt(best_prompt, remote_prompt_id)
        
        return history
    
    def _get_best_prompt(self, history: list[EvalRunSummary]) -> Optional[PromptVersion]:
        """Get the prompt version with the best pass rate."""
        if not history:
            return None
        
        best_summary = max(history, key=lambda s: s.pass_rate)
        return self.prompt_manager.get_version_by_id(best_summary.prompt_version_id)
    
    def _display_loop_summary(self, history: list[EvalRunSummary]):
        """Display summary of the entire refinement loop."""
        console.print("\n[bold magenta]╔══════════════════════════════════════════╗[/bold magenta]")
        console.print("[bold magenta]║       REFINEMENT LOOP SUMMARY            ║[/bold magenta]")
        console.print("[bold magenta]╚══════════════════════════════════════════╝[/bold magenta]\n")
        
        if not history:
            console.print("[dim]No iterations completed[/dim]")
            return
        
        # Progress table
        table = Table(title="Iteration History")
        table.add_column("Iteration", style="cyan")
        table.add_column("Pass Rate", style="yellow")
        table.add_column("Passed", style="green")
        table.add_column("Failed", style="red")
        table.add_column("Avg Score", style="blue")
        
        for i, summary in enumerate(history, 1):
            table.add_row(
                str(i),
                f"{summary.pass_rate:.1%}",
                str(summary.passed),
                str(summary.failed),
                f"{summary.avg_score:.2f}",
            )
        
        console.print(table)
        
        # Improvement
        if len(history) > 1:
            initial_rate = history[0].pass_rate
            final_rate = history[-1].pass_rate
            improvement = final_rate - initial_rate
            console.print(f"\n[bold]Total Improvement: {improvement:+.1%}[/bold]")
            console.print(f"  Initial: {initial_rate:.1%} → Final: {final_rate:.1%}")
