"""
Prompt Refinement Workflow CLI

An automated workflow for refining AI agent prompts through:
- Test case management
- Agent evaluation via API
- GPT-4 as LLM judge
- Human feedback collection
- Automated prompt optimization
- Iterative refinement until convergence

Usage:
    python main.py --help
    python main.py test add --input "question" --expected "answer"
    python main.py prompt create --system-prompt "You are..."
    python main.py eval run --prompt-version 1
    python main.py review --run-id abc123
    python main.py optimize --from-version 1 --run-id abc123
    python main.py loop start --prompt-version 1
"""
import asyncio
import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from config import settings
from core import (
    PromptManager, TestCaseManager, AgentInvoker, 
    IterationController, PromptVersion
)
from graders import LLMJudge
from refinement import FailureAnalyzer, PromptOptimizer
from human_feedback import HumanReviewer

console = Console()


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """Prompt Refinement Workflow - Automated agent prompt optimization."""
    pass


# ============== Test Case Commands ==============

@cli.group()
def test():
    """Manage test cases (questions with expected outputs)."""
    pass


@test.command("add")
@click.option("--input", "-i", "input_text", required=True, help="The question/input to test")
@click.option("--expected", "-e", "expected_output", required=True, help="Expected output from agent")
@click.option("--tags", "-t", multiple=True, help="Tags for categorization")
@click.option("--difficulty", "-d", type=click.Choice(["easy", "medium", "hard"]), default="medium")
@click.option("--dataset", help="Dataset name (default: dataset)")
def test_add(input_text, expected_output, tags, difficulty, dataset):
    """Add a new test case."""
    manager = TestCaseManager()
    test_case = manager.add_test_case(
        input_text=input_text,
        expected_output=expected_output,
        tags=list(tags),
        difficulty=difficulty,
        dataset_name=dataset,
    )
    console.print(f"[green]✓[/green] Added test case: {test_case.id}")


@test.command("list")
@click.option("--dataset", help="Dataset name")
@click.option("--tag", "-t", help="Filter by tag")
@click.option("--difficulty", "-d", type=click.Choice(["easy", "medium", "hard"]))
def test_list(dataset, tag, difficulty):
    """List all test cases."""
    manager = TestCaseManager()
    
    if tag or difficulty:
        cases = manager.filter_cases(
            dataset_name=dataset,
            tags=[tag] if tag else None,
            difficulty=difficulty,
        )
    else:
        cases = manager.load_test_cases(dataset)
    
    if not cases:
        console.print("[yellow]No test cases found.[/yellow]")
        return
    
    table = Table(title=f"Test Cases ({len(cases)} total)")
    table.add_column("ID", style="cyan")
    table.add_column("Input", style="white", max_width=40)
    table.add_column("Expected", style="green", max_width=40)
    table.add_column("Tags", style="yellow")
    table.add_column("Difficulty", style="blue")
    
    for case in cases:
        table.add_row(
            case.id,
            case.input_text[:40] + "..." if len(case.input_text) > 40 else case.input_text,
            case.expected_output[:40] + "..." if len(case.expected_output) > 40 else case.expected_output,
            ", ".join(case.tags),
            case.difficulty,
        )
    
    console.print(table)


@test.command("import")
@click.argument("file_path", type=click.Path(exists=True))
@click.option("--dataset", help="Dataset name")
def test_import(file_path, dataset):
    """Import test cases from a JSONL file."""
    manager = TestCaseManager()
    count = manager.import_jsonl(file_path, dataset)
    console.print(f"[green]✓[/green] Imported {count} test cases")


@test.command("export")
@click.argument("output_path", type=click.Path())
@click.option("--dataset", help="Dataset name")
def test_export(output_path, dataset):
    """Export test cases to a JSONL file."""
    manager = TestCaseManager()
    count = manager.export_jsonl(output_path, dataset)
    console.print(f"[green]✓[/green] Exported {count} test cases to {output_path}")


@test.command("stats")
@click.option("--dataset", help="Dataset name")
def test_stats(dataset):
    """Show test dataset statistics."""
    manager = TestCaseManager()
    stats = manager.get_dataset_stats(dataset)
    
    console.print(Panel(f"[bold]Test Dataset Statistics[/bold]"))
    console.print(f"Total cases: {stats['total_cases']}")
    
    if stats['by_difficulty']:
        console.print("\n[bold]By Difficulty:[/bold]")
        for diff, count in stats['by_difficulty'].items():
            console.print(f"  • {diff}: {count}")
    
    if stats['by_tag']:
        console.print("\n[bold]By Tag:[/bold]")
        for tag, count in stats['by_tag'].items():
            console.print(f"  • {tag}: {count}")


# ============== Prompt Commands ==============

@cli.group()
def prompt():
    """Manage prompt versions."""
    pass


@prompt.command("create")
@click.option("--system-prompt", "-s", required=True, help="System prompt text")
@click.option("--instruction", "-i", help="Instruction template with {{user_input}}")
@click.option("--notes", "-n", help="Notes about this version")
def prompt_create(system_prompt, instruction, notes):
    """Create a new prompt version."""
    manager = PromptManager()
    version = manager.create_version(
        system_prompt=system_prompt,
        instruction_template=instruction or "",
        optimization_notes=notes or "Initial version",
    )
    console.print(f"[green]✓[/green] Created prompt version {version.version_number}")
    console.print(f"  ID: {version.id}")


@prompt.command("list")
def prompt_list():
    """List all prompt versions."""
    manager = PromptManager()
    versions = manager.list_versions()
    
    if not versions:
        console.print("[yellow]No prompt versions found.[/yellow]")
        return
    
    table = Table(title="Prompt Versions")
    table.add_column("Version", style="cyan")
    table.add_column("ID", style="dim")
    table.add_column("Parent", style="blue")
    table.add_column("Pass Rate", style="green")
    table.add_column("Notes", style="yellow", max_width=40)
    table.add_column("Created", style="dim")
    
    for v in versions:
        parent = f"v{v.parent_version_id[:4]}..." if v.parent_version_id else "-"
        pass_rate = f"{v.pass_rate:.1%}" if v.pass_rate else "-"
        table.add_row(
            f"v{v.version_number}",
            v.id,
            parent,
            pass_rate,
            v.optimization_notes[:40] + "..." if len(v.optimization_notes) > 40 else v.optimization_notes,
            v.created_at[:10],
        )
    
    console.print(table)


@prompt.command("show")
@click.argument("version", type=int)
def prompt_show(version):
    """Show details of a prompt version."""
    manager = PromptManager()
    prompt = manager.get_version(version)
    
    if not prompt:
        console.print(f"[red]Version {version} not found[/red]")
        return
    
    console.print(Panel(f"[bold]Prompt Version {version}[/bold]", style="blue"))
    console.print(f"[bold]ID:[/bold] {prompt.id}")
    console.print(f"[bold]Created:[/bold] {prompt.created_at}")
    console.print(f"[bold]Parent:[/bold] {prompt.parent_version_id or 'None'}")
    console.print(f"[bold]Pass Rate:[/bold] {prompt.pass_rate or 'Not evaluated'}")
    
    console.print("\n[bold]System Prompt:[/bold]")
    console.print(Panel(prompt.system_prompt, style="green"))
    
    if prompt.instruction_template:
        console.print("\n[bold]Instruction Template:[/bold]")
        console.print(Panel(prompt.instruction_template, style="cyan"))
    
    if prompt.few_shot_examples:
        console.print("\n[bold]Few-shot Examples:[/bold]")
        for i, ex in enumerate(prompt.few_shot_examples, 1):
            console.print(f"  {i}. Input: {ex.get('input', '')[:50]}...")
            console.print(f"     Output: {ex.get('output', '')[:50]}...")
    
    console.print(f"\n[bold]Optimization Notes:[/bold] {prompt.optimization_notes}")


@prompt.command("lineage")
@click.argument("version", type=int)
def prompt_lineage(version):
    """Show the lineage (ancestry) of a prompt version."""
    manager = PromptManager()
    lineage = manager.get_lineage(version)
    
    if not lineage:
        console.print(f"[red]Version {version} not found[/red]")
        return
    
    console.print(Panel("[bold]Prompt Lineage[/bold]", style="blue"))
    
    for i, v in enumerate(lineage):
        prefix = "└── " if i == len(lineage) - 1 else "├── "
        indent = "    " * i
        console.print(f"{indent}{prefix}v{v.version_number}: {v.optimization_notes[:50]}")


@prompt.command("sync")
@click.option("--prompt-id", help="Remote prompt ID to fetch")
def prompt_sync(prompt_id):
    """Sync prompt from remote API."""
    manager = PromptManager()
    asyncio.run(manager.sync_from_api(prompt_id))


@prompt.command("push")
@click.argument("version", type=int)
@click.option("--prompt-id", required=True, help="Remote prompt ID to update")
def prompt_push(version, prompt_id):
    """Push a prompt version to remote API."""
    manager = PromptManager()
    asyncio.run(manager.push_to_api(version, prompt_id))


# ============== Evaluation Commands ==============

@cli.group()
def eval():
    """Run evaluations and view results."""
    pass


@eval.command("run")
@click.option("--prompt-version", "-p", type=int, help="Prompt version to test (default: latest)")
@click.option("--dataset", help="Test dataset name")
def eval_run(prompt_version, dataset):
    """Run evaluation with specified prompt version."""
    controller = IterationController()
    prompt_manager = PromptManager()
    test_manager = TestCaseManager()
    
    # Get prompt
    if prompt_version:
        prompt = prompt_manager.get_version(prompt_version)
        if not prompt:
            console.print(f"[red]Prompt version {prompt_version} not found[/red]")
            return
    else:
        prompt = prompt_manager.get_latest_version()
        if not prompt:
            console.print("[red]No prompt versions found. Create one first.[/red]")
            return
    
    # Get test cases
    test_cases = test_manager.load_test_cases(dataset)
    if not test_cases:
        console.print("[red]No test cases found. Add some first.[/red]")
        return
    
    # Run evaluation
    summary = asyncio.run(controller.run_evaluation(prompt, test_cases))
    console.print(f"\n[green]✓[/green] Evaluation complete. Run ID: {summary.run_id}")


@eval.command("show")
@click.argument("run_id")
def eval_show(run_id):
    """Show results of an evaluation run."""
    controller = IterationController()
    summary = controller.load_run_summary(run_id)
    
    if not summary:
        console.print(f"[red]Run {run_id} not found[/red]")
        return
    
    controller._display_eval_summary(summary)
    
    # Show failures
    failures = [r for r in summary.results if not r.passed]
    if failures:
        console.print(f"\n[bold]Failed Cases ({len(failures)}):[/bold]")
        for f in failures[:5]:  # Show first 5
            console.print(f"  • {f.test_case_id}: {f.grader_result.feedback[:60]}...")
        if len(failures) > 5:
            console.print(f"  ... and {len(failures) - 5} more")


# ============== Review Commands ==============

@cli.command("review")
@click.option("--run-id", required=True, help="Evaluation run ID to review")
@click.option("--quick", is_flag=True, help="Quick review mode (just good/bad)")
@click.option("--detailed", is_flag=True, help="Detailed review with NLP feedback")
def review(run_id, quick, detailed):
    """Review all failures from an evaluation run."""
    controller = IterationController()
    reviewer = HumanReviewer()
    
    summary = controller.load_run_summary(run_id)
    if not summary:
        console.print(f"[red]Run {run_id} not found[/red]")
        return
    
    if quick:
        reviewer.quick_review(summary.results, run_id)
    elif detailed:
        annotations, refinement_feedback = reviewer.detailed_review(summary.results, run_id)
        console.print(f"\n[green]✓[/green] Detailed review complete with {len(annotations)} annotations")
        if refinement_feedback and refinement_feedback.general_instructions:
            console.print("[dim]Refinement feedback saved for optimization.[/dim]")
    else:
        reviewer.review_all_failures(summary.results, run_id)


# ============== Optimize Commands ==============

@cli.command("optimize")
@click.option("--from-version", "-f", type=int, required=True, help="Prompt version to optimize")
@click.option("--run-id", "-r", required=True, help="Evaluation run ID with failures")
@click.option("--push", "remote_prompt_id", help="Push result to remote API with this prompt ID")
def optimize(from_version, run_id, remote_prompt_id):
    """Generate optimized prompt based on evaluation failures."""
    prompt_manager = PromptManager()
    controller = IterationController()
    reviewer = HumanReviewer()
    analyzer = FailureAnalyzer()
    optimizer = PromptOptimizer()
    
    # Load prompt
    prompt = prompt_manager.get_version(from_version)
    if not prompt:
        console.print(f"[red]Prompt version {from_version} not found[/red]")
        return
    
    # Load evaluation
    summary = controller.load_run_summary(run_id)
    if not summary:
        console.print(f"[red]Run {run_id} not found[/red]")
        return
    
    # Load annotations
    annotations = reviewer.load_annotations(run_id)
    
    # Load refinement feedback (from detailed review)
    refinement_feedback = reviewer.load_refinement_feedback(run_id)
    if refinement_feedback:
        console.print(f"[cyan]Found refinement feedback with {len(refinement_feedback.priority_issues)} priority issues[/cyan]")
    
    # Analyze failures
    analysis = analyzer.analyze_failures(summary.results, annotations)
    
    # Generate optimized prompt
    new_version = optimizer.create_optimized_version(
        current_version=prompt,
        failure_analysis=analysis,
        human_annotations=annotations,
        refinement_feedback=refinement_feedback,
    )
    
    console.print(f"\n[green]✓[/green] Created optimized prompt v{new_version.version_number}")
    
    # Push to remote if requested
    if remote_prompt_id:
        asyncio.run(optimizer.push_optimized_prompt(new_version, remote_prompt_id))


# ============== Loop Commands ==============

@cli.group()
def loop():
    """Run the full refinement loop."""
    pass


@loop.command("start")
@click.option("--prompt-version", "-p", type=int, help="Starting prompt version (default: latest)")
@click.option("--dataset", help="Test dataset name")
@click.option("--skip-review", is_flag=True, help="Skip human review (automated mode)")
@click.option("--push", "remote_prompt_id", help="Push final prompt to remote API")
def loop_start(prompt_version, dataset, skip_review, remote_prompt_id):
    """Start the full refinement loop."""
    controller = IterationController()
    
    asyncio.run(controller.run_loop(
        initial_prompt_version=prompt_version,
        dataset_name=dataset,
        skip_human_review=skip_review,
        remote_prompt_id=remote_prompt_id,
    ))


@loop.command("config")
def loop_config():
    """Show current loop configuration."""
    console.print(Panel("[bold]Loop Configuration[/bold]", style="blue"))
    console.print(f"Max Iterations: {settings.MAX_ITERATIONS}")
    console.print(f"Convergence Threshold: {settings.CONVERGENCE_THRESHOLD:.0%}")
    console.print(f"Min Improvement: {settings.IMPROVEMENT_MIN:.1%}")
    console.print(f"Patience: {settings.PATIENCE} iterations")
    console.print(f"Pass Score Threshold: {settings.PASS_SCORE_THRESHOLD}/10")


# ============== Config Commands ==============

@cli.command("config")
def config():
    """Show current configuration."""
    console.print(Panel("[bold]Configuration[/bold]", style="blue"))
    
    console.print("\n[bold]API Endpoints:[/bold]")
    console.print(f"  Agent API: {settings.AGENT_API_URL}")
    console.print(f"  Prompt GET: {settings.PROMPT_API_GET_URL}")
    console.print(f"  Prompt UPDATE: {settings.PROMPT_API_UPDATE_URL}")
    
    console.print("\n[bold]Directories:[/bold]")
    console.print(f"  Test Cases: {settings.TEST_CASES_DIR}")
    console.print(f"  Prompts: {settings.PROMPTS_DIR}")
    console.print(f"  Results: {settings.RESULTS_DIR}")
    
    # Validate
    missing = settings.validate()
    if missing:
        console.print(f"\n[yellow]⚠ Missing configuration:[/yellow]")
        for m in missing:
            console.print(f"  • {m}")
    else:
        console.print("\n[green]✓ All required configuration present[/green]")


if __name__ == "__main__":
    cli()
