#!/usr/bin/env python3
"""
High-concurrency API stress tester for OpenAI-compatible endpoints.

Usage:
    python main.py --api-key YOUR_POE_API_KEY [options]

Examples:
    # Basic run with 20 concurrent workers
    python main.py --api-key sk-xxx --concurrency 20

    # Full blast: 50 concurrent, all 100 questions
    python main.py --api-key sk-xxx --concurrency 50 --num-questions 100

    # Quick smoke test: 5 questions, 3 concurrent
    python main.py --api-key sk-xxx --concurrency 3 --num-questions 5
"""

import argparse
import asyncio
import os
import random
import time
from dataclasses import dataclass, field

import openai
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

from questions import QUESTIONS

console = Console()


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RequestResult:
    """Result of a single API request."""
    question_idx: int
    question: str
    status: str  # "success" | "error"
    latency: float  # seconds
    tokens_prompt: int = 0
    tokens_completion: int = 0
    error_message: str = ""
    response_preview: str = ""


@dataclass
class Stats:
    """Aggregated statistics across all requests."""
    results: list[RequestResult] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def successes(self) -> int:
        return sum(1 for r in self.results if r.status == "success")

    @property
    def failures(self) -> int:
        return sum(1 for r in self.results if r.status == "error")

    @property
    def latencies(self) -> list[float]:
        return [r.latency for r in self.results if r.status == "success"]

    @property
    def avg_latency(self) -> float:
        lats = self.latencies
        return sum(lats) / len(lats) if lats else 0.0

    @property
    def min_latency(self) -> float:
        lats = self.latencies
        return min(lats) if lats else 0.0

    @property
    def max_latency(self) -> float:
        lats = self.latencies
        return max(lats) if lats else 0.0

    @property
    def p50_latency(self) -> float:
        return self._percentile(50)

    @property
    def p95_latency(self) -> float:
        return self._percentile(95)

    @property
    def p99_latency(self) -> float:
        return self._percentile(99)

    @property
    def total_prompt_tokens(self) -> int:
        return sum(r.tokens_prompt for r in self.results)

    @property
    def total_completion_tokens(self) -> int:
        return sum(r.tokens_completion for r in self.results)

    @property
    def wall_time(self) -> float:
        return self.end_time - self.start_time if self.end_time else 0.0

    @property
    def throughput(self) -> float:
        return self.successes / self.wall_time if self.wall_time else 0.0

    def _percentile(self, p: int) -> float:
        lats = sorted(self.latencies)
        if not lats:
            return 0.0
        k = (len(lats) - 1) * (p / 100)
        f = int(k)
        c = f + 1
        if c >= len(lats):
            return lats[f]
        return lats[f] + (k - f) * (lats[c] - lats[f])


# ---------------------------------------------------------------------------
# Core: single API request
# ---------------------------------------------------------------------------

async def send_request(
    client: openai.AsyncOpenAI,
    model: str,
    question: str,
    question_idx: int,
    semaphore: asyncio.Semaphore,
    max_tokens: int,
) -> RequestResult:
    """Send a single chat completion request, guarded by a semaphore."""
    async with semaphore:
        t0 = time.perf_counter()
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": question}],
                max_tokens=max_tokens,
            )
            latency = time.perf_counter() - t0

            choice = response.choices[0]
            usage = response.usage
            preview = (choice.message.content or "")[:120].replace("\n", " ")

            return RequestResult(
                question_idx=question_idx,
                question=question,
                status="success",
                latency=latency,
                tokens_prompt=usage.prompt_tokens if usage else 0,
                tokens_completion=usage.completion_tokens if usage else 0,
                response_preview=preview,
            )
        except Exception as exc:
            latency = time.perf_counter() - t0
            return RequestResult(
                question_idx=question_idx,
                question=question,
                status="error",
                latency=latency,
                error_message=f"{type(exc).__name__}: {exc}",
            )


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def build_summary_table(stats: Stats) -> Table:
    """Build a rich Table summarizing the test run."""
    table = Table(title="Concurrency Test Results", show_header=False, padding=(0, 2))
    table.add_column("Metric", style="bold cyan", min_width=28)
    table.add_column("Value", style="bold white", min_width=20)

    table.add_row("Total Requests", str(stats.total))
    table.add_row(
        "Successes",
        Text(str(stats.successes), style="bold green"),
    )
    table.add_row(
        "Failures",
        Text(str(stats.failures), style="bold red" if stats.failures else "bold green"),
    )
    table.add_row("", "")
    table.add_row("Wall Clock Time", f"{stats.wall_time:.2f}s")
    table.add_row("Throughput", f"{stats.throughput:.2f} req/s")
    table.add_row("", "")
    table.add_row("Avg Latency", f"{stats.avg_latency:.2f}s")
    table.add_row("Min Latency", f"{stats.min_latency:.2f}s")
    table.add_row("Max Latency", f"{stats.max_latency:.2f}s")
    table.add_row("P50 Latency", f"{stats.p50_latency:.2f}s")
    table.add_row("P95 Latency", f"{stats.p95_latency:.2f}s")
    table.add_row("P99 Latency", f"{stats.p99_latency:.2f}s")
    table.add_row("", "")
    table.add_row("Total Prompt Tokens", f"{stats.total_prompt_tokens:,}")
    table.add_row("Total Completion Tokens", f"{stats.total_completion_tokens:,}")
    table.add_row(
        "Total Tokens",
        f"{stats.total_prompt_tokens + stats.total_completion_tokens:,}",
    )

    return table


def build_error_table(stats: Stats) -> Table | None:
    """Build a table of failed requests, or None if no errors."""
    errors = [r for r in stats.results if r.status == "error"]
    if not errors:
        return None

    table = Table(title="Failed Requests", show_lines=True)
    table.add_column("#", style="dim", width=4)
    table.add_column("Question", style="yellow", max_width=50, overflow="ellipsis")
    table.add_column("Error", style="red", max_width=60, overflow="ellipsis")
    table.add_column("Latency", style="dim", width=8)

    for r in errors:
        table.add_row(
            str(r.question_idx),
            r.question[:50],
            r.error_message[:60],
            f"{r.latency:.2f}s",
        )

    return table


def build_per_request_table(stats: Stats, show_all: bool = False) -> Table:
    """Build a table showing per-request details."""
    table = Table(title="Per-Request Details", show_lines=False)
    table.add_column("#", style="dim", width=4)
    table.add_column("Status", width=8)
    table.add_column("Latency", width=10)
    table.add_column("Tokens", width=12)
    table.add_column("Question", max_width=40, overflow="ellipsis")
    table.add_column("Response Preview", max_width=50, overflow="ellipsis")

    results = stats.results if show_all else stats.results[:20]

    for r in sorted(results, key=lambda x: x.question_idx):
        status_text = Text("OK", style="green") if r.status == "success" else Text("FAIL", style="red")
        tokens = f"{r.tokens_prompt}+{r.tokens_completion}" if r.status == "success" else "-"
        preview = r.response_preview if r.status == "success" else r.error_message[:50]

        table.add_row(
            str(r.question_idx),
            status_text,
            f"{r.latency:.2f}s",
            tokens,
            r.question[:40],
            preview,
        )

    if not show_all and len(stats.results) > 20:
        table.add_row("...", "", "", "", f"({len(stats.results) - 20} more)", "")

    return table


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

async def run_test(
    api_key: str,
    base_url: str,
    model: str,
    concurrency: int,
    num_questions: int,
    max_tokens: int,
    shuffle: bool,
    show_all: bool,
) -> None:
    """Run the full concurrency test."""

    # Select and optionally shuffle questions
    selected = QUESTIONS[:num_questions]
    if shuffle:
        selected = selected.copy()
        random.shuffle(selected)

    console.print()
    console.print(
        Panel(
            f"[bold]API Concurrency Stress Test[/bold]\n\n"
            f"  Endpoint:     [cyan]{base_url}[/cyan]\n"
            f"  Model:        [cyan]{model}[/cyan]\n"
            f"  Questions:    [cyan]{len(selected)}[/cyan]\n"
            f"  Concurrency:  [cyan]{concurrency}[/cyan]\n"
            f"  Max Tokens:   [cyan]{max_tokens}[/cyan]\n"
            f"  Shuffle:      [cyan]{shuffle}[/cyan]",
            title="Configuration",
            border_style="blue",
        )
    )
    console.print()

    # Set up async client
    client = openai.AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
    )

    semaphore = asyncio.Semaphore(concurrency)
    stats = Stats()
    stats.start_time = time.perf_counter()

    # Progress bar
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )

    task_id = progress.add_task("Sending requests...", total=len(selected))

    # Fire all requests concurrently (semaphore limits actual parallelism)
    async def _tracked_request(idx: int, q: str) -> RequestResult:
        result = await send_request(client, model, q, idx, semaphore, max_tokens)
        stats.results.append(result)
        progress.advance(task_id)

        # Live status indicator
        if result.status == "success":
            progress.console.print(
                f"  [green]✓[/green] [dim]#{idx:>3}[/dim]  {result.latency:>6.2f}s  "
                f"[dim]{result.response_preview[:60]}[/dim]"
            )
        else:
            progress.console.print(
                f"  [red]✗[/red] [dim]#{idx:>3}[/dim]  {result.latency:>6.2f}s  "
                f"[red]{result.error_message[:60]}[/red]"
            )

        return result

    with progress:
        tasks = [
            _tracked_request(idx, q) for idx, q in enumerate(selected)
        ]
        await asyncio.gather(*tasks)

    stats.end_time = time.perf_counter()

    # --- Print results ---
    console.print()
    console.print(build_summary_table(stats))
    console.print()

    error_table = build_error_table(stats)
    if error_table:
        console.print(error_table)
        console.print()

    console.print(build_per_request_table(stats, show_all=show_all))
    console.print()

    # Final verdict
    if stats.failures == 0:
        console.print(
            Panel(
                f"[bold green]ALL {stats.total} REQUESTS SUCCEEDED[/bold green]\n"
                f"Throughput: {stats.throughput:.2f} req/s  |  "
                f"Avg latency: {stats.avg_latency:.2f}s  |  "
                f"P95: {stats.p95_latency:.2f}s",
                border_style="green",
            )
        )
    else:
        console.print(
            Panel(
                f"[bold red]{stats.failures}/{stats.total} REQUESTS FAILED[/bold red]\n"
                f"Success rate: {stats.successes/stats.total*100:.1f}%  |  "
                f"Throughput: {stats.throughput:.2f} req/s",
                border_style="red",
            )
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="High-concurrency stress tester for OpenAI-compatible APIs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("POE_API_KEY", ""),
        help="API key (default: $POE_API_KEY env var)",
    )
    parser.add_argument(
        "--base-url",
        default="https://api.poe.com/v1",
        help="API base URL (default: https://api.poe.com/v1)",
    )
    parser.add_argument(
        "--model",
        default="gpt-5.3-codex",
        help="Model name to use (default: gpt-5.3-codex)",
    )
    parser.add_argument(
        "--concurrency", "-c",
        type=int,
        default=20,
        help="Max concurrent requests (default: 20)",
    )
    parser.add_argument(
        "--num-questions", "-n",
        type=int,
        default=100,
        help="Number of questions to send (default: 100, max: 100)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Max tokens per response (default: 256)",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        default=True,
        help="Shuffle question order (default: True)",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_false",
        dest="shuffle",
        help="Don't shuffle questions",
    )
    parser.add_argument(
        "--show-all",
        action="store_true",
        help="Show all per-request details (default: first 20)",
    )

    args = parser.parse_args()

    if not args.api_key:
        console.print(
            "[bold red]Error:[/bold red] No API key provided. "
            "Use --api-key or set $POE_API_KEY environment variable."
        )
        raise SystemExit(1)

    args.num_questions = min(args.num_questions, len(QUESTIONS))

    asyncio.run(
        run_test(
            api_key=args.api_key,
            base_url=args.base_url,
            model=args.model,
            concurrency=args.concurrency,
            num_questions=args.num_questions,
            max_tokens=args.max_tokens,
            shuffle=args.shuffle,
            show_all=args.show_all,
        )
    )


if __name__ == "__main__":
    main()
