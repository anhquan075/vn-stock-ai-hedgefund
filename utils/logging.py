"""Lightweight logging utilities with optional Rich formatting.

Falls back to plain prints if Rich is unavailable.
"""

from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from colorama import Fore, Style, init
from tabulate import tabulate

_console = Console()
init(autoreset=True)


def log_info(message: str) -> None:
    if _console:
        _console.log(message)
    else:
        print(message)


def log_error(message: str) -> None:
    if _console:
        _console.log(f"[red]{message}[/red]")
    else:
        print(message)


def log_markdown(md_text: str) -> None:
    if _console:
        _console.print(Markdown(md_text))
    else:
        print(md_text)


def log_panel(title: str, content: str) -> None:
    if _console:
        _console.print(Panel.fit(content, title=title))
    else:
        print(f"--- {title} ---\n{content}")


def log_dataframe(df: Any, title: str | None = None) -> None:
    if _console:
        try:
            table = Table(title=title)
            for col in df.columns:
                table.add_column(str(col))
            for _, row in df.iterrows():
                table.add_row(*[str(v) for v in row.tolist()])
            _console.print(table)
            return
        except Exception as err:
            _console.log(
                f"[yellow]Fallback to plain dataframe render due to error:[/yellow] {err}"
            )
    # Fallback
    if title:
        print(f"--- {title} ---")
    try:
        print(df.to_string(index=False))
    except Exception:
        print(str(df))


def log_markdown_panel(title: str, md_text: str) -> None:
    """Render markdown content inside a titled panel.

    Falls back to plain text if Rich is unavailable.
    """
    if _console:
        _console.print(Panel.fit(Markdown(md_text), title=title))
    else:
        print(f"--- {title} ---\n{md_text}")


def print_backtest_stats(stats: dict[str, Any]) -> None:
    """Render backtest statistics in a colored table similar to ai-hedge-fund."""

    def _fmt(name: str, value: float, invert: bool = False) -> list[str]:
        color = Fore.GREEN if (value >= 0) ^ invert else Fore.RED
        if "drawdown" in name.lower():
            color = Fore.RED if value > 0 else Fore.GREEN
        return [name, f"{color}{value:,.2f}{Style.RESET_ALL}"]

    rows = [
        _fmt("Final Equity", stats.get("final_equity", 0.0)),
        _fmt("CAGR", stats.get("cagr", 0.0)),
        _fmt("Max Drawdown", stats.get("max_drawdown", 0.0), invert=True),
        _fmt("Sharpe", stats.get("sharpe", 0.0)),
    ]

    print(
        tabulate(
            rows,
            headers=[f"{Fore.WHITE}Metric", f"{Fore.WHITE}Value"],
            tablefmt="grid",
            colalign=("left", "right"),
        )
    )
