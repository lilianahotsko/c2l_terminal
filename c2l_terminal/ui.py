"""Terminal rendering helpers.

Uses ``rich`` when available for nice panels/markdown/spinners, and degrades
gracefully to plain ANSI output otherwise so the tool still runs in minimal
environments.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from typing import Iterator, Optional

try:  # optional, but recommended
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich.text import Text
    _RICH = True
    _console = Console()
except Exception:  # pragma: no cover - fallback path
    _RICH = False
    _console = None


# ANSI fallbacks
_C = {
    "dim": "\033[2m", "bold": "\033[1m", "cyan": "\033[36m",
    "green": "\033[32m", "yellow": "\033[33m", "red": "\033[31m",
    "magenta": "\033[35m", "reset": "\033[0m",
}


def _plain(msg: str, color: Optional[str] = None) -> None:
    if color and sys.stdout.isatty():
        sys.stdout.write(f"{_C.get(color, '')}{msg}{_C['reset']}\n")
    else:
        sys.stdout.write(msg + "\n")
    sys.stdout.flush()


def banner(title: str, subtitle: str = "") -> None:
    if _RICH:
        body = Text(title, style="bold cyan")
        if subtitle:
            body.append("\n" + subtitle, style="dim")
        _console.print(Panel(body, border_style="cyan", expand=False))
    else:
        _plain("=" * 60, "cyan")
        _plain(f"  {title}", "bold")
        if subtitle:
            _plain(f"  {subtitle}", "dim")
        _plain("=" * 60, "cyan")


def info(msg: str) -> None:
    _console.print(f"[dim]{msg}[/dim]") if _RICH else _plain(msg, "dim")


def ok(msg: str) -> None:
    _console.print(f"[green]{msg}[/green]") if _RICH else _plain(msg, "green")


def warn(msg: str) -> None:
    _console.print(f"[yellow]{msg}[/yellow]") if _RICH else _plain(msg, "yellow")


def error(msg: str) -> None:
    _console.print(f"[red]{msg}[/red]") if _RICH else _plain(msg, "red")


def rule(label: str = "") -> None:
    if _RICH:
        _console.rule(label, style="dim")
    else:
        _plain(f"-- {label} " + "-" * max(0, 50 - len(label)), "dim")


def assistant_code(text: str, lang: str = "python") -> None:
    """Render a model completion as a syntax-highlighted block."""
    text = text.rstrip("\n")
    if _RICH:
        try:
            _console.print(Syntax(text, lang, theme="ansi_dark",
                                  word_wrap=True, background_color="default"))
            return
        except Exception:
            pass
    _plain(text, "green")


def markdown(text: str) -> None:
    _console.print(Markdown(text)) if _RICH else _plain(text)


@contextmanager
def status(msg: str) -> Iterator[None]:
    """Spinner context (rich) or a simple printed line (fallback)."""
    if _RICH:
        with _console.status(f"[cyan]{msg}[/cyan]", spinner="dots"):
            yield
    else:
        _plain(msg + " ...", "cyan")
        yield


def make_progress():
    """Return a ``(callback, finish)`` pair for c2l's ``progress(msg, frac)``."""
    if _RICH:
        from rich.progress import (BarColumn, Progress, SpinnerColumn,
                                    TextColumn, TimeElapsedColumn)
        prog = Progress(
            SpinnerColumn(), TextColumn("[cyan]{task.description}"),
            BarColumn(), TimeElapsedColumn(), console=_console, transient=True)
        prog.start()
        task_id = prog.add_task("starting", total=1.0)

        def cb(msg: str, frac: float) -> None:
            prog.update(task_id, description=msg[:48], completed=min(max(frac, 0.0), 1.0))

        def finish() -> None:
            prog.stop()

        return cb, finish

    def cb(msg: str, frac: float) -> None:  # plain fallback
        sys.stdout.write(f"\r  [{frac*100:5.1f}%] {msg[:60]:<60}")
        sys.stdout.flush()
        if frac >= 1.0:
            sys.stdout.write("\n")

    return cb, (lambda: None)
