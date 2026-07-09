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

# "C2L" in the ANSI Shadow figlet style; rendered in the welcome screen.
WORDMARK = r"""
 ██████╗██████╗ ██╗
██╔════╝╚════██╗██║
██║      █████╔╝██║
██║     ██╔═══╝ ██║
╚██████╗███████╗███████╗
 ╚═════╝╚══════╝╚══════╝"""


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


def welcome(*, version: str, repo_id: Optional[str] = None,
            backend: str = "", adapter: Optional[str] = None,
            device: Optional[str] = None) -> None:
    """Stylish start screen: wordmark, live session status, and quickstart."""
    ready = bool(adapter)
    if _RICH:
        from rich.table import Table

        _console.print(Text(WORDMARK, style="bold cyan"))
        _console.print(
            f"[bold]code2lora[/bold] [dim]· on-device, repo-aware code "
            f"completion · v{version}[/dim]\n")

        status = Table.grid(padding=(0, 2))
        status.add_column(style="dim", justify="right", no_wrap=True)
        status.add_column()
        status.add_row("repo", f"[cyan]{repo_id or '(none)'}[/cyan]")
        status.add_row("backend", f"{backend}")
        dot = "[green]●[/green]" if ready else "[yellow]○[/yellow]"
        adapter_txt = (f"{dot} [green]{adapter[:12]}…[/green]" if ready
                       else f"{dot} [yellow]none[/yellow] "
                            f"[dim]— run[/dim] [bold]/adapt[/bold]")
        status.add_row("adapter", adapter_txt)
        if device:
            status.add_row("device", f"[dim]{device}[/dim]")
        _console.print(status)

        quick = Table.grid(padding=(0, 2))
        quick.add_column(style="bold cyan", justify="right", no_wrap=True)
        quick.add_column()
        quick.add_row("/adapt", "build the adapter for this repo")
        quick.add_row("<prompt>",
                      "type code to complete  [dim]e.g.[/dim]  "
                      "[green]assert add(2, 3) ==[/green]")
        quick.add_row("/help", "all commands   [dim]·[/dim]   /quit to exit")
        _console.rule(style="dim")
        _console.print(quick)
        return

    # ---- plain fallback ----
    _plain(WORDMARK, "cyan")
    _plain(f"code2lora  ·  on-device, repo-aware code completion  ·  v{version}\n",
           "bold")
    _plain(f"  repo     {repo_id or '(none)'}")
    _plain(f"  backend  {backend}")
    _plain(f"  adapter  {'● ' + adapter[:12] + '…' if ready else '○ none — run /adapt'}")
    if device:
        _plain(f"  device   {device}")
    _plain("  " + "-" * 52, "dim")
    _plain("  /adapt      build the adapter for this repo")
    _plain("  <prompt>    type code to complete  (e.g. assert add(2, 3) ==)")
    _plain("  /help       all commands  ·  /quit to exit")


def prompt_text(repo_id: Optional[str] = None, ready: bool = False) -> str:
    """Return an ANSI-styled input prompt reflecting session state."""
    name = repo_id or "c2l"
    dot = (f"{_C['green']}●{_C['reset']}" if ready
           else f"{_C['yellow']}○{_C['reset']}")
    return f"\n{dot} {_C['bold']}{_C['cyan']}{name}{_C['reset']} ▸ "


# Command reference grouped for the /help screen: (command, description).
_HELP_GROUPS = (
    ("Adapter", (
        ("/adapt [repo]", "build / update the adapter (incremental)"),
        ("/adapt --full", "force a full re-walk from scratch"),
        ("/adapt --local", "update from uncommitted local changes"),
        ("/load [path|fp]", "load a cached adapter (no regeneration)"),
        ("/export-gguf", "convert the adapter → GGUF for llama.cpp"),
    )),
    ("Session", (
        ("/backend [name]", "4bit · 8bit · hf · gguf"),
        ("/task [name]", "assert_rhs · qa · …"),
        ("/tokens [n]", "max new tokens to generate"),
        ("/context add <file>", "pin a file into every prompt"),
        ("/status", "show the current session state"),
    )),
    ("General", (
        ("/setup", "(re)download models + base GGUF + llama.cpp"),
        ("/help", "show this help"),
        ("/clear", "clear the screen"),
        ("/quit", "leave  (also /exit, Ctrl-D)"),
    )),
)


def help_panel() -> None:
    """Render the grouped command reference."""
    if _RICH:
        from rich.table import Table

        for title, rows in _HELP_GROUPS:
            t = Table.grid(padding=(0, 2))
            t.add_column(style="bold cyan", justify="right", no_wrap=True)
            t.add_column()
            for cmd, desc in rows:
                t.add_row(cmd, desc)
            _console.print(f"\n[bold]{title}[/bold]")
            _console.print(t)
        _console.print(
            "\n[dim]Prompting[/dim]  type a code prefix to complete "
            "(an assertion start, a signature + docstring, …).\n"
            "[dim]         [/dim] inline [cyan]@path/to/file.py[/cyan] "
            "pins that file into the current prompt.\n")
        return

    for title, rows in _HELP_GROUPS:
        _plain(f"\n{title}", "bold")
        for cmd, desc in rows:
            _plain(f"  {cmd:<20} {desc}")
    _plain("\nPrompting: type a code prefix to complete; inline @path pins a file.\n",
           "dim")


def keyvals(title: str, mapping) -> None:
    """Render a titled key/value block (used by /status)."""
    if _RICH:
        from rich.table import Table

        t = Table.grid(padding=(0, 2))
        t.add_column(style="dim", justify="right", no_wrap=True)
        t.add_column()
        for k, v in mapping.items():
            t.add_row(str(k), str(v))
        _console.print(f"\n[bold]{title}[/bold]")
        _console.print(t)
        return
    _plain(f"\n{title}", "bold")
    for k, v in mapping.items():
        _plain(f"  {str(k):<16} {v}")


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
