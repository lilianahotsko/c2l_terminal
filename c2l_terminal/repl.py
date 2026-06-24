"""Interactive REPL entry point for c2l-terminal."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__, commands, ui
from .session import Session


def _make_reader():
    """Return a callable ``read(prompt) -> str`` with history if available.

    Falls back to plain ``input`` when stdin is not a TTY (piped/scripted) or
    when prompt_toolkit is unavailable.
    """
    if sys.stdin.isatty():
        try:
            from prompt_toolkit import PromptSession
            from prompt_toolkit.history import FileHistory
            hist = Path("~/.cache/c2l/terminal_history").expanduser()
            hist.parent.mkdir(parents=True, exist_ok=True)
            ps = PromptSession(history=FileHistory(str(hist)))
            return lambda prompt: ps.prompt(prompt)
        except Exception:
            pass
    return lambda prompt: input(prompt)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="c2lt",
        description="Terminal coding assistant powered by quantized Code2LoRA.")
    p.add_argument("repo", nargs="?", default=None,
                   help="repo path or git URL (default: current directory)")
    p.add_argument("--task", default="assert_rhs")
    p.add_argument("--backend", default="4bit",
                   choices=["4bit", "8bit", "hf", "gguf"])
    p.add_argument("--tokens", type=int, default=64, dest="tokens")
    p.add_argument("--device", default=None, help="cuda | cpu (default: auto)")
    p.add_argument("--offline", action="store_true")
    p.add_argument("--adapt", action="store_true",
                   help="generate the adapter immediately on startup")
    p.add_argument("--version", action="version", version=f"c2lt {__version__}")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    session = Session(
        repo=args.repo, task=args.task, backend=args.backend,
        max_new_tokens=args.tokens, device=args.device, offline=args.offline)

    ui.banner(
        f"c2l-terminal {__version__}",
        "quantized Code2LoRA coding REPL  -  /help for commands, /quit to exit")

    detected = session.detect_repo()
    if detected:
        ui.info(f"repo: {detected}  (backend={session.backend}, task={session.task})")
    else:
        ui.warn("no git repo in the current directory -- run `/adapt <path|url>`.")

    if args.adapt and detected:
        commands.handle(session, "/adapt")
    elif detected:
        cached = session.try_load_cached()
        if cached:
            ui.ok(f"loaded cached adapter for {cached['repo_id']} "
                  f"({cached['fingerprint'][:12]}…)")

    if not session.adapter_dir:
        ui.info("tip: run `/load` or `/adapt` before prompting.")

    read = _make_reader()
    while True:
        try:
            line = read("\nc2l> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line or not line.strip():
            continue
        line = line.strip()

        if line.startswith("/"):
            if commands.handle(session, line) == "quit":
                break
            continue

        # Treat as a completion prompt.
        if not session.adapter_dir:
            ui.warn("no adapter loaded -- trying cached adapter…")
            cached = session.try_load_cached()
            if cached is None:
                ui.warn("no cache found -- run `/load <path>` or `/adapt`")
                continue
            ui.ok(f"loaded {cached['fingerprint'][:12]}…")
        try:
            with ui.status(f"generating ({session.backend})"):
                out = session.complete(line)
        except Exception as e:
            ui.error(f"inference failed: {e}")
            continue
        ui.rule(session.repo_id or "completion")
        ui.assistant_code(out)

    ui.info("bye.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
