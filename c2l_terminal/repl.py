"""Interactive REPL entry point for c2l-terminal."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__, commands, ui
from .session import Session


def _make_reader():
    """Return ``(read, styled)`` where ``read(prompt) -> str``.

    ``styled`` is True when the prompt may contain ANSI styling (interactive
    TTY). Falls back to plain ``input`` when stdin is not a TTY (piped/scripted)
    or when prompt_toolkit is unavailable.
    """
    if sys.stdin.isatty():
        try:
            from prompt_toolkit import ANSI, PromptSession
            from prompt_toolkit.history import FileHistory
            hist = Path("~/.cache/c2l/terminal_history").expanduser()
            hist.parent.mkdir(parents=True, exist_ok=True)
            ps = PromptSession(history=FileHistory(str(hist)))
            return (lambda prompt: ps.prompt(ANSI(prompt))), True
        except Exception:
            pass
        return (lambda prompt: input(prompt)), True
    return (lambda prompt: input(prompt)), False


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="c2lt",
        description="Terminal coding assistant powered by quantized Code2LoRA.")
    p.add_argument("repo", nargs="?", default=None,
                   help="repo path or git URL (default: current directory)")
    p.add_argument("--task", default="assert_rhs")
    p.add_argument("--backend", default="gguf",
                   choices=["4bit", "8bit", "hf", "gguf"])
    p.add_argument("--no-setup", action="store_true",
                   help="skip first-run auto-provisioning (models/llama.cpp/GGUF)")
    p.add_argument("--tokens", type=int, default=16, dest="tokens")
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

    # First-run provisioning: download models + base GGUF + llama.cpp so the
    # tool "just works" after `pip install`. One-time; cached under ~/.cache/c2l.
    if not args.no_setup and not args.offline:
        try:
            from c2l.bootstrap import auto_setup_if_needed, is_setup_done
            if not is_setup_done():
                ui.info("first run: setting up models, base GGUF, and llama.cpp "
                        "(one-time, into ~/.cache/c2l)…")
                cb, finish = ui.make_progress()
                try:
                    auto_setup_if_needed(offline=False, progress=cb)
                finally:
                    finish()
                ui.ok("setup complete.")
        except Exception as e:  # noqa: BLE001
            ui.warn(f"auto-setup skipped ({e}); run `c2l-setup` manually if needed.")

    detected = session.detect_repo()
    # Pick up a cached adapter before drawing the banner so it shows real state
    # (skip when --adapt: we regenerate right after the banner instead).
    if detected and not args.adapt:
        try:
            session.try_load_cached()
        except Exception:
            pass

    display_repo = session.repo_id or (Path(detected).name if detected else None)
    ui.welcome(
        version=__version__, repo_id=display_repo, backend=session.backend,
        adapter=session.adapter_fingerprint, device=session.device or "auto")

    if not detected:
        ui.warn("no git repo in the current directory — "
                "run `/adapt <path|url>` to get started.")
    elif args.adapt:
        commands.handle(session, "/adapt")

    read, styled = _make_reader()
    while True:
        try:
            prompt = (ui.prompt_text(session.repo_id or display_repo,
                                     ready=bool(session.adapter_dir))
                      if styled else "\nc2l> ")
            line = read(prompt)
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
        ui.rule(f"{session.repo_id or 'completion'}  ·  {session.backend}")
        ui.assistant_code(out)

    ui.info("bye — happy hacking.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
