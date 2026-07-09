"""Slash-command handling for the c2l terminal REPL."""

from __future__ import annotations

from typing import List

from . import ui
from .session import VALID_BACKENDS, Session

# Short aliases -> canonical command, so users can type less.
_ALIASES = {
    "/a": "/adapt", "/l": "/load", "/b": "/backend", "/t": "/task",
    "/s": "/status", "/c": "/clear", "/h": "/help", "/?": "/help",
    "/q": "/quit", "/exit": "/quit",
}


def handle(session: Session, line: str) -> str:
    """Process one slash command. Returns 'quit' to exit, else 'continue'."""
    parts = line.strip().split()
    cmd, args = parts[0].lower(), parts[1:]
    cmd = _ALIASES.get(cmd, cmd)

    if cmd == "/quit":
        return "quit"
    if cmd == "/help":
        ui.help_panel()
    elif cmd == "/clear":
        print("\033[2J\033[H", end="")
    elif cmd == "/status":
        _status(session)
    elif cmd == "/tasks":
        _tasks()
    elif cmd == "/adapt":
        _adapt(session, args)
    elif cmd == "/load":
        _load(session, args)
    elif cmd == "/task":
        _task(session, args)
    elif cmd == "/backend":
        _backend(session, args)
    elif cmd == "/tokens":
        _tokens(session, args)
    elif cmd == "/context":
        _context(session, args)
    elif cmd == "/export-gguf":
        _export_gguf(session, args)
    elif cmd == "/setup":
        _setup(session, args)
    else:
        ui.warn(f"unknown command {cmd!r} — try /help")
    return "continue"


def _status(session: Session) -> None:
    ui.keyvals("session", session.status())


def _tasks() -> None:
    try:
        from c2l import tasks as T
        ui.rule("tasks")
        for tid in T.list_tasks():
            t = T.get_task(tid)
            ui.info(f"  {tid:<14} idx={t.task_index}  -- {t.description}")
    except Exception as e:
        ui.error(f"could not list tasks: {e}")


def _load(session: Session, args: List[str]) -> None:
    try:
        if args:
            meta = session.load_adapter(args[0])
        else:
            meta = session.try_load_cached()
            if meta is None:
                ui.error("no cached adapter for this repo/task; run /adapt first")
                return
    except Exception as e:
        ui.error(f"load failed: {e}")
        return
    ui.ok(f"loaded adapter for {meta['repo_id']}")
    ui.info(f"  fingerprint : {meta['fingerprint']}")
    ui.info(f"  adapter dir : {meta['adapter_dir']}")
    ui.info(f"  gguf        : {'yes' if meta.get('has_gguf') else 'no (run /export-gguf)'}")


def _adapt(session: Session, args: List[str]) -> None:
    incremental = True
    working_tree = False
    # flags may appear in any order before the optional repo arg
    while args and args[0] in ("--full", "--local", "--dirty", "--wt"):
        flag = args[0]
        if flag == "--full":
            incremental = False
        else:  # --local / --dirty / --wt
            working_tree = True
        args = args[1:]
    repo = args[0] if args else None
    cb, finish = ui.make_progress()
    try:
        meta = session.adapt(repo, incremental=incremental,
                             working_tree=working_tree, progress=cb)
    except Exception as e:
        finish()
        ui.error(f"adapt failed: {e}")
        return
    finish()
    mode = ("working-tree (uncommitted, 1 GRU step)" if working_tree
            else "full" if not incremental else "incremental")
    ui.ok(f"adapter ready for {meta['repo_id']}")
    ui.info(f"  mode        : {mode}")
    ui.info(f"  task        : {meta['task']} (conditioned={meta['task_conditioned']})")
    ui.info(f"  fingerprint : {meta['fingerprint']}")
    ui.info(f"  commits     : {meta['commits_walked']} walked  endpoint={meta['endpoint']}")
    ui.info(f"  base model  : {meta['base_model']}")
    ui.info(f"  adapter dir : {meta['adapter_dir']}")


def _task(session: Session, args: List[str]) -> None:
    if not args:
        ui.info(f"task = {session.task}")
        return
    try:
        session.set_task(args[0])
        ui.ok(f"task set to {session.task} (adapter cleared -- run /adapt)")
    except Exception as e:
        ui.error(str(e))


def _backend(session: Session, args: List[str]) -> None:
    if not args:
        ui.info(f"backend = {session.backend}  (choices: {', '.join(VALID_BACKENDS)})")
        return
    try:
        session.set_backend(args[0])
        ui.ok(f"backend set to {session.backend}")
    except Exception as e:
        ui.error(str(e))


def _tokens(session: Session, args: List[str]) -> None:
    if not args:
        ui.info(f"max_new_tokens = {session.max_new_tokens}")
        return
    try:
        session.max_new_tokens = max(1, int(args[0]))
        ui.ok(f"max_new_tokens = {session.max_new_tokens}")
    except ValueError:
        ui.error("usage: /tokens <int>")


def _export_gguf(session: Session, args: List[str]) -> None:
    if not session.adapter_dir:
        ui.error("no adapter loaded -- run /load or /adapt first")
        return
    out = args[0] if args else None
    try:
        from c2l.export import convert_to_gguf
        path = convert_to_gguf(session.adapter_dir, out_path=out)
        ui.ok(f"GGUF LoRA written to: {path}")
        ui.info("  set /backend gguf  (and C2L_BASE_GGUF + C2L_LLAMACPP) to use it")
    except Exception as e:
        ui.error(f"GGUF export failed: {e}")


def _setup(session: Session, args: List[str]) -> None:
    """(Re-)provision models, base GGUF, and llama.cpp into ~/.cache/c2l."""
    force = "--force" in args
    try:
        from c2l.bootstrap import bootstrap_all
    except Exception as e:  # noqa: BLE001
        ui.error(f"setup unavailable: {e}")
        return
    cb, finish = ui.make_progress()
    try:
        summary = bootstrap_all(offline=session.offline, force=force, progress=cb)
    except Exception as e:  # noqa: BLE001
        finish()
        ui.error(f"setup failed: {e}")
        return
    finish()
    ui.ok("setup complete")
    ui.keyvals("provisioned", summary)


def _context(session: Session, args: List[str]) -> None:
    if not args or args[0] == "list":
        files = session.context_files or ["(none)"]
        ui.rule("context files")
        for f in files:
            ui.info(f"  {f}")
        return
    sub = args[0]
    if sub == "clear":
        session.context_files.clear()
        ui.ok("context cleared")
    elif sub == "add" and len(args) > 1:
        for f in args[1:]:
            session.context_files.append(f)
        ui.ok(f"added {len(args) - 1} context file(s)")
    else:
        ui.warn("usage: /context add <file> | /context clear | /context list")
