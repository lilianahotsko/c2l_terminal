"""Session state for the c2l terminal: config, repo, adapter, and backend.

All heavy ``c2l`` imports are deferred so that starting the REPL (and running
``--help``) never pulls in torch/transformers until inference is actually
needed.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

VALID_BACKENDS = ("4bit", "8bit", "hf", "gguf")


def _is_git_repo(path: Path) -> bool:
    try:
        r = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--is-inside-work-tree"],
            capture_output=True, text=True)
        return r.returncode == 0 and r.stdout.strip() == "true"
    except Exception:
        return False


@dataclass
class Session:
    """Mutable terminal session. One adapter + one backend live at a time."""

    repo: Optional[str] = None            # git URL or local path
    repo_id: Optional[str] = None
    task: str = "assert_rhs"
    backend: str = "4bit"
    max_new_tokens: int = 16
    device: Optional[str] = None          # None -> auto
    offline: bool = False
    adapter_dir: Optional[str] = None
    adapter_fingerprint: Optional[str] = None
    context_files: List[str] = field(default_factory=list)

    _cfg: object = None
    _engine: object = None
    _engine_key: Optional[Tuple[str, str]] = None

    # ---- config -----------------------------------------------------------
    def config(self):
        if self._cfg is None:
            from c2l.config import load_config
            self._cfg = load_config(device=self.device,
                                    offline=self.offline or None)
        return self._cfg

    # ---- repo / adapter ---------------------------------------------------
    def detect_repo(self) -> Optional[str]:
        """Default the repo to the current working directory if it's a git repo."""
        if self.repo:
            return self.repo
        cwd = Path.cwd()
        if _is_git_repo(cwd):
            self.repo = str(cwd)
            return self.repo
        return None

    def adapt(self, repo: Optional[str] = None, *, incremental: bool = True,
              progress=None) -> Dict[str, object]:
        """Generate (or incrementally update) and cache an adapter for ``repo``.

        When ``incremental`` is True (default) and a prior walk exists for this
        repo+task, only new commit diffs after the last endpoint are embedded and
        the GRU continues from the saved hidden state.
        """
        from c2l.export import export_peft_adapter
        from c2l.pipeline import AdapterGenerator, resolve_repo

        repo = repo or self.detect_repo()
        if not repo:
            raise RuntimeError(
                "no repo set; pass a path/URL (e.g. `/adapt .` or "
                "`/adapt https://github.com/org/name`) -- the current "
                "directory is not a git repository.")

        cfg = self.config()
        wd = Path(os.path.expanduser("~/.cache/c2l/repos"))
        wd.mkdir(parents=True, exist_ok=True)
        repo_dir, repo_id = resolve_repo(repo, wd)
        gen = AdapterGenerator(cfg)
        adapter = gen.generate(repo_dir, repo_id, task=self.task,
                               incremental=incremental, progress=progress)
        out_dir = cfg.resolved_adapters_dir() / adapter.fingerprint()
        path = export_peft_adapter(adapter, str(out_dir))

        self.repo = str(repo_dir) if repo_dir else repo
        self.repo_id = adapter.repo_id
        self.adapter_dir = str(path)
        self.adapter_fingerprint = adapter.fingerprint()
        self._invalidate_engine()
        return {
            "repo_id": adapter.repo_id,
            "task": adapter.task,
            "task_conditioned": getattr(adapter, "task_conditioned", False),
            "fingerprint": adapter.fingerprint(),
            "commits_walked": getattr(adapter, "n_commits_walked", "?"),
            "endpoint": (getattr(adapter, "endpoint_sha", "") or "")[:10],
            "base_model": adapter.base_model,
            "adapter_dir": str(path),
        }

    def load_adapter(self, path: Optional[str] = None) -> Dict[str, object]:
        """Load a previously exported adapter from disk (no regeneration)."""
        cfg = self.config()
        adir = self._resolve_adapter_dir(path, cfg)
        meta_path = adir / "c2l_adapter.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"not a C2L adapter directory: {adir}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        task = meta.get("task", "assert_rhs")
        if task != self.task:
            raise ValueError(
                f"cached adapter task is {task!r} but session task is "
                f"{self.task!r}; run `/task {task}` then `/load`")

        self.adapter_dir = str(adir)
        self.adapter_fingerprint = meta.get("fingerprint") or adir.name
        self.repo_id = meta.get("repo_id")
        self._invalidate_engine()
        return {
            "repo_id": self.repo_id,
            "task": task,
            "fingerprint": self.adapter_fingerprint,
            "commits_walked": meta.get("n_commits_walked", "?"),
            "endpoint": (meta.get("endpoint_sha") or "")[:10],
            "base_model": meta.get("base_model", cfg.base_model),
            "adapter_dir": str(adir),
            "has_gguf": (adir / "adapter.gguf").exists(),
        }

    def try_load_cached(self) -> Optional[Dict[str, object]]:
        """Load the newest cached adapter for the current repo + task, if any."""
        repo = self.detect_repo()
        if not repo:
            return None
        cfg = self.config()
        root = cfg.resolved_adapters_dir()
        if not root.is_dir():
            return None
        repo_path = str(Path(repo).resolve())
        repo_name = Path(repo).name
        best: Optional[Tuple[float, Path]] = None
        for d in root.iterdir():
            if not d.is_dir():
                continue
            meta_path = d / "c2l_adapter.json"
            if not meta_path.exists():
                continue
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if meta.get("task") != self.task:
                continue
            rid = str(meta.get("repo_id", ""))
            if not (rid.endswith(repo_name) or repo_name in rid):
                continue
            mtime = meta_path.stat().st_mtime
            if best is None or mtime > best[0]:
                best = (mtime, d)
        if best is None:
            return None
        return self.load_adapter(str(best[1]))

    @staticmethod
    def _resolve_adapter_dir(path: Optional[str], cfg) -> Path:
        if path:
            p = Path(path).expanduser()
            if p.is_dir():
                return p.resolve()
            # fingerprint / short name under the adapters cache
            cand = cfg.resolved_adapters_dir() / p.name
            if cand.is_dir():
                return cand.resolve()
            raise FileNotFoundError(f"adapter directory not found: {path}")
        raise ValueError("pass an adapter path or fingerprint")

    # ---- backend ----------------------------------------------------------
    def _invalidate_engine(self) -> None:
        self._engine = None
        self._engine_key = None

    def set_backend(self, name: str) -> None:
        if name not in VALID_BACKENDS:
            raise ValueError(f"backend must be one of {VALID_BACKENDS}")
        if name != self.backend:
            self.backend = name
            self._invalidate_engine()

    def set_task(self, name: str) -> None:
        from c2l import tasks as T
        if name not in T.list_tasks():
            raise ValueError(f"unknown task {name!r}; known: {T.list_tasks()}")
        # Changing the task means the cached adapter no longer matches it.
        if name != self.task:
            self.task = name
            self.adapter_dir = None
            self.adapter_fingerprint = None
            self._invalidate_engine()

    def _engine_for(self):
        key = (self.backend, self.adapter_dir or "")
        if self._engine is not None and self._engine_key == key:
            return self._engine
        from c2l.infer import make_backend
        kw = {}
        if self.backend == "gguf":
            base_gguf = os.environ.get("C2L_BASE_GGUF")
            if not base_gguf:
                raise RuntimeError(
                    "gguf backend needs C2L_BASE_GGUF=<base model GGUF>. "
                    "Also build a LoRA GGUF (c2l export --adapter ... --gguf ...).")
            kw["base_gguf"] = base_gguf
            lora = None
            if self.adapter_dir:
                cand = Path(self.adapter_dir) / "adapter.gguf"
                if cand.exists():
                    lora = str(cand)
            kw["lora_gguf"] = os.environ.get("C2L_LORA_GGUF", lora)
        self._engine = make_backend(self.backend, adapter_dir=self.adapter_dir,
                                    config=self.config(), **kw)
        self._engine_key = key
        return self._engine

    # ---- inference --------------------------------------------------------
    def build_prefix(self, prompt: str) -> str:
        """Assemble the completion prefix from pinned context files + prompt.

        Inline ``@path`` tokens in the prompt are also expanded to file contents.
        """
        chunks: List[str] = []
        for fp in self.context_files:
            chunks.append(self._read_file_block(fp))
        # inline @path expansion
        words = prompt.split()
        kept: List[str] = []
        for w in words:
            if w.startswith("@") and len(w) > 1:
                chunks.append(self._read_file_block(w[1:]))
            else:
                kept.append(w)
        body = " ".join(kept) if kept else prompt
        chunks.append(body)
        return "\n".join(c for c in chunks if c).rstrip() + "\n"

    @staticmethod
    def _read_file_block(path: str) -> str:
        p = Path(path).expanduser()
        if not p.exists():
            return f"# (context file not found: {path})"
        try:
            txt = p.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return f"# (could not read {path}: {e})"
        # keep it bounded; the base model has a finite context window
        if len(txt) > 6000:
            txt = txt[-6000:]
        return f"# --- {p.name} ---\n{txt}"

    def complete(self, prompt: str) -> str:
        eng = self._engine_for()
        prefix = self.build_prefix(prompt)
        return eng.generate(prefix, max_new_tokens=self.max_new_tokens)

    # ---- introspection ----------------------------------------------------
    def status(self) -> Dict[str, object]:
        cfg = self.config()
        return {
            "repo": self.repo or "(none)",
            "repo_id": self.repo_id or "(none)",
            "task": self.task,
            "backend": self.backend,
            "max_new_tokens": self.max_new_tokens,
            "adapter": self.adapter_fingerprint or "(none -- run /adapt or /load)",
            "base_model": getattr(cfg, "base_model", "?"),
            "device": self.device or "auto",
            "offline": self.offline,
            "context_files": list(self.context_files),
        }
