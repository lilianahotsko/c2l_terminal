"""Repository -> generated LoRA, with **no dependency on the base LLM**.

This is the cheap half of C2L: embed the repo state + commit diffs with the
0.6B encoder, stream the :class:`CommitGRU`, and run the (multi-task)
:class:`Code2LoRAHead`. The result is a :class:`GeneratedAdapter` holding one
``(A, B)`` pair per module type -- a few MB of tensors. Materializing it into a
runnable adapter (and loading the base model) happens later in
:mod:`c2l.export` / :mod:`c2l.infer`.

Because generation never loads the multi-GB base model and is a single forward
pass (not training), it runs on CPU for small/medium repos -- the basis for the
no-GPU and air-gapped delivery modes.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .config import C2LConfig, load_config
from .core import (
    Code2LoRAHead,
    CommitGRU,
    ModuleSpec,
    discover_module_types_and_dims,
    specs_from_hf_config,
)
from .embedding import Qwen3Embedder
from .git_pipeline import (GitCatFileBatch, RepoProcessResult,
                           filter_diff, get_working_tree_diff, ls_tree_py,
                           process_repo, working_tree_is_dirty)

ProgressFn = Optional[Callable[[str, float], None]]

ENV_CKPT_PATH = "C2L_CKPT"
ENV_CKPT_REPO = "C2L_CKPT_REPO"
ENV_CKPT_FILE = "C2L_CKPT_FILE"

WALK_STATE_DIR = ".walk_state"
WALK_STATE_FILE = "gru_walk.pt"

_GIT_URL_RE = re.compile(r"^(https?://|git@|ssh://)")


def _is_git_url(s: str) -> bool:
    return bool(_GIT_URL_RE.match(s.strip()))


@dataclass
class GeneratedAdapter:
    """A repository-conditioned LoRA, ready to export or inject."""

    repo_id: str
    task: str
    base_model: str
    target_modules: List[str]
    rank: int
    alpha: float
    # Per module *type* matrices on CPU (float32): A[t]=[rank,in], B[t]=[out,rank]
    A: Dict[str, "np.ndarray"]
    B: Dict[str, "np.ndarray"]
    # type -> (in_features, out_features)
    type_dims: Dict[str, Tuple[int, int]]
    # provenance
    endpoint_sha: str = ""
    n_commits_walked: int = 0
    checkpoint_id: str = ""
    task_conditioned: bool = False

    def fingerprint(self) -> str:
        """Stable content key: repo+commit+task+checkpoint+base model."""
        import hashlib
        h = hashlib.sha256()
        for part in (self.repo_id, self.endpoint_sha, self.task,
                     self.checkpoint_id, self.base_model,
                     ",".join(self.target_modules), str(self.rank)):
            h.update(part.encode("utf-8"))
            h.update(b"\x00")
        return h.hexdigest()[:16]


def resolve_repo(repo_input: str, work_dir: Path,
                 progress: ProgressFn = None) -> Tuple[Path, str]:
    """Return (local_repo_dir, repo_id). Clones a URL into ``work_dir``."""
    repo_input = repo_input.strip()
    if not repo_input:
        raise ValueError("Please provide a repository path or git URL.")
    if _is_git_url(repo_input):
        repo_id = _repo_id_from_url(repo_input)
        dest = work_dir / repo_id.replace("/", "__")
        if dest.exists():
            shutil.rmtree(dest, ignore_errors=True)
        if progress:
            progress(f"Cloning {repo_input}…", 0.0)
        res = subprocess.run(
            ["git", "clone", "--no-single-branch", repo_input, str(dest)],
            capture_output=True, text=True, timeout=900)
        if res.returncode != 0 or not (dest / ".git").exists():
            raise RuntimeError(f"git clone failed:\n{res.stderr[-2000:]}")
        return dest, repo_id
    path = Path(repo_input).expanduser()
    if not (path / ".git").exists():
        raise ValueError(f"{path} is not a git repository (no .git found).")
    return path, _repo_id_from_path(path)


class AdapterGenerator:
    """Loads the GRU + multi-task head + encoder once, generates many adapters."""

    def __init__(self, config: Optional[C2LConfig] = None):
        self.cfg = config or load_config()
        self.device = self.cfg.device or self._auto_device()
        import torch
        self.dtype = torch.float32  # generation is small; fp32 is safe on CPU/GPU
        self._gru: Optional[CommitGRU] = None
        self._head: Optional[Code2LoRAHead] = None
        self._embedder: Optional[Qwen3Embedder] = None
        self._specs: Optional[List[ModuleSpec]] = None
        self._type_dims: Optional[Dict[str, Tuple[int, int]]] = None
        self._rank = 16
        self._alpha = 32.0
        self._ckpt_id = ""
        self._loaded = False
        #: last RepoProcessResult produced by :meth:`generate` (QnAs, stats).
        self.last_result: Optional[RepoProcessResult] = None

    @staticmethod
    def _auto_device() -> str:
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    @property
    def embedder(self) -> Qwen3Embedder:
        if self._embedder is None:
            import torch
            self._embedder = Qwen3Embedder(
                device=self.device,
                dtype=(torch.float16 if self.device == "cuda" else None),
                model_name=self.cfg.encoder_model,
            )
        return self._embedder

    # ---- checkpoint loading (no base model weights) ----

    def _resolve_ckpt(self) -> str:
        local = self.cfg.ckpt_path or os.environ.get(ENV_CKPT_PATH)
        if local:
            if Path(local).exists():
                return local
            raise FileNotFoundError(f"checkpoint path {local} does not exist.")
        if self.cfg.offline:
            raise FileNotFoundError(
                "offline mode: set ckpt_path / C2L_CKPT to a local checkpoint.")
        from huggingface_hub import hf_hub_download
        repo = os.environ.get(ENV_CKPT_REPO, self.cfg.ckpt_repo)
        fname = os.environ.get(ENV_CKPT_FILE, self.cfg.ckpt_file)
        return hf_hub_download(repo_id=repo, filename=fname)

    def _type_dims_for_base(self, head_cfg: dict) -> Dict[str, Tuple[int, int]]:
        # Prefer dims stored in the checkpoint; otherwise derive from the base
        # model's HF *config* (no weights) so generation stays lightweight.
        td = head_cfg.get("type_dims")
        if td:
            return {t: tuple(v) for t, v in td.items()
                    if t in self.cfg.target_modules}
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(self.cfg.base_model)
        specs = specs_from_hf_config(config, self.cfg.target_modules)
        self._specs = specs
        return discover_module_types_and_dims(specs)

    def load(self, progress: ProgressFn = None) -> None:
        if self._loaded:
            return
        import torch

        def _p(msg, frac):
            if progress:
                progress(msg, frac)

        _p("Loading checkpoint…", 0.05)
        ckpt_path = self._resolve_ckpt()
        self._ckpt_id = Path(ckpt_path).name
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        head_cfg = state.get("head_config") or {}
        gru_cfg = state.get("gru_config") or {}
        self._rank = int(head_cfg.get("rank", 16))
        self._alpha = float((state.get("args") or {}).get("alpha", 32.0))
        head_hidden = int(head_cfg.get("hidden_dim", 1024))
        input_dim = int(head_cfg.get("input_dim", gru_cfg.get("hidden_dim", 2048)))
        num_tasks = int(head_cfg.get("num_tasks", 0))
        task_dim = int(head_cfg.get("task_dim", 64))

        _p("Resolving base-model dims (config only)…", 0.3)
        self._type_dims = self._type_dims_for_base(head_cfg)

        _p("Building GRU + head…", 0.6)
        self._head = Code2LoRAHead(
            input_dim=input_dim, type_dims=self._type_dims,
            hidden_dim=head_hidden, rank=self._rank,
            num_tasks=num_tasks, task_dim=task_dim,
        ).to(self.device)
        self._head.load_state_dict(state["head_state"])
        self._head.eval()

        self._gru = CommitGRU(
            diff_input_dim=int(gru_cfg.get("diff_input_dim", 2048)),
            repo_state_dim=int(gru_cfg.get("repo_state_dim", 2048)),
            hidden_dim=int(gru_cfg.get("hidden_dim", 2048)),
        ).to(self.device)
        self._gru.load_state_dict(state["gru_state"])
        self._gru.eval()
        _p("Generator ready.", 1.0)
        self._loaded = True

    # ---- generation ----

    def generate(self, repo_dir: Path, repo_id: str, task: str = "assert_rhs",
                 walk_fraction: Optional[float] = None,
                 incremental: bool = True,
                 progress: ProgressFn = None) -> GeneratedAdapter:
        import torch
        from .tasks import get_task, num_tasks as registry_num_tasks  # noqa

        self.load(progress=progress)
        wf = walk_fraction if walk_fraction is not None else self.cfg.walk_fraction

        result: RepoProcessResult = process_repo(repo_dir, repo_id=repo_id,
                                                  progress=progress)
        self.last_result = result
        if result.kept_commits == 0:
            raise RuntimeError(
                "No commits introduced new assertions in test files; this repo "
                "has no usable Code2LoRA commit stream.")

        endpoint = result.split_boundary_index
        n_kept = result.kept_commits
        if endpoint < 0:
            endpoint = max(0, min(n_kept - 1, int(round(wf * n_kept)) - 1))

        gru = self._gru
        device = self.device
        walk_path = _walk_state_path(self.cfg, repo_id, task)
        prior = (_load_walk_state(walk_path, repo_id=repo_id, task=task,
                                  ckpt_id=self._ckpt_id)
                 if incremental else None)

        sha_to_idx = {c.sha: i for i, c in enumerate(result.commits)}
        start_ci = 0
        h = None
        mode = "full"

        if prior is not None:
            init_sha = result.commits[0].sha
            old_idx = sha_to_idx.get(prior.get("endpoint_sha", ""))
            if (prior.get("initial_commit_sha") == init_sha
                    and old_idx is not None
                    and old_idx <= endpoint):
                h = prior["h"].to(device)
                if old_idx < endpoint:
                    start_ci = old_idx + 1
                    mode = "incremental"
                    if progress:
                        progress(
                            f"Incremental adapt: {endpoint - old_idx} new commit(s) "
                            f"(skipping {old_idx + 1} cached)…",
                            0.0,
                        )
                else:
                    start_ci = endpoint + 1  # no new commits to walk
                    mode = "cached"
                    if progress:
                        progress("Repo unchanged at endpoint — reusing GRU state…",
                                 1.0)

        if h is None:
            if progress:
                progress("Embedding initial repository state…", 0.0)
            repo_state_0, _ = self._embed_repo_state(repo_dir, result.commits[0].sha)
            h = gru.init_hidden(
                torch.from_numpy(repo_state_0).to(device).unsqueeze(0))
            if progress and prior is not None:
                progress("History changed — full GRU walk from scratch…", 0.0)

        hidden_endpoint = None
        with torch.no_grad():
            if start_ci <= endpoint:
                for ci in range(start_ci, endpoint + 1):
                    commit = result.commits[ci]
                    if progress:
                        label = ("incremental" if mode == "incremental"
                                 else "full")
                        progress(
                            f"GRU {label} commit {ci + 1}/{endpoint + 1}…",
                            (ci + 1) / max(endpoint + 1, 1),
                        )
                    diff_emb = self.embedder.embed_diff(commit.production_code_diff)
                    diff_t = torch.from_numpy(diff_emb).to(device).unsqueeze(0)
                    h = gru.step(diff_t, h)
            hidden_endpoint = h[-1].detach()

        ctx = gru.output_norm(hidden_endpoint)              # [1, H]

        task_obj = get_task(task)
        task_conditioned = bool(self._head.num_tasks > 0)
        task_idx = task_obj.task_index if task_conditioned else None
        if (task_conditioned and task_obj.task_index >= self._head.num_tasks):
            raise ValueError(
                f"checkpoint only supports {self._head.num_tasks} task(s); "
                f"task {task!r} has index {task_obj.task_index}.")

        with torch.no_grad():
            head_out = self._head(ctx, task_id=task_idx)

        A = {t: head_out["A"][t][0].float().cpu().numpy() for t in self._head.types}
        B = {t: head_out["B"][t][0].float().cpu().numpy() for t in self._head.types}

        _save_walk_state(
            walk_path,
            repo_id=repo_id,
            task=task,
            ckpt_id=self._ckpt_id,
            initial_commit_sha=result.commits[0].sha,
            endpoint_sha=result.commits[endpoint].sha,
            endpoint_index=endpoint,
            n_kept_commits=n_kept,
            h=h,
        )

        return GeneratedAdapter(
            repo_id=repo_id, task=task, base_model=self.cfg.base_model,
            target_modules=list(self._head.types), rank=self._rank,
            alpha=self._alpha, A=A, B=B, type_dims=dict(self._type_dims),
            endpoint_sha=result.commits[endpoint].sha,
            n_commits_walked=endpoint + 1, checkpoint_id=self._ckpt_id,
            task_conditioned=task_conditioned)

    def generate_working_tree(self, repo_dir: Path, repo_id: str,
                              task: str = "assert_rhs",
                              progress: ProgressFn = None) -> GeneratedAdapter:
        """Update the adapter from *uncommitted* local changes.

        Diffs the last committed adapter endpoint against the current working
        tree (tracked files), embeds that single diff, and takes **one** GRU
        step from the saved committed hidden state. The persistent walk state is
        left untouched, so the committed anchor is preserved and every
        working-tree adapt re-derives from the same committed endpoint.
        """
        import hashlib

        import torch
        from .tasks import get_task

        self.load(progress=progress)
        walk_path = _walk_state_path(self.cfg, repo_id, task)
        prior = _load_walk_state(walk_path, repo_id=repo_id, task=task,
                                 ckpt_id=self._ckpt_id)
        if prior is None:
            raise RuntimeError(
                "no committed adapter state for this repo+task yet. Run "
                "`/adapt` once (on committed history) to create the anchor, "
                "then `/adapt --local` to fold in uncommitted changes.")

        base_sha = prior.get("endpoint_sha", "")
        if progress:
            progress("Diffing working tree against last adapter endpoint…", 0.0)
        raw = get_working_tree_diff(repo_dir, base_sha)
        diff = filter_diff(raw)
        if not diff.strip():
            raise RuntimeError(
                "no uncommitted source changes vs the last adapter endpoint "
                f"({base_sha[:10]}). Edit tracked .py/.md/.rst files first, or "
                "use `/adapt` to walk newly committed history.")

        gru = self._gru
        device = self.device
        h = prior["h"].to(device)

        if progress:
            progress("GRU working-tree step (1)…", 0.5)
        with torch.no_grad():
            diff_emb = self.embedder.embed_diff(diff)
            diff_t = torch.from_numpy(diff_emb).to(device).unsqueeze(0)
            h_wt = gru.step(diff_t, h)
            hidden_endpoint = h_wt[-1].detach()

        ctx = gru.output_norm(hidden_endpoint)

        task_obj = get_task(task)
        task_conditioned = bool(self._head.num_tasks > 0)
        task_idx = task_obj.task_index if task_conditioned else None
        if (task_conditioned and task_obj.task_index >= self._head.num_tasks):
            raise ValueError(
                f"checkpoint only supports {self._head.num_tasks} task(s); "
                f"task {task!r} has index {task_obj.task_index}.")

        with torch.no_grad():
            head_out = self._head(ctx, task_id=task_idx)

        A = {t: head_out["A"][t][0].float().cpu().numpy() for t in self._head.types}
        B = {t: head_out["B"][t][0].float().cpu().numpy() for t in self._head.types}

        # Distinct provenance so the working-tree adapter caches under its own
        # fingerprint and never clobbers the committed one. NOTE: we do NOT call
        # _save_walk_state -- the committed anchor (prior["h"]) is preserved.
        diff_hash = hashlib.sha256(diff.encode("utf-8")).hexdigest()[:8]
        synthetic_endpoint = f"{base_sha}+wt-{diff_hash}"
        walked = int(prior.get("endpoint_index", 0)) + 1

        return GeneratedAdapter(
            repo_id=repo_id, task=task, base_model=self.cfg.base_model,
            target_modules=list(self._head.types), rank=self._rank,
            alpha=self._alpha, A=A, B=B, type_dims=dict(self._type_dims),
            endpoint_sha=synthetic_endpoint,
            n_commits_walked=walked, checkpoint_id=self._ckpt_id,
            task_conditioned=task_conditioned)

    def _embed_repo_state(self, repo_dir: Path, commit_sha: str) -> Tuple["np.ndarray", int]:
        files = ls_tree_py(repo_dir, commit_sha)
        cap = self.cfg.max_repo_state_files
        if len(files) > cap:
            files = sorted(files, key=lambda x: -x[2])[:cap]
        texts: List[str] = []
        with GitCatFileBatch(repo_dir) as cat:
            for blob, _path, _size in files:
                txt = cat.read_blob_text(blob)
                if txt:
                    texts.append(txt)
        vec = self.embedder.embed_repo_state(texts)
        return vec, len(texts)


def generate_adapter(repo: str, task: str = "assert_rhs",
                     config: Optional[C2LConfig] = None,
                     work_dir: Optional[str] = None,
                     incremental: bool = True,
                     progress: ProgressFn = None) -> GeneratedAdapter:
    """One-shot: resolve a repo (URL or path) and generate its adapter."""
    cfg = config or load_config()
    gen = AdapterGenerator(cfg)
    wd = Path(work_dir).expanduser() if work_dir else (
        Path(os.path.expanduser("~/.cache/c2l/repos")))
    wd.mkdir(parents=True, exist_ok=True)
    repo_dir, repo_id = resolve_repo(repo, wd, progress=progress)
    return gen.generate(repo_dir, repo_id, task=task,
                        incremental=incremental, progress=progress)


def _repo_id_from_url(url: str) -> str:
    s = url.strip()
    s = re.sub(r"\.git$", "", s)
    s = re.sub(r"^git@[^:]+:", "", s)
    s = re.sub(r"^(https?|ssh)://[^/]+/", "", s)
    parts = [p for p in s.split("/") if p]
    if len(parts) >= 2:
        return f"{parts[-2]}/{parts[-1]}"
    return parts[-1] if parts else "repo"


def _repo_id_from_path(path: Path) -> str:
    parts = path.resolve().parts
    if len(parts) >= 2:
        return f"{parts[-2]}/{parts[-1]}"
    return path.name


def _walk_state_path(cfg: C2LConfig, repo_id: str, task: str) -> Path:
    safe = repo_id.replace("/", "__")
    return cfg.resolved_adapters_dir() / WALK_STATE_DIR / safe / task / WALK_STATE_FILE


def _load_walk_state(path: Path, *, repo_id: str, task: str, ckpt_id: str
                     ) -> Optional[dict]:
    import torch
    if not path.exists():
        return None
    try:
        state = torch.load(str(path), map_location="cpu", weights_only=False)
    except Exception:
        return None
    if not isinstance(state, dict):
        return None
    if (state.get("repo_id") == repo_id and state.get("task") == task
            and state.get("checkpoint_id") == ckpt_id and "h" in state):
        return state
    return None


def _save_walk_state(path: Path, *, repo_id: str, task: str, ckpt_id: str,
                     initial_commit_sha: str, endpoint_sha: str,
                     endpoint_index: int, n_kept_commits: int, h) -> None:
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "repo_id": repo_id,
        "task": task,
        "checkpoint_id": ckpt_id,
        "initial_commit_sha": initial_commit_sha,
        "endpoint_sha": endpoint_sha,
        "endpoint_index": int(endpoint_index),
        "n_kept_commits": int(n_kept_commits),
        "h": h.detach().cpu(),
    }, str(path))


__all__ = [
    "GeneratedAdapter",
    "AdapterGenerator",
    "generate_adapter",
    "resolve_repo",
]
