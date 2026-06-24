"""Git → commit-sequence pipeline.

Faithful port of the repository-processing logic from
``create_dataset/build_commit_assertion_db.py`` and
``create_dataset/build_commit_parquet_db.py``:

* stream first-parent commits oldest-first,
* keep only commits that introduce *new or changed* assertions in test files
  (this is the "GRU pass" subset),
* build the ``production_code_diff`` (non-test source) between consecutive
  kept commits,
* split the kept commits chronologically 80/10/10 into train/val/test,
* emit one QnA per new assertion.

Everything runs against a local clone with full history.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Set, Tuple

from .assertions import (
    ExtractedAssertion,
    extract_assertions_from_source,
    normalize_for_id,
)

# ---------------------------------------------------------------------------
# Constants (must match the dataset builders)
# ---------------------------------------------------------------------------

SOURCE_EXTS = {".py", ".md", ".rst"}
SKIP_DIRS = {
    ".git", "__pycache__", ".venv", "venv", "env",
    "node_modules", "dist", "build", ".tox", ".mypy_cache",
    "TEST_HYPERNET",
}
EMPTY_TREE_SHA = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"
_NULL_SHA = "0000000000000000000000000000000000000000"


# ---------------------------------------------------------------------------
# Path filtering
# ---------------------------------------------------------------------------

def _is_test_path(path: str) -> bool:
    for part in Path(path).parts:
        if "test" in part.lower():
            return True
    return False


def _is_python_test_file(path: str) -> bool:
    p = Path(path)
    if p.suffix.lower() != ".py":
        return False
    if any(part in SKIP_DIRS for part in p.parts):
        return False
    return _is_test_path(path)


def _should_skip_diff_path(path: str) -> bool:
    p = Path(path)
    if p.suffix.lower() not in SOURCE_EXTS:
        return True
    if any(part in SKIP_DIRS for part in p.parts):
        return True
    if _is_test_path(path):
        return True
    return False


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def _run_git(repo_dir: Path, args: List[str], timeout: int = 300) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_dir)] + args,
            capture_output=True, timeout=timeout,
        )
        if result.returncode != 0:
            return None
        return result.stdout.decode("utf-8", errors="replace")
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None


@dataclass
class RawFileChange:
    status: str
    new_blob: str
    path: str
    old_path: Optional[str] = None


@dataclass
class RawCommit:
    sha: str
    parent_sha: Optional[str]
    timestamp: str
    changes: List[RawFileChange]


def stream_commit_changes(repo_dir: Path) -> Iterator[RawCommit]:
    """Yield one ``RawCommit`` per first-parent commit, oldest-first."""
    proc = subprocess.Popen(
        [
            "git", "-C", str(repo_dir),
            "log", "--first-parent", "--reverse",
            "--raw", "--no-abbrev", "-M",
            "--format=COMMITHDR %H %P %aI",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1 << 20,
    )
    assert proc.stdout is not None
    cur: Optional[RawCommit] = None
    try:
        for raw in proc.stdout:
            line = raw.decode("utf-8", errors="replace").rstrip("\n")
            if not line:
                continue
            if line.startswith("COMMITHDR "):
                if cur is not None:
                    yield cur
                tokens = line.split()
                if len(tokens) < 3:
                    cur = None
                    continue
                sha = tokens[1]
                timestamp = tokens[-1]
                parents = tokens[2:-1]
                parent_sha = parents[0] if parents else None
                cur = RawCommit(sha=sha, parent_sha=parent_sha,
                                timestamp=timestamp, changes=[])
                continue
            if cur is None or not line.startswith(":"):
                continue
            try:
                meta, _, tail = line.partition("\t")
                if not tail:
                    continue
                fields = meta.split()
                if len(fields) < 5:
                    continue
                new_blob = fields[3].lstrip(":")
                status = fields[4]
                parts = tail.split("\t")
                if status.startswith("R") or status.startswith("C"):
                    old_path = parts[0]
                    new_path = parts[1] if len(parts) > 1 else parts[0]
                    cur.changes.append(RawFileChange(
                        status=status, new_blob=new_blob,
                        path=new_path, old_path=old_path))
                else:
                    cur.changes.append(RawFileChange(
                        status=status, new_blob=new_blob, path=parts[0]))
            except Exception:
                continue
        if cur is not None:
            yield cur
    finally:
        try:
            proc.stdout.close()
        finally:
            proc.wait(timeout=5)


class GitCatFileBatch:
    """Persistent ``git cat-file --batch`` for cheap blob reads."""

    def __init__(self, repo_dir: Path):
        self._proc = subprocess.Popen(
            ["git", "-C", str(repo_dir), "cat-file", "--batch"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, bufsize=0,
        )

    def read_blob(self, blob_sha: str) -> Optional[bytes]:
        if not blob_sha or blob_sha == _NULL_SHA:
            return None
        p = self._proc
        assert p.stdin is not None and p.stdout is not None
        try:
            p.stdin.write((blob_sha + "\n").encode("ascii"))
            p.stdin.flush()
        except (BrokenPipeError, OSError):
            return None
        header = p.stdout.readline()
        if not header:
            return None
        parts = header.decode("utf-8", errors="replace").strip().split()
        if len(parts) < 3 or parts[1] != "blob":
            return None
        try:
            size = int(parts[2])
        except ValueError:
            return None
        body = b""
        remaining = size
        while remaining > 0:
            chunk = p.stdout.read(remaining)
            if not chunk:
                break
            body += chunk
            remaining -= len(chunk)
        try:
            p.stdout.read(1)
        except Exception:
            pass
        return body

    def read_blob_text(self, blob_sha: str) -> Optional[str]:
        data = self.read_blob(blob_sha)
        return None if data is None else data.decode("utf-8", errors="replace")

    def close(self) -> None:
        try:
            if self._proc.stdin is not None:
                self._proc.stdin.close()
        except Exception:
            pass
        try:
            self._proc.wait(timeout=5)
        except Exception:
            try:
                self._proc.kill()
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def get_diff(repo_dir: Path, parent_sha: Optional[str], child_sha: str,
             timeout: int = 600) -> str:
    base = parent_sha if parent_sha else EMPTY_TREE_SHA
    out = _run_git(repo_dir, ["diff", "--no-color", "-U3", base, child_sha],
                   timeout=timeout)
    return out if out else ""


_DIFF_HEADER_RE = re.compile(r"^diff --git a/(.+?) b/(.+?)(?:\n|$)", re.MULTILINE)


def filter_diff(raw_diff: str) -> str:
    if not raw_diff:
        return ""
    chunks = re.split(r"(?=^diff --git )", raw_diff, flags=re.MULTILINE)
    kept: List[str] = []
    for chunk in chunks:
        if not chunk.startswith("diff --git "):
            continue
        m = _DIFF_HEADER_RE.match(chunk)
        if not m:
            continue
        if _should_skip_diff_path(m.group(2)):
            continue
        kept.append(chunk)
    return "".join(kept)


def ls_tree_py(repo_dir: Path, commit_sha: str,
               max_file_bytes: int = 2_000_000) -> List[Tuple[str, str, int]]:
    """Return [(blob_sha, path, size)] for tracked .py files at a commit."""
    out = _run_git(repo_dir, ["ls-tree", "-r", "-l", commit_sha], timeout=120)
    if not out:
        return []
    rows: List[Tuple[str, str, int]] = []
    for ln in out.splitlines():
        try:
            meta, fpath = ln.split("\t", 1)
            _mode, ftype, blob, size_str = meta.split()
        except ValueError:
            continue
        if ftype != "blob" or not fpath.endswith(".py"):
            continue
        try:
            size = int(size_str)
        except ValueError:
            continue
        if size <= 0 or size > max_file_bytes:
            continue
        if any(part in SKIP_DIRS for part in Path(fpath).parts):
            continue
        rows.append((blob, fpath, size))
    return rows


# ---------------------------------------------------------------------------
# Chronological split
# ---------------------------------------------------------------------------

def chronological_splits(n_kept: int) -> List[str]:
    """80/10/10 chronological split of ``n_kept`` kept commits."""
    if n_kept <= 0:
        return []
    if n_kept < 3:
        return ["train"] * n_kept
    n_train = max(1, int(n_kept * 0.8))
    n_val = max(1, int(n_kept * 0.1))
    n_test = n_kept - n_train - n_val
    if n_test < 1:
        n_val = max(1, n_val - 1) if n_val > 1 else n_val
        n_test = n_kept - n_train - n_val
        if n_test < 1:
            n_train = max(1, n_train - 1)
            n_test = n_kept - n_train - n_val
    return ["train"] * n_train + ["val"] * n_val + ["test"] * n_test


# ---------------------------------------------------------------------------
# Per-repo processing
# ---------------------------------------------------------------------------

@dataclass
class QnaItem:
    test_file: str
    test_function: str
    assertion_type: str
    prefix: str
    target: str
    lineno: int
    col_offset: int


@dataclass
class KeptCommit:
    commit_index: int          # index among kept commits (0-based)
    original_index: int        # index in the full first-parent history
    sha: str
    timestamp: str
    in_repo_split: str         # train / val / test
    production_code_diff: str
    n_new_assertions: int
    qnas: List[QnaItem] = field(default_factory=list)


@dataclass
class RepoProcessResult:
    repo_id: str
    total_commits: int            # all first-parent commits
    test_touching_commits: int    # commits that changed a test file
    kept_commits: int             # commits selected for the GRU pass
    n_assertions: int             # total QnA pairs emitted
    n_train_kept: int
    n_val_kept: int
    n_test_kept: int
    commits: List[KeptCommit]

    @property
    def split_boundary_index(self) -> int:
        """Index (within kept commits) of the last train commit (the 80% point)."""
        last_train = -1
        for c in self.commits:
            if c.in_repo_split == "train":
                last_train = c.commit_index
        return last_train


ProgressFn = Optional[Callable[[str, float], None]]


def process_repo(repo_dir: Path, repo_id: str = "",
                 diff_timeout: int = 600,
                 progress: ProgressFn = None) -> RepoProcessResult:
    """Process a local clone into the kept-commit sequence + QnAs + stats."""
    prev_assert_keys: Dict[str, Set[Tuple[str, str, str]]] = {}
    kept: List[Dict] = []
    total_commits = 0
    test_touching = 0

    def _report(msg: str, frac: float) -> None:
        if progress is not None:
            progress(msg, frac)

    # ---- Phase 1: detect kept commits ----
    with GitCatFileBatch(repo_dir) as cat:
        for ci, raw in enumerate(stream_commit_changes(repo_dir)):
            total_commits += 1
            if ci % 50 == 0:
                _report(f"Scanning commit history… ({total_commits} commits)", 0.0)

            candidate_changes: List[RawFileChange] = []
            for ch in raw.changes:
                if ch.old_path and ch.path != ch.old_path:
                    if ch.old_path in prev_assert_keys:
                        prev_assert_keys[ch.path] = prev_assert_keys.pop(ch.old_path)
                if ch.status.startswith("D"):
                    prev_assert_keys.pop(ch.path, None)
                    continue
                if _is_python_test_file(ch.path):
                    candidate_changes.append(ch)

            if not candidate_changes:
                continue
            test_touching += 1

            per_file_new: List[Tuple[str, List[ExtractedAssertion]]] = []
            for ch in candidate_changes:
                content = cat.read_blob_text(ch.new_blob)
                if content is None:
                    prev_assert_keys.pop(ch.path, None)
                    continue
                extracted = extract_assertions_from_source(content)
                current_keys: Set[Tuple[str, str, str]] = set()
                for ex in extracted:
                    current_keys.add((
                        ex.assertion_type,
                        normalize_for_id(ex.prefix),
                        normalize_for_id(ex.target),
                    ))
                prev_keys = prev_assert_keys.get(ch.path, set())
                new_keys = current_keys - prev_keys
                if new_keys:
                    new_extracted = [
                        ex for ex in extracted
                        if (ex.assertion_type, normalize_for_id(ex.prefix),
                            normalize_for_id(ex.target)) in new_keys
                    ]
                    if new_extracted:
                        per_file_new.append((ch.path, new_extracted))
                prev_assert_keys[ch.path] = current_keys

            if not per_file_new:
                continue
            kept.append({
                "original_idx": ci,
                "sha": raw.sha,
                "timestamp": raw.timestamp,
                "per_file_new": per_file_new,
            })

    if not kept:
        return RepoProcessResult(
            repo_id=repo_id, total_commits=total_commits,
            test_touching_commits=test_touching, kept_commits=0,
            n_assertions=0, n_train_kept=0, n_val_kept=0, n_test_kept=0,
            commits=[])

    # ---- Phase 2: build diffs + chronological split ----
    splits = chronological_splits(len(kept))
    commits: List[KeptCommit] = []
    prev_kept_sha: Optional[str] = None
    n_assertions = 0
    for ki, kc in enumerate(kept):
        _report(f"Building production-code diffs… ({ki + 1}/{len(kept)} kept commits)",
                (ki + 1) / max(len(kept), 1))
        sha = kc["sha"]
        raw_diff = get_diff(repo_dir, prev_kept_sha, sha, timeout=diff_timeout)
        filtered = filter_diff(raw_diff) if raw_diff else ""

        qnas: List[QnaItem] = []
        for tfile, extracted in kc["per_file_new"]:
            for ex in extracted:
                qnas.append(QnaItem(
                    test_file=tfile, test_function=ex.test_function,
                    assertion_type=ex.assertion_type, prefix=ex.prefix,
                    target=ex.target, lineno=ex.lineno, col_offset=ex.col_offset))
        n_assertions += len(qnas)

        commits.append(KeptCommit(
            commit_index=ki, original_index=kc["original_idx"], sha=sha,
            timestamp=kc.get("timestamp", ""), in_repo_split=splits[ki],
            production_code_diff=filtered, n_new_assertions=len(qnas), qnas=qnas))
        prev_kept_sha = sha

    n_train = sum(1 for s in splits if s == "train")
    n_val = sum(1 for s in splits if s == "val")
    n_test = sum(1 for s in splits if s == "test")
    return RepoProcessResult(
        repo_id=repo_id, total_commits=total_commits,
        test_touching_commits=test_touching, kept_commits=len(kept),
        n_assertions=n_assertions, n_train_kept=n_train, n_val_kept=n_val,
        n_test_kept=n_test, commits=commits)


__all__ = [
    "process_repo",
    "RepoProcessResult",
    "KeptCommit",
    "QnaItem",
    "ls_tree_py",
    "GitCatFileBatch",
    "get_diff",
    "filter_diff",
    "chronological_splits",
]
