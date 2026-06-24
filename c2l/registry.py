"""Content-addressed adapter registry.

An adapter is fully determined by ``(repo_id, endpoint_sha, task,
checkpoint_id, base_model, rank, target_modules)`` -- captured by
:meth:`GeneratedAdapter.fingerprint`. The registry stores each exported PEFT
adapter under that fingerprint, so the same repo + commit + task never has to
be regenerated. This is what makes the hosted SaaS path cheap and cacheable.

The default backend is a local directory + a JSON-lines index, which is enough
for single-node deployments and the CLI cache. The same interface can be backed
by object storage + a database for multi-node SaaS.
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .config import C2LConfig, load_config
from .export import export_peft_adapter
from .pipeline import GeneratedAdapter

INDEX_NAME = "index.jsonl"


@dataclass
class RegistryEntry:
    fingerprint: str
    repo_id: str
    task: str
    base_model: str
    endpoint_sha: str
    checkpoint_id: str
    rank: int
    path: str
    created_at: float


class AdapterRegistry:
    def __init__(self, root: Optional[str] = None, config: Optional[C2LConfig] = None):
        cfg = config or load_config()
        self.root = Path(root).expanduser() if root else cfg.resolved_adapters_dir()
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._index_path = self.root / INDEX_NAME

    # ---- lookup ----

    def path_for(self, fingerprint: str) -> Path:
        return self.root / fingerprint

    def lookup(self, fingerprint: str) -> Optional[Path]:
        p = self.path_for(fingerprint)
        if (p / "adapter_config.json").exists():
            return p
        return None

    def lookup_adapter(self, adapter: GeneratedAdapter) -> Optional[Path]:
        return self.lookup(adapter.fingerprint())

    # ---- write ----

    def put(self, adapter: GeneratedAdapter, overwrite: bool = False) -> Path:
        import time

        fp = adapter.fingerprint()
        dest = self.path_for(fp)
        if dest.exists() and not overwrite and (dest / "adapter_config.json").exists():
            return dest
        export_peft_adapter(adapter, str(dest))
        entry = RegistryEntry(
            fingerprint=fp, repo_id=adapter.repo_id, task=adapter.task,
            base_model=adapter.base_model, endpoint_sha=adapter.endpoint_sha,
            checkpoint_id=adapter.checkpoint_id, rank=adapter.rank,
            path=str(dest), created_at=time.time())
        self._append_index(entry)
        return dest

    def _append_index(self, entry: RegistryEntry) -> None:
        with self._lock:
            with open(self._index_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(asdict(entry)) + "\n")

    # ---- list ----

    def entries(self) -> List[RegistryEntry]:
        if not self._index_path.exists():
            return []
        out: List[RegistryEntry] = []
        seen: Dict[str, RegistryEntry] = {}
        with open(self._index_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    seen[json.loads(line)["fingerprint"]] = RegistryEntry(**json.loads(line))
                except Exception:
                    continue
        out = list(seen.values())
        out.sort(key=lambda e: e.created_at, reverse=True)
        return out


__all__ = ["AdapterRegistry", "RegistryEntry"]
