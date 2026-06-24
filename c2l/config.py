"""Single source of configuration for the whole C2L platform.

Resolution order for every field:

1. explicit argument to :func:`load_config`,
2. environment variable (``C2L_*``),
3. a YAML file (``$C2L_CONFIG`` or ``./c2l.yaml`` or the packaged default),
4. the hard-coded defaults below.

Defaults match the published paper artefacts (``code2lora/data_paths.py``)
and the demo engine (``hf_space_code2lora/c2l_demo/engine.py``).
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

# Defaults (kept in sync with hf_space_code2lora/c2l_demo/engine.py)
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B"
DEFAULT_ENCODER_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "up_proj", "gate_proj", "down_proj",
]
DEFAULT_CKPT_REPO = "code2lora/code2lora-gru"
DEFAULT_CKPT_FILE = "code2lora_gru.pt"

PACKAGED_CONFIG = Path(__file__).with_name("c2l.yaml")


@dataclass
class C2LConfig:
    """Resolved platform configuration."""

    base_model: str = DEFAULT_BASE_MODEL
    encoder_model: str = DEFAULT_ENCODER_MODEL
    target_modules: List[str] = field(default_factory=lambda: list(DEFAULT_TARGET_MODULES))

    # Checkpoint (GRU + multi-task head). Either a local path or a Hub repo+file.
    ckpt_path: Optional[str] = None
    ckpt_repo: str = DEFAULT_CKPT_REPO
    ckpt_file: str = DEFAULT_CKPT_FILE

    # Tasks enabled in this deployment (registry ids). Empty == all registered.
    tasks: List[str] = field(default_factory=list)

    # Generation knobs.
    max_repo_state_files: int = 400
    walk_fraction: float = 0.8

    # Where exported adapters / registry blobs live.
    adapters_dir: str = "~/.cache/c2l/adapters"

    # Force a device for generation ("cuda" / "cpu" / None == auto).
    device: Optional[str] = None

    # Air-gapped / secure mode: never touch the network.
    offline: bool = False

    def resolved_adapters_dir(self) -> Path:
        return Path(os.path.expanduser(self.adapters_dir))

    def to_dict(self) -> dict:
        return asdict(self)


def _coerce_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def _load_yaml(path: Path) -> dict:
    if not path or not path.exists():
        return {}
    try:
        import yaml  # type: ignore
    except ImportError:
        return _mini_yaml(path)
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return data if isinstance(data, dict) else {}


def _mini_yaml(path: Path) -> dict:
    """Tiny dependency-free fallback parser for flat ``key: value`` / list YAML.

    Only supports the simple structure used by the packaged ``c2l.yaml`` so the
    SDK can read its own config even when PyYAML is not installed.
    """
    out: dict = {}
    cur_list_key: Optional[str] = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if line.lstrip().startswith("- ") and cur_list_key is not None:
            out.setdefault(cur_list_key, []).append(line.lstrip()[2:].strip())
            continue
        if ":" in line and not line.startswith(" "):
            key, _, val = line.partition(":")
            key = key.strip()
            val = val.strip()
            if val == "":
                cur_list_key = key
                out.setdefault(key, [])
            else:
                cur_list_key = None
                out[key] = val.strip().strip('"').strip("'")
    return out


def load_config(path: Optional[str] = None, **overrides) -> C2LConfig:
    """Build a :class:`C2LConfig` from YAML + env + explicit overrides."""
    yaml_path = (
        Path(path) if path
        else Path(os.environ["C2L_CONFIG"]) if os.environ.get("C2L_CONFIG")
        else Path("c2l.yaml") if Path("c2l.yaml").exists()
        else PACKAGED_CONFIG
    )
    data = _load_yaml(yaml_path)
    cfg = C2LConfig()

    def pick(field_name, env_name, cast=str):
        if field_name in overrides and overrides[field_name] is not None:
            return overrides[field_name]
        if os.environ.get(env_name):
            return cast(os.environ[env_name])
        if field_name in data and data[field_name] not in (None, ""):
            return cast(data[field_name])
        return getattr(cfg, field_name)

    def pick_list(field_name, env_name):
        if field_name in overrides and overrides[field_name] is not None:
            return list(overrides[field_name])
        if os.environ.get(env_name):
            return [x for x in os.environ[env_name].replace(",", " ").split() if x]
        if field_name in data and data[field_name]:
            v = data[field_name]
            if isinstance(v, str):
                return [x for x in v.replace(",", " ").split() if x]
            return list(v)
        return getattr(cfg, field_name)

    cfg.base_model = pick("base_model", "C2L_BASE_MODEL")
    cfg.encoder_model = pick("encoder_model", "C2L_ENCODER_MODEL")
    cfg.target_modules = pick_list("target_modules", "C2L_TARGET_MODULES")
    cfg.ckpt_path = pick("ckpt_path", "C2L_CKPT") if (
        os.environ.get("C2L_CKPT") or data.get("ckpt_path") or overrides.get("ckpt_path")
    ) else None
    cfg.ckpt_repo = pick("ckpt_repo", "C2L_CKPT_REPO")
    cfg.ckpt_file = pick("ckpt_file", "C2L_CKPT_FILE")
    cfg.tasks = pick_list("tasks", "C2L_TASKS")
    cfg.max_repo_state_files = int(pick("max_repo_state_files", "C2L_MAX_REPO_STATE_FILES", int))
    cfg.walk_fraction = float(pick("walk_fraction", "C2L_WALK_FRACTION", float))
    cfg.adapters_dir = pick("adapters_dir", "C2L_ADAPTERS_DIR")
    cfg.device = pick("device", "C2L_DEVICE") if (
        os.environ.get("C2L_DEVICE") or data.get("device") or overrides.get("device")
    ) else None
    cfg.offline = _coerce_bool(pick("offline", "C2L_OFFLINE", _coerce_bool))
    return cfg


__all__ = [
    "C2LConfig",
    "load_config",
    "DEFAULT_BASE_MODEL",
    "DEFAULT_ENCODER_MODEL",
    "DEFAULT_TARGET_MODULES",
    "DEFAULT_CKPT_REPO",
    "DEFAULT_CKPT_FILE",
]
