"""Materialize a :class:`~c2l.pipeline.GeneratedAdapter` into a *standard*
PEFT LoRA adapter (and, optionally, a GGUF LoRA for llama.cpp).

Why this matters: the C2L head emits one ``(A, B)`` pair per module *type*,
shared across layers. PEFT's LoRA delta is ``scaling * lora_B(lora_A(x))`` with
``scaling = lora_alpha / r`` -- identical to the C2L wrapper -- so we can write
a portable ``adapter_model.safetensors`` + ``adapter_config.json`` by simply
replicating each type's matrices onto every layer of that type.

A standard PEFT adapter is portable to:

* ``peft`` / ``transformers`` (full precision or 4-bit base),
* ``vLLM`` / ``TGI`` multi-LoRA serving,
* ``llama.cpp`` after GGUF conversion (CPU / no-GPU).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from .core import ModuleSpec, specs_from_hf_config
from .pipeline import GeneratedAdapter

ADAPTER_WEIGHTS_NAME = "adapter_model.safetensors"
ADAPTER_CONFIG_NAME = "adapter_config.json"
C2L_META_NAME = "c2l_adapter.json"


def _layer_specs(adapter: GeneratedAdapter) -> List[ModuleSpec]:
    """Per-layer specs for the base model (config only, no weights)."""
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(adapter.base_model)
    return specs_from_hf_config(config, adapter.target_modules)


def build_peft_state_dict(adapter: GeneratedAdapter):
    """Return a PEFT-convention state dict (torch tensors) for the adapter."""
    import torch

    specs = _layer_specs(adapter)
    state: Dict[str, "torch.Tensor"] = {}
    A = {t: torch.tensor(v, dtype=torch.float32) for t, v in adapter.A.items()}
    B = {t: torch.tensor(v, dtype=torch.float32) for t, v in adapter.B.items()}
    for sp in specs:
        if sp.type not in A:
            continue
        prefix = f"base_model.model.{sp.full_name}"
        state[f"{prefix}.lora_A.weight"] = A[sp.type].clone()
        state[f"{prefix}.lora_B.weight"] = B[sp.type].clone()
    return state


def build_peft_config(adapter: GeneratedAdapter) -> dict:
    return {
        "peft_type": "LORA",
        "auto_mapping": None,
        "base_model_name_or_path": adapter.base_model,
        "revision": None,
        "task_type": "CAUSAL_LM",
        "inference_mode": True,
        "r": int(adapter.rank),
        "lora_alpha": float(adapter.alpha),
        "lora_dropout": 0.0,
        "target_modules": list(adapter.target_modules),
        "fan_in_fan_out": False,
        "bias": "none",
        "modules_to_save": None,
        "init_lora_weights": True,
        "layers_to_transform": None,
        "layers_pattern": None,
        "rank_pattern": {},
        "alpha_pattern": {},
        "use_rslora": False,
        "use_dora": False,
    }


def export_peft_adapter(adapter: GeneratedAdapter, out_dir: str) -> Path:
    """Write a standard PEFT adapter directory and return its path."""
    from safetensors.torch import save_file

    out = Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    state = build_peft_state_dict(adapter)
    save_file(state, str(out / ADAPTER_WEIGHTS_NAME),
              metadata={"format": "pt"})

    with open(out / ADAPTER_CONFIG_NAME, "w", encoding="utf-8") as fh:
        json.dump(build_peft_config(adapter), fh, indent=2)

    meta = {
        "c2l_version": _version(),
        "fingerprint": adapter.fingerprint(),
        "repo_id": adapter.repo_id,
        "task": adapter.task,
        "task_conditioned": adapter.task_conditioned,
        "base_model": adapter.base_model,
        "endpoint_sha": adapter.endpoint_sha,
        "n_commits_walked": adapter.n_commits_walked,
        "checkpoint_id": adapter.checkpoint_id,
        "rank": adapter.rank,
        "alpha": adapter.alpha,
        "target_modules": list(adapter.target_modules),
    }
    with open(out / C2L_META_NAME, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    return out


def convert_to_gguf(adapter_dir: str, out_path: Optional[str] = None,
                    llama_cpp_dir: Optional[str] = None) -> Path:
    """Convert an exported PEFT adapter to a GGUF LoRA (for llama.cpp / CPU).

    Best-effort wrapper around llama.cpp's ``convert_lora_to_gguf.py``. Point at
    a llama.cpp checkout via ``llama_cpp_dir`` or the ``C2L_LLAMACPP`` env var.
    """
    import os
    import subprocess

    adir = Path(adapter_dir).expanduser()
    if not (adir / ADAPTER_CONFIG_NAME).exists():
        raise FileNotFoundError(f"{adir} is not a PEFT adapter directory.")
    out = Path(out_path).expanduser() if out_path else adir / "adapter.gguf"

    root = llama_cpp_dir or os.environ.get("C2L_LLAMACPP")
    script = None
    if root:
        cand = Path(root).expanduser() / "convert_lora_to_gguf.py"
        if cand.exists():
            script = cand
    if script is None:
        from shutil import which
        found = which("convert_lora_to_gguf.py")
        if found:
            script = Path(found)
    if script is None:
        raise RuntimeError(
            "convert_lora_to_gguf.py not found. Clone llama.cpp and set "
            "C2L_LLAMACPP=/path/to/llama.cpp (or pass llama_cpp_dir).")

    cmd = ["python", str(script), str(adir), "--outfile", str(out)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"GGUF conversion failed:\n{res.stderr[-2000:]}")
    return out


def _version() -> str:
    try:
        from . import __version__
        return __version__
    except Exception:
        return "0"


__all__ = [
    "export_peft_adapter",
    "build_peft_state_dict",
    "build_peft_config",
    "convert_to_gguf",
]
