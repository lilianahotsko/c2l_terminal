"""Code2LoRA (C2L) SDK.

A single, installable library that turns any git repository into a portable
LoRA adapter for a frozen code LLM -- without per-user training -- and runs
that adapter across full-precision, 4-bit, or GGUF/CPU backends.

The SDK is the single source of truth for every delivery surface:

* ``c2l.cli``         -- local command line (``c2l adapt`` / ``c2l run`` ...)
* ``c2l.app``         -- local Gradio app
* ``c2l.api``         -- hosted FastAPI service + generation worker

Layered modules:

* :mod:`c2l.core`      -- model definitions (GRU, multi-task head, LoRA) + ckpt loading
* :mod:`c2l.tasks`     -- pluggable task registry (extract / format / metric)
* :mod:`c2l.pipeline`  -- repo -> adapter context -> generated LoRA (no base LLM needed)
* :mod:`c2l.export`    -- generated LoRA -> standard PEFT adapter (+ GGUF)
* :mod:`c2l.infer`     -- run an adapter: fp16 / 4-bit / GGUF backends
* :mod:`c2l.registry`  -- content-addressed adapter registry (SaaS)
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "load_config",
    "C2LConfig",
    "generate_adapter",
    "GeneratedAdapter",
]


def __getattr__(name):  # lazy re-exports so importing c2l stays cheap
    if name in ("load_config", "C2LConfig"):
        from .config import C2LConfig, load_config
        return {"load_config": load_config, "C2LConfig": C2LConfig}[name]
    if name in ("generate_adapter", "GeneratedAdapter"):
        from .pipeline import GeneratedAdapter, generate_adapter
        return {"generate_adapter": generate_adapter,
                "GeneratedAdapter": GeneratedAdapter}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
