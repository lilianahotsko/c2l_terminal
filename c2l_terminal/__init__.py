"""c2l-terminal: a terminal coding assistant powered by quantized Code2LoRA.

A small, Claude-Code-style REPL that:

* generates a repository-conditioned LoRA adapter for *your* project (the cheap
  encoder+GRU+head forward pass -- no base LLM, CPU-friendly), then
* runs the frozen base model **quantized** (4-bit / 8-bit via bitsandbytes, or
  GGUF via llama.cpp) with that adapter injected, to complete code locally.

The heavy lifting lives in the :mod:`c2l` SDK; this package is just the
interactive front-end.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
