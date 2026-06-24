"""Run a generated/exported C2L adapter across backends.

Backends (selected by ``backend=``):

* ``hf``    -- transformers + PEFT, full precision (GPU or CPU).
* ``4bit``  -- transformers + PEFT on a bitsandbytes 4-bit base (low VRAM).
* ``8bit``  -- transformers + PEFT on a bitsandbytes 8-bit base.
* ``gguf``  -- llama.cpp with a quantized base GGUF + the LoRA GGUF (CPU / no-GPU).

The ``hf`` / ``4bit`` / ``8bit`` paths consume the *standard PEFT adapter*
produced by :func:`c2l.export.export_peft_adapter`, which validates that the
exported artifact is interchangeable with the original C2L injection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Union

from .config import C2LConfig, load_config

Backend = str  # "hf" | "4bit" | "8bit" | "gguf"


def _select_device(cfg: C2LConfig) -> str:
    if cfg.device:
        return cfg.device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


class HFInference:
    """transformers + PEFT inference, optionally on a bitsandbytes-quantized base."""

    def __init__(self, adapter_dir: Optional[str] = None,
                 quantize: Optional[str] = None,
                 config: Optional[C2LConfig] = None):
        self.cfg = config or load_config()
        self.device = _select_device(self.cfg)
        self.quantize = quantize  # None | "4bit" | "8bit"
        self.adapter_dir = adapter_dir
        self._tok = None
        self._model = None
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(self.cfg.base_model)
        if self._tok.pad_token is None:
            self._tok.pad_token = self._tok.eos_token

        quant_cfg = None
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        if self.quantize in ("4bit", "8bit"):
            from transformers import BitsAndBytesConfig
            if self.quantize == "4bit":
                quant_cfg = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16)
            else:
                quant_cfg = BitsAndBytesConfig(load_in_8bit=True)

        model = self._load_base(dtype, quant_cfg)
        if self.adapter_dir:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, self.adapter_dir)
        model.eval()
        self._model = model
        self._loaded = True

    def _load_base(self, dtype, quant_cfg):
        from transformers import AutoModelForCausalLM
        last = None
        for attn in ("flash_attention_2", "sdpa", "eager"):
            try:
                kw = dict(torch_dtype=dtype, attn_implementation=attn)
                if quant_cfg is not None:
                    kw["quantization_config"] = quant_cfg
                    kw["device_map"] = "auto"
                elif self.device == "cuda":
                    kw["device_map"] = {"": self.device}
                m = AutoModelForCausalLM.from_pretrained(self.cfg.base_model, **kw)
                if quant_cfg is None:
                    m = m.to(self.device)
                return m
            except Exception as e:
                last = e
        raise RuntimeError(f"Could not load base model: {last}")

    def generate(self, prefix: str, max_new_tokens: int = 32,
                 max_input_tokens: int = 4096, use_adapter: bool = True) -> str:
        import contextlib

        import torch
        self.load()
        tok = self._tok
        bos = getattr(tok, "bos_token_id", None) or getattr(tok, "eos_token_id", None)
        ids = tok.encode(prefix, add_special_tokens=False)
        if bos is not None:
            ids = [bos] + ids
        if len(ids) > max_input_tokens:
            ids = ids[-max_input_tokens:]
        dev = next(self._model.parameters()).device
        inp = torch.tensor([ids], dtype=torch.long, device=dev)
        attn = torch.ones_like(inp)

        # Compare against the plain base model by disabling the LoRA adapter.
        ctx = contextlib.nullcontext()
        if (not use_adapter) and self.adapter_dir and hasattr(self._model, "disable_adapter"):
            ctx = self._model.disable_adapter()
        with torch.no_grad(), ctx:
            out = self._model.generate(
                input_ids=inp, attention_mask=attn,
                max_new_tokens=int(max_new_tokens), do_sample=False,
                pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
                use_cache=True)
        gen = out[0, inp.shape[1]:].tolist()
        return tok.decode(gen, skip_special_tokens=True)


class GGUFInference:
    """llama.cpp backend: quantized base GGUF + LoRA GGUF, CPU-friendly.

    Requires a llama.cpp build (``llama-cli`` on PATH or under ``C2L_LLAMACPP``),
    a base-model GGUF (``base_gguf``), and a converted LoRA GGUF (``lora_gguf``,
    see :func:`c2l.export.convert_to_gguf`).
    """

    def __init__(self, base_gguf: str, lora_gguf: Optional[str] = None,
                 n_threads: Optional[int] = None):
        self.base_gguf = base_gguf
        self.lora_gguf = lora_gguf
        self.n_threads = n_threads

    def _llama_cli(self) -> str:
        import os
        from shutil import which
        root = os.environ.get("C2L_LLAMACPP")
        if root:
            cand = Path(root).expanduser() / "llama-cli"
            if cand.exists():
                return str(cand)
        found = which("llama-cli")
        if found:
            return found
        raise RuntimeError(
            "llama-cli not found. Build llama.cpp and set C2L_LLAMACPP, or put "
            "llama-cli on PATH.")

    def generate(self, prefix: str, max_new_tokens: int = 32) -> str:
        import subprocess
        cli = self._llama_cli()
        cmd = [cli, "-m", self.base_gguf, "-p", prefix,
               "-n", str(int(max_new_tokens)), "--no-display-prompt",
               "--temp", "0"]
        if self.lora_gguf:
            cmd += ["--lora", self.lora_gguf]
        if self.n_threads:
            cmd += ["-t", str(int(self.n_threads))]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"llama-cli failed:\n{res.stderr[-2000:]}")
        return res.stdout


def make_backend(backend: Backend = "hf", *, adapter_dir: Optional[str] = None,
                 base_gguf: Optional[str] = None, lora_gguf: Optional[str] = None,
                 config: Optional[C2LConfig] = None):
    """Factory returning a backend object exposing ``.generate(prefix, ...)``."""
    if backend in ("hf", "4bit", "8bit"):
        quant = None if backend == "hf" else backend
        return HFInference(adapter_dir=adapter_dir, quantize=quant, config=config)
    if backend == "gguf":
        if not base_gguf:
            raise ValueError("gguf backend requires base_gguf=<path to base GGUF>.")
        return GGUFInference(base_gguf=base_gguf, lora_gguf=lora_gguf)
    raise ValueError(f"Unknown backend {backend!r}")


def predict(prefix: str, adapter_dir: Optional[str] = None,
            backend: Backend = "hf", max_new_tokens: int = 32,
            config: Optional[C2LConfig] = None, **kw) -> str:
    be = make_backend(backend, adapter_dir=adapter_dir, config=config, **kw)
    return be.generate(prefix, max_new_tokens=max_new_tokens)


# ---------------------------------------------------------------------------
# Numerical fidelity check (export correctness + quantization impact)
# ---------------------------------------------------------------------------

def verify_export_fidelity(adapter, sample_prefix: str = "assert add(2, 2) == ",
                           quantize: Optional[str] = None,
                           config: Optional[C2LConfig] = None) -> Dict:
    """Compare the C2L custom-injection delta against the exported PEFT adapter.

    Loads the base model once, injects the generated LoRA via the original C2L
    wrapper, captures the next-token logits, then loads the *exported* PEFT
    adapter and captures its logits, and reports the max/mean logit deviation.
    A tiny deviation confirms the export is faithful; comparing ``quantize=None``
    vs ``"4bit"`` shows the quantization impact.
    """
    import tempfile

    import torch

    from .core import (clear_all_lora_weights, get_module_specs,
                       inject_lora_weights, replace_with_lora)
    from .export import export_peft_adapter

    cfg = config or load_config()
    device = _select_device(cfg)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    ids = tok.encode(sample_prefix, add_special_tokens=False)
    inp = torch.tensor([ids], dtype=torch.long)

    # --- reference: C2L custom injection on an fp32 base ---
    base = AutoModelForCausalLM.from_pretrained(
        cfg.base_model, torch_dtype=torch.float32).to(device).eval()
    specs = get_module_specs(base, adapter.target_modules)
    replace_with_lora(base, specs, rank=adapter.rank, alpha=adapter.alpha)
    head_out = {
        "A": {t: torch.tensor(adapter.A[t]).unsqueeze(0) for t in adapter.A},
        "B": {t: torch.tensor(adapter.B[t]).unsqueeze(0) for t in adapter.B},
    }
    inject_lora_weights(base, specs, head_out, batch_index=0)
    with torch.no_grad():
        ref_logits = base(inp.to(device)).logits[0, -1].float().cpu()
    clear_all_lora_weights(base, specs)
    del base
    if device == "cuda":
        torch.cuda.empty_cache()

    # --- exported PEFT adapter path ---
    with tempfile.TemporaryDirectory() as td:
        export_peft_adapter(adapter, td)
        eng = HFInference(adapter_dir=td, quantize=quantize, config=cfg)
        eng.load()
        with torch.no_grad():
            dev = next(eng._model.parameters()).device
            peft_logits = eng._model(inp.to(dev)).logits[0, -1].float().cpu()

    diff = (ref_logits - peft_logits).abs()
    return {
        "quantize": quantize or "none",
        "max_abs_logit_diff": float(diff.max()),
        "mean_abs_logit_diff": float(diff.mean()),
        "ref_argmax": int(ref_logits.argmax()),
        "peft_argmax": int(peft_logits.argmax()),
        "argmax_match": bool(ref_logits.argmax() == peft_logits.argmax()),
    }


__all__ = [
    "HFInference",
    "GGUFInference",
    "make_backend",
    "predict",
    "verify_export_fidelity",
]
