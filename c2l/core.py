"""Model core for Code2LoRA generation + injection.

Promoted from ``hf_space_code2lora/c2l_demo/core.py`` (itself a faithful port
of ``hypernetwork/code2lora_core.py``) with two additions for the platform:

* :class:`Code2LoRAHead` optionally accepts a **task id** and conditions on a
  learned task embedding concatenated to the context vector, so one checkpoint
  serves many tasks. When ``num_tasks`` is 0/None it is byte-for-byte the
  original single-task head (full backward compatibility).
* :func:`specs_from_hf_config` builds the full :class:`ModuleSpec` list from a
  Hugging Face *config only* -- no model weights -- so adapter generation and
  PEFT export do not require loading the (large) base LLM.

The two networks loaded from a ``code2lora_gru.pt`` checkpoint are
:class:`CommitGRU` and :class:`Code2LoRAHead`.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# LoRA module + injection
# ---------------------------------------------------------------------------

class LoRA(nn.Module):
    """Wraps an ``nn.Linear`` with an additive low-rank update.

    Forward: ``y = base(x) + scaling * (x @ A^T) @ B^T`` where the per-call
    A: ``[rank, in_features]`` and B: ``[out_features, rank]`` come from the
    hypernet via :meth:`set_lora_weights`.
    """

    def __init__(self, base: nn.Linear, in_features: int, out_features: int,
                 rank: int, alpha: float):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scaling = float(alpha) / float(max(1, rank))
        self.A: Optional[torch.Tensor] = None  # [rank, in_features]
        self.B: Optional[torch.Tensor] = None  # [out_features, rank]

    def set_lora_weights(self, A: torch.Tensor, B: torch.Tensor) -> None:
        self.A = A
        self.B = B

    def clear_lora_weights(self) -> None:
        self.A = None
        self.B = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        if self.A is None or self.B is None:
            return y
        x_f32 = x.detach().to(torch.float32)
        A = self.A.to(torch.float32)
        B = self.B.to(torch.float32)
        delta = F.linear(F.linear(x_f32, A), B) * self.scaling
        return y + delta.to(dtype=y.dtype)


@dataclass
class ModuleSpec:
    full_name: str   # e.g. 'model.layers.5.self_attn.q_proj'
    layer_idx: int
    type: str        # e.g. 'q_proj'
    in_features: int
    out_features: int


def get_module_specs(model: nn.Module, target_module_types: List[str]
                     ) -> List[ModuleSpec]:
    """Discover every nn.Linear whose name contains a target type."""
    specs: List[ModuleSpec] = []
    pat = re.compile(r"\bmodel\.layers\.(\d+)\.")
    for name, m in model.named_modules():
        match_type = next((t for t in target_module_types if t in name), None)
        if match_type is None:
            continue
        if not isinstance(m, nn.Linear):
            continue
        m_layer = pat.search(name)
        layer_idx = int(m_layer.group(1)) if m_layer else -1
        specs.append(ModuleSpec(
            full_name=name,
            layer_idx=layer_idx,
            type=match_type,
            in_features=int(m.in_features),
            out_features=int(m.out_features),
        ))
    specs.sort(key=lambda s: (s.layer_idx, s.full_name))
    return specs


# Linear submodule of each transformer block by type. ``attn`` / ``mlp`` is the
# parent module name used by Qwen2 / Llama-style architectures.
_TYPE_PARENT = {
    "q_proj": "self_attn", "k_proj": "self_attn", "v_proj": "self_attn",
    "o_proj": "self_attn",
    "gate_proj": "mlp", "up_proj": "mlp", "down_proj": "mlp",
}


def specs_from_hf_config(config: Any, target_module_types: List[str]
                         ) -> List[ModuleSpec]:
    """Build the full :class:`ModuleSpec` list from an HF config only.

    Works for Qwen2 / Llama-style decoder configs. Lets generation/export run
    without loading the base model weights (key for CPU / air-gapped use).
    """
    hidden = int(getattr(config, "hidden_size"))
    n_layers = int(getattr(config, "num_hidden_layers"))
    n_heads = int(getattr(config, "num_attention_heads"))
    n_kv = int(getattr(config, "num_key_value_heads", n_heads))
    inter = int(getattr(config, "intermediate_size"))
    head_dim = int(getattr(config, "head_dim", hidden // n_heads))
    q_out = n_heads * head_dim
    kv_out = n_kv * head_dim

    dims = {
        "q_proj": (hidden, q_out),
        "k_proj": (hidden, kv_out),
        "v_proj": (hidden, kv_out),
        "o_proj": (q_out, hidden),
        "gate_proj": (hidden, inter),
        "up_proj": (hidden, inter),
        "down_proj": (inter, hidden),
    }
    specs: List[ModuleSpec] = []
    for layer_idx in range(n_layers):
        for t in target_module_types:
            if t not in dims:
                raise KeyError(f"Unknown target module type for config build: {t!r}")
            parent = _TYPE_PARENT[t]
            in_f, out_f = dims[t]
            specs.append(ModuleSpec(
                full_name=f"model.layers.{layer_idx}.{parent}.{t}",
                layer_idx=layer_idx, type=t,
                in_features=in_f, out_features=out_f))
    specs.sort(key=lambda s: (s.layer_idx, s.full_name))
    return specs


def replace_with_lora(model: nn.Module, specs: List[ModuleSpec],
                      rank: int, alpha: float) -> None:
    """Replace each target ``nn.Linear`` with a :class:`LoRA` wrapper. Idempotent."""
    named = dict(model.named_modules())
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    for sp in specs:
        parent_name, attr = sp.full_name.rsplit(".", 1)
        orig = getattr(named[parent_name], attr)
        if isinstance(orig, LoRA):
            continue
        assert isinstance(orig, nn.Linear), \
            f"{sp.full_name} is not nn.Linear (got {type(orig)})"
        wrapped = LoRA(orig, sp.in_features, sp.out_features,
                       rank, alpha).to(device=device, dtype=dtype)
        setattr(named[parent_name], attr, wrapped)


def inject_lora_weights(model: nn.Module, specs: List[ModuleSpec],
                        head_out: Dict[str, Dict[str, torch.Tensor]],
                        batch_index: int = 0) -> None:
    """Push ``head_out['A'][type]`` / ``head_out['B'][type]`` into each LoRA."""
    A_by_type = head_out["A"]
    B_by_type = head_out["B"]
    named = dict(model.named_modules())
    for sp in specs:
        named[sp.full_name].set_lora_weights(
            A_by_type[sp.type][batch_index],
            B_by_type[sp.type][batch_index],
        )


def clear_all_lora_weights(model: nn.Module, specs: List[ModuleSpec]) -> None:
    """Reset every wrapper LoRA to identity (base model only)."""
    named = dict(model.named_modules())
    for sp in specs:
        mod = named.get(sp.full_name)
        if isinstance(mod, LoRA):
            mod.clear_lora_weights()


def discover_module_types_and_dims(specs: List[ModuleSpec]
                                   ) -> Dict[str, Tuple[int, int]]:
    type_dims: Dict[str, Tuple[int, int]] = {}
    for sp in specs:
        if sp.type in type_dims:
            assert type_dims[sp.type] == (sp.in_features, sp.out_features), \
                f"type {sp.type} appears with inconsistent dims"
            continue
        type_dims[sp.type] = (sp.in_features, sp.out_features)
    return type_dims


# ---------------------------------------------------------------------------
# Shared LoRA generation head (now optionally task-conditioned)
# ---------------------------------------------------------------------------

class Code2LoRAHead(nn.Module):
    """Maps a context vector to one LoRA (A, B) pair per module type.

    If ``num_tasks > 0`` the head learns a per-task embedding that is
    concatenated to the context vector before the trunk, so a single
    checkpoint can serve multiple tasks. With ``num_tasks == 0`` (default) the
    head is identical to the original single-task Code2LoRAHead.
    """

    def __init__(
        self,
        input_dim: int,
        type_dims: Dict[str, Tuple[int, int]],
        hidden_dim: int = 1024,
        rank: int = 16,
        init_log_scale: float = -3.5,
        num_tasks: int = 0,
        task_dim: int = 64,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.rank = rank
        self.type_dims = dict(type_dims)
        self.types = sorted(type_dims.keys())
        self.num_tasks = int(num_tasks)
        self.task_dim = int(task_dim) if self.num_tasks > 0 else 0

        if self.num_tasks > 0:
            self.task_embedding = nn.Embedding(self.num_tasks, self.task_dim)
            nn.init.normal_(self.task_embedding.weight, std=0.02)
        else:
            self.task_embedding = None

        trunk_in = input_dim + self.task_dim
        self.trunk = nn.Sequential(
            nn.Linear(trunk_in, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.heads_A = nn.ModuleDict({
            t: nn.Linear(hidden_dim, rank * type_dims[t][0]) for t in self.types
        })
        self.heads_B = nn.ModuleDict({
            t: nn.Linear(hidden_dim, type_dims[t][1] * rank) for t in self.types
        })
        self.log_scale_A = nn.ParameterDict({
            t: nn.Parameter(torch.tensor(init_log_scale)) for t in self.types
        })
        self.log_scale_B = nn.ParameterDict({
            t: nn.Parameter(torch.tensor(init_log_scale)) for t in self.types
        })

    def forward(self, ctx: torch.Tensor, task_id: Optional[Any] = None
                ) -> Dict[str, Dict[str, torch.Tensor]]:
        if ctx.dim() == 3:
            ctx = torch.max(ctx, dim=1).values
        ctx = ctx.float()
        if self.task_embedding is not None:
            te = self._task_embed(task_id, ctx)
            ctx = torch.cat([ctx, te], dim=-1)
        h = self.trunk(ctx)
        h = F.normalize(h, p=2, dim=-1) * math.sqrt(self.hidden_dim)

        A_out: Dict[str, torch.Tensor] = {}
        B_out: Dict[str, torch.Tensor] = {}
        for t in self.types:
            in_f, out_f = self.type_dims[t]
            A_raw = self.heads_A[t](h).view(-1, self.rank, in_f)
            B_raw = self.heads_B[t](h).view(-1, out_f, self.rank)
            scale_A = torch.exp(self.log_scale_A[t]).clamp(1e-5, 0.3)
            scale_B = torch.exp(self.log_scale_B[t]).clamp(1e-5, 0.3)
            A_out[t] = torch.tanh(A_raw) * scale_A
            B_out[t] = torch.tanh(B_raw) * scale_B
        return {"A": A_out, "B": B_out}

    def _task_embed(self, task_id: Optional[Any], ctx: torch.Tensor) -> torch.Tensor:
        bsz = ctx.shape[0]
        if task_id is None:
            idx = torch.zeros(bsz, dtype=torch.long, device=ctx.device)
        elif torch.is_tensor(task_id):
            idx = task_id.to(ctx.device).long().view(-1)
            if idx.numel() == 1 and bsz > 1:
                idx = idx.expand(bsz)
        else:
            idx = torch.full((bsz,), int(task_id), dtype=torch.long, device=ctx.device)
        return self.task_embedding(idx)

    def config_dict(self) -> Dict[str, Any]:
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "rank": self.rank,
            "types": self.types,
            "type_dims": {t: list(v) for t, v in self.type_dims.items()},
            "num_tasks": self.num_tasks,
            "task_dim": self.task_dim,
        }


# ---------------------------------------------------------------------------
# Commit-streaming GRU
# ---------------------------------------------------------------------------

class CommitGRU(nn.Module):
    """One-layer GRU that ingests per-commit diff embeddings.

    ``h_0`` is a learnable projection of the initial repo-state embedding;
    ``h_T`` is fed into :class:`Code2LoRAHead`.
    """

    def __init__(
        self,
        diff_input_dim: int = 2048,
        repo_state_dim: int = 2048,
        hidden_dim: int = 2048,
    ):
        super().__init__()
        self.diff_input_dim = diff_input_dim
        self.repo_state_dim = repo_state_dim
        self.hidden_dim = hidden_dim

        self.diff_proj = nn.Sequential(
            nn.Linear(diff_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.repo_init_proj = nn.Sequential(
            nn.Linear(repo_state_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def init_hidden(self, repo_state_emb_0: torch.Tensor) -> torch.Tensor:
        h0 = self.repo_init_proj(repo_state_emb_0.float())
        return h0.unsqueeze(0)

    def step(self, diff_emb: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        x = self.diff_proj(diff_emb.float()).unsqueeze(1)  # [B, 1, H]
        _, h_new = self.gru(x, h_prev)
        return h_new

    def context(self, h: torch.Tensor) -> torch.Tensor:
        """Match the eval path: ctx = output_norm(h[-1])."""
        return self.output_norm(h[-1])


__all__ = [
    "LoRA",
    "ModuleSpec",
    "get_module_specs",
    "specs_from_hf_config",
    "replace_with_lora",
    "inject_lora_weights",
    "clear_all_lora_weights",
    "discover_module_types_and_dims",
    "Code2LoRAHead",
    "CommitGRU",
]
