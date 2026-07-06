# code2lora (c2l-terminal)

A **self-contained**, Claude-Code-style terminal coding assistant powered by
**quantized Code2LoRA (C2L)** and **per-repository adapters**.

```bash
pip install code2lora
```

Project page: <https://lilianahotsko.github.io/Code2LoRA/> ·
Checkpoint: [`code2lora/code2lora-gru`](https://huggingface.co/code2lora/code2lora-gru) ·
Live demo: [Hugging Face Space](https://huggingface.co/spaces/code2lora/code2lora-gru-demo)

Installing from source instead? Copy **only this folder** to your machine — it
includes the full `c2l` SDK vendored inside (`c2l/`). No parent repository
required.

It does two things:

1. **Generates an adapter for *your* repo** — encoder + GRU + head (CPU-friendly,
   no base LLM loaded).
2. **Runs the frozen base model quantized** — 4-bit / 8-bit (bitsandbytes) or
   **GGUF (llama.cpp)** on CPU — with that adapter injected.

> C2L is a *code-completion* model (`Qwen2.5-Coder-1.5B` + repo LoRA), not a
> chat agent. Give it a code prefix and it completes it.

---

## Laptop quick start (CPU + llama.cpp)

Everything below runs from **this folder only**.

### 1. Install Python deps

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install code2lora             # or: pip install -e . (from this folder)
```

### 2. Build llama.cpp (one-time)

```bash
git clone https://github.com/ggerganov/llama.cpp ~/llama.cpp
cmake -B ~/llama.cpp/build -S ~/llama.cpp
cmake --build ~/llama.cpp/build -j
```

### 3. Download a base GGUF (one-time)

Get a **Qwen2.5-Coder-1.5B** quantized GGUF (Q4_K_M ≈ 1 GB), e.g. from HuggingFace:

```bash
pip install huggingface_hub
huggingface-cli download Qwen/Qwen2.5-Coder-1.5B-GGUF \
  --include "*Q4_K_M*" --local-dir ~/models/qwen-coder-gguf
```

### 4. Set environment variables

```bash
export C2L_LLAMACPP=~/llama.cpp
export PATH="$C2L_LLAMACPP/build/bin:$PATH"
export C2L_BASE_GGUF=~/models/qwen-coder-gguf/<your-q4-file>.gguf
export C2L_DEVICE=cpu
```

### 5. Run in your git project

```bash
cd /path/to/your/git/repo
c2lt --device cpu --backend gguf
```

Inside the REPL:

```
c2l> /adapt
c2l> /export-gguf          # converts adapter → adapter.gguf for llama.cpp
c2l> /backend gguf
c2l> assert response.status_code ==
```

On first `/adapt`, the encoder + C2L checkpoint download automatically from
HuggingFace (~3 GB). Set `C2L_OFFLINE=1` after caching for air-gapped use.

---

## What's inside this folder

```
c2l_terminal/
├── pyproject.toml       # installs c2lt + c2l CLI
├── README.md
├── c2l/                 # vendored Code2LoRA SDK (runtime modules)
│   ├── pipeline.py      # repo → adapter generation
│   ├── export.py        # PEFT + GGUF export
│   ├── infer.py         # 4bit / 8bit / gguf backends
│   └── ...
└── c2l_terminal/        # interactive REPL
    ├── repl.py
    ├── session.py
    └── commands.py
```

**Bundled `c2l` modules:** config, core, pipeline, export, infer, embedding,
git_pipeline, assertions, metrics, tasks, cli, registry.

**Not included** (not needed for the terminal tool): API server, Gradio app,
training scripts.

---

## Install (any backend)

```bash
pip install code2lora            # CPU + GGUF path
pip install "code2lora[quant]"   # optional: bitsandbytes 4/8-bit

# or from a source checkout:
cd c2l_terminal
pip install -e .
```

Commands installed:

| Command | Purpose |
|---|---|
| `c2lt` | Interactive terminal REPL |
| `c2l` | CLI (`adapt`, `run`, `export`, `verify`) |

### Models (downloaded on first `/adapt`)

| Artifact | HuggingFace id | Used for |
|---|---|---|
| Repo encoder | `Qwen/Qwen3-Embedding-0.6B` | `/adapt` |
| C2L checkpoint | `code2lora/code2lora-gru` | `/adapt` |
| Base LLM (GGUF) | you download manually | completions via llama.cpp |
| Base LLM (HF) | `Qwen/Qwen2.5-Coder-1.5B` | `4bit`/`8bit`/`hf` backends |

---

## REPL commands

| Command | Description |
|---|---|
| `/adapt [repo]` | Generate or **incrementally update** adapter (only new commit diffs) |
| `/adapt --full [repo]` | Force full re-walk from scratch |
| `/adapt --local` | Update from **uncommitted** local changes (diff vs last endpoint, 1 GRU step; committed anchor preserved) |
| `/export-gguf [path]` | Convert loaded adapter to GGUF LoRA |
| `/task [name]` | `assert_rhs` or `code_gen` |
| `/backend [name]` | `4bit` · `8bit` · `hf` · `gguf` |
| `/tokens [n]` | Max new tokens (default 64) |
| `/context add <file>` | Pin file as standing context |
| `/status` | Session state |
| `/help` | Help |
| `/quit` | Exit |

**Prompting:** type a code prefix. Use `@path/to/file.py` to inject a file once.

---

## CLI flags

```bash
c2lt [repo] [--task assert_rhs] [--backend 4bit|8bit|hf|gguf]
     [--tokens 64] [--device cuda|cpu] [--offline] [--adapt]
```

---

## Choosing a backend

| Situation | Backend |
|---|---|
| CPU, no bitsandbytes | `gguf` + llama.cpp (see quick start above) |
| CPU or small GPU + bitsandbytes | `4bit` (default) |
| More quality, more memory | `8bit` |
| Full GPU quality | `hf` |

### GGUF without the REPL

```bash
c2l adapt https://github.com/org/repo -o ./adapter
c2l export --adapter ./adapter --gguf ./adapter/adapter.gguf
export C2L_BASE_GGUF=~/models/<q4>.gguf C2L_LLAMACPP=~/llama.cpp
c2l run --adapter ./adapter --backend gguf --prefix "assert x == "
```

---

## Environment variables

| Variable | Purpose |
|---|---|
| `C2L_OFFLINE=1` | Never touch the network |
| `C2L_DEVICE=cpu` | Force CPU |
| `C2L_LLAMACPP` | Path to llama.cpp checkout |
| `C2L_BASE_GGUF` | Base model GGUF for `gguf` backend |
| `C2L_LORA_GGUF` | Override LoRA GGUF path (default: `<adapter>/adapter.gguf`) |
| `C2L_CKPT` | Local path to `code2lora_gru.pt` |

Adapters cache under `~/.cache/c2l/adapters/` (keyed by fingerprint). GRU walk
state for incremental `/adapt` is stored under
`~/.cache/c2l/adapters/.walk_state/<repo>/<task>/`.

---

## System requirements

| | Minimum | Comfortable |
|---|---|---|
| RAM | 8 GB | 16 GB |
| Disk | ~5 GB (models + deps) | 10 GB |
| CPU | any 64-bit | 4+ cores |

`/adapt` on CPU: a few minutes for a small repo.  
GGUF completion: ~5–30 s per prompt on a laptop CPU.

---

## Troubleshooting

- **"no git repo"** — `/adapt /path/to/repo` or clone a repo first.
- **`llama-cli` / `llama-completion` not found** — build llama.cpp; add `build/bin` to `PATH`. Newer builds use `llama-completion` for raw completion; the tool auto-detects either.
- **`--no-conversation is not supported by llama-cli`** — your build split out `llama-completion`; update `c2l` (`git pull`) so it prefers that binary.
- **`convert_lora_to_gguf.py` not found** — set `C2L_LLAMACPP=~/llama.cpp`.
- **Wrong base GGUF** — must be **Qwen2.5-Coder-1.5B**, not another Qwen variant.
- **bitsandbytes on CPU** — needs ≥ 0.43; otherwise use `gguf`.
