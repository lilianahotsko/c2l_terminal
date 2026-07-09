# code2lora (c2l-terminal)

A **self-contained**, Claude-Code-style terminal coding assistant powered by
**quantized Code2LoRA (C2L)** and **per-repository adapters** — on-device and
offline-capable.

```bash
pip install code2lora
c2lt                       # first run auto-provisions everything, then you're in
```

Project page: <https://lilianahotsko.github.io/Code2LoRA/> ·
Checkpoint: [`code2lora/c2l-evo-multitask`](https://huggingface.co/code2lora/c2l-evo-multitask) ·
Live demo: [Hugging Face Space](https://huggingface.co/spaces/code2lora/code2lora-gru-demo)

---

## Zero-config install

There is **no manual setup**. `pip install code2lora`, then run `c2lt`. On the
**first run** the tool provisions everything it needs into `~/.cache/c2l/`
(one-time, then cached):

| Provisioned automatically | What | Size |
|---|---|---|
| Repo encoder | `Qwen/Qwen3-Embedding-0.6B` | ~1.2 GB |
| C2L checkpoint | `code2lora/c2l-evo-multitask` | ~40 MB |
| Base model GGUF | `Qwen2.5-Coder-1.5B` Q4_K_M | ~1 GB |
| `llama.cpp` | prebuilt binary + convert script | ~30 MB |
| Base config/tokenizer | for adapter export | ~5 MB |

That's it — no cloning llama.cpp, no building, no `huggingface-cli download`, no
environment variables. The default backend is **GGUF (llama.cpp on CPU)**, so it
runs on a laptop with no GPU and no bitsandbytes.

Prefer to run setup explicitly (e.g. in a Dockerfile or CI)?

```bash
c2l-setup                  # provision everything now
c2l-setup --skip-base-gguf # everything except the ~1 GB base model
c2l-setup --offline        # just verify what's already cached
c2l-setup --force          # re-provision from scratch
```

> C2L is a *code-completion* model (`Qwen2.5-Coder-1.5B` + repo LoRA), not a
> chat agent. Give it a code prefix and it completes it.

---

## Quick start

```bash
cd /path/to/your/git/repo
c2lt
```

Inside the REPL:

```
c2l> /adapt                     # build the repo adapter (also makes the GGUF)
c2l> assert response.status_code ==
c2l> /task qa                   # switch to Q&A-style completion
c2l> /adapt
c2l> What does the retry helper in this repo do?
```

On first `/adapt`, the encoder + C2L checkpoint are already cached (from setup),
so it just walks your commit history and emits the adapter — CPU-friendly, a few
minutes for a small repo.

---

## Install options

```bash
pip install code2lora            # CPU + GGUF (default, recommended)
pip install "code2lora[quant]"   # optional: bitsandbytes 4/8-bit backends

# from a source checkout:
cd c2l_terminal && pip install -e .
```

Commands installed:

| Command | Purpose |
|---|---|
| `c2lt` / `c2l-terminal` | Interactive terminal REPL |
| `c2l` | Scriptable CLI (`adapt`, `run`, `export`, `verify`) |
| `c2l-setup` | Provision models + base GGUF + llama.cpp |

---

## REPL commands

| Command | Description |
|---|---|
| `/adapt [repo]` | Generate or **incrementally update** the adapter (only new commit diffs) |
| `/adapt --full [repo]` | Force a full re-walk from scratch |
| `/adapt --local` | Update from **uncommitted** local changes (1 GRU step; committed anchor kept) |
| `/task [name]` | `assert_rhs` (assertions) · `qa` (repo Q&A) |
| `/backend [name]` | `gguf` (default) · `4bit` · `8bit` · `hf` |
| `/tokens [n]` | Max new tokens (default 16) |
| `/load [path]` | Load a cached adapter (no regeneration) |
| `/export-gguf` | Convert the loaded adapter → GGUF LoRA |
| `/context add <file>` | Pin a file into every prompt |
| `/setup [--force]` | Re-run provisioning |
| `/status` · `/help` · `/quit` | Session state · help · exit |

**Prompting:** type a code prefix. Use `@path/to/file.py` to inject a file once
(handy for grounding a `qa` question in a specific module).

---

## The `c2l` CLI (scripting)

```bash
# 1. build a portable adapter for a repo (CPU, no base LLM loaded)
c2l adapt https://github.com/org/repo --task assert_rhs -o ./adapter --gguf

# 2. run it (gguf backend uses the auto-provisioned base GGUF + llama.cpp)
c2l run --adapter ./adapter --backend gguf --prefix "assert add(2, 2) == "
```

`c2l run --backend gguf` finds the base GGUF and llama.cpp automatically from
`~/.cache/c2l` — no flags needed. You can still override with `--base-gguf`,
`--lora-gguf`, or the env vars below.

---

## Configuration (all optional)

Everything works with **no configuration**. Override only if you want to:

| Variable | Purpose |
|---|---|
| `C2L_HOME` | Where models/llama.cpp are provisioned (default `~/.cache/c2l`) |
| `C2L_NO_AUTOSETUP=1` | Disable first-run auto-provisioning |
| `C2L_OFFLINE=1` | Never touch the network (use only local caches) |
| `C2L_DEVICE=cpu` | Force CPU for adapter generation |
| `C2L_BASE_GGUF` | Use a specific base GGUF instead of the provisioned one |
| `C2L_LLAMACPP` | Use an existing llama.cpp build instead of the provisioned one |
| `C2L_CKPT` | Local path to a `gru_head.best.pt` checkpoint |
| `C2L_LLAMACPP_TAG` | Pin a specific llama.cpp release (default: latest) |

Adapters cache under `~/.cache/c2l/adapters/` (keyed by fingerprint). GRU walk
state for incremental `/adapt` lives under `~/.cache/c2l/adapters/.walk_state/`.

---

## What's inside this folder

```
c2l_terminal/
├── pyproject.toml        # installs c2lt + c2l + c2l-setup
├── c2l/                  # vendored Code2LoRA SDK (runtime modules)
│   ├── bootstrap.py      # one-command provisioning (models / llama.cpp / GGUF)
│   ├── pipeline.py       # repo → adapter generation
│   ├── export.py         # PEFT + GGUF export
│   ├── infer.py          # gguf / 4bit / 8bit / hf backends
│   └── ...
└── c2l_terminal/         # interactive REPL (repl, session, commands, ui)
```

Installing from source? Copy **only this folder** — it vendors the full `c2l`
SDK; no parent repository is required.

---

## System requirements

| | Minimum | Comfortable |
|---|---|---|
| RAM | 8 GB | 16 GB |
| Disk | ~4 GB (models + deps) | 8 GB |
| CPU | any 64-bit | 4+ cores |

`/adapt` on CPU: a few minutes for a small repo. GGUF completion: ~5–30 s per
prompt on a laptop CPU.

---

## Choosing a backend

| Situation | Backend |
|---|---|
| Laptop / CPU, no GPU (default) | `gguf` (auto-provisioned) |
| CPU or small GPU + bitsandbytes | `4bit` |
| More quality, more memory | `8bit` |
| Full GPU quality | `hf` |

---

## Troubleshooting

- **First run is slow** — it's downloading ~2 GB of models + base GGUF once.
  Subsequent runs are instant. Watch the progress bar; re-run `c2l-setup` if it
  was interrupted.
- **No llama.cpp binary provisioned** — your platform may lack a prebuilt
  release asset. Install `cmake` + a C/C++ compiler and re-run `c2l-setup` (it
  will build from source), or set `C2L_LLAMACPP` to an existing build.
- **Air-gapped machine** — run `c2l-setup` once on a networked box with the same
  `C2L_HOME`, copy `~/.cache/c2l` over, then run with `C2L_OFFLINE=1`.
- **Wrong base GGUF** — must be **Qwen2.5-Coder-1.5B**; override with
  `C2L_BASE_GGUF` if you manage your own.
- **bitsandbytes on CPU** — needs ≥ 0.43; otherwise just use the default `gguf`.
