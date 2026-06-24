"""``c2l`` command-line interface.

Examples
--------
Generate a portable adapter for a repo (CPU-friendly, no base model needed)::

    c2l adapt https://github.com/psf/cachecontrol --task assert_rhs -o ./adapter

Run the adapter on a 4-bit base (low VRAM) or GGUF (CPU / no-GPU)::

    c2l run --adapter ./adapter --backend 4bit --prefix "assert add(2, 2) == "
    c2l run --adapter ./adapter --backend gguf --base-gguf qwen.gguf \
            --lora-gguf ./adapter/adapter.gguf --prefix "assert add(2, 2) == "

Convert an exported adapter to a GGUF LoRA for llama.cpp::

    c2l export --adapter ./adapter --gguf ./adapter/adapter.gguf

Everything works fully offline when ``C2L_OFFLINE=1`` and the models +
checkpoint are already cached locally (secure / air-gapped deployments).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_config


def _progress(msg: str, frac: float) -> None:
    sys.stderr.write(f"\r[{frac * 100:5.1f}%] {msg[:70]:<70}")
    sys.stderr.flush()
    if frac >= 1.0:
        sys.stderr.write("\n")


def _cmd_adapt(args) -> int:
    from .pipeline import generate_adapter
    from .export import export_peft_adapter, convert_to_gguf

    cfg = load_config(args.config, device=args.device, offline=args.offline or None)
    adapter = generate_adapter(args.repo, task=args.task, config=cfg,
                               work_dir=args.work_dir, progress=_progress)
    out_dir = args.out or str(cfg.resolved_adapters_dir() / adapter.fingerprint())
    path = export_peft_adapter(adapter, out_dir)
    print(f"Adapter exported to: {path}")
    print(f"  fingerprint : {adapter.fingerprint()}")
    print(f"  repo        : {adapter.repo_id}")
    print(f"  task        : {adapter.task} (conditioned={adapter.task_conditioned})")
    print(f"  commits     : {adapter.n_commits_walked} walked  endpoint={adapter.endpoint_sha[:10]}")
    print(f"  base model  : {adapter.base_model}")

    if args.register:
        from .registry import AdapterRegistry
        reg = AdapterRegistry(config=cfg)
        rpath = reg.put(adapter, overwrite=True)
        print(f"  registered  : {rpath}")
    if args.gguf:
        try:
            g = convert_to_gguf(str(path))
            print(f"  gguf        : {g}")
        except Exception as e:
            print(f"  gguf        : skipped ({e})", file=sys.stderr)
    return 0


def _cmd_run(args) -> int:
    from .infer import make_backend

    cfg = load_config(args.config, device=args.device, offline=args.offline or None)
    prefix = args.prefix
    if args.prefix_file:
        prefix = Path(args.prefix_file).read_text(encoding="utf-8")
    if not prefix:
        print("error: provide --prefix or --prefix-file", file=sys.stderr)
        return 2

    backend = make_backend(
        args.backend, adapter_dir=args.adapter, base_gguf=args.base_gguf,
        lora_gguf=args.lora_gguf, config=cfg)
    out = backend.generate(prefix, max_new_tokens=args.max_new_tokens)
    print(out)
    return 0


def _cmd_export(args) -> int:
    from .export import convert_to_gguf

    if not args.gguf:
        print("error: nothing to do; pass --gguf <out> to convert to GGUF",
              file=sys.stderr)
        return 2
    g = convert_to_gguf(args.adapter, out_path=args.gguf,
                        llama_cpp_dir=args.llama_cpp)
    print(f"GGUF LoRA written to: {g}")
    return 0


def _cmd_verify(args) -> int:
    from .pipeline import generate_adapter
    from .infer import verify_export_fidelity

    cfg = load_config(args.config, device=args.device)
    adapter = generate_adapter(args.repo, task=args.task, config=cfg,
                               progress=_progress)
    for quant in ([None] + (["4bit"] if args.with_4bit else [])):
        res = verify_export_fidelity(adapter, sample_prefix=args.prefix,
                                     quantize=quant, config=cfg)
        print(json.dumps(res, indent=2))
    return 0


def _cmd_tasks(args) -> int:
    from . import tasks as T
    for tid in T.list_tasks():
        t = T.get_task(tid)
        print(f"{tid:<14} idx={t.task_index}  test_files={t.mines_test_files}  "
              f"-- {t.description}")
    return 0


def _cmd_config(args) -> int:
    cfg = load_config(args.config)
    print(json.dumps(cfg.to_dict(), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="c2l", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", help="path to a c2l.yaml (else env / packaged default)")
    p.add_argument("--device", help="cuda / cpu (default: auto)")
    p.add_argument("--offline", action="store_true", help="never touch the network")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("adapt", help="generate a portable LoRA adapter for a repo")
    a.add_argument("repo", help="git URL or local path to a repository")
    a.add_argument("--task", default="assert_rhs")
    a.add_argument("-o", "--out", help="output adapter directory")
    a.add_argument("--work-dir", help="where to clone repos")
    a.add_argument("--register", action="store_true", help="store in the registry")
    a.add_argument("--gguf", action="store_true", help="also convert to GGUF")
    a.set_defaults(func=_cmd_adapt)

    r = sub.add_parser("run", help="run an adapter on a backend")
    r.add_argument("--adapter", help="exported PEFT adapter directory")
    r.add_argument("--backend", default="hf", choices=["hf", "4bit", "8bit", "gguf"])
    r.add_argument("--prefix", help="prompt prefix text")
    r.add_argument("--prefix-file", help="read the prompt prefix from a file")
    r.add_argument("--max-new-tokens", type=int, default=32)
    r.add_argument("--base-gguf", help="base model GGUF (gguf backend)")
    r.add_argument("--lora-gguf", help="LoRA GGUF (gguf backend)")
    r.set_defaults(func=_cmd_run)

    e = sub.add_parser("export", help="convert an adapter to GGUF")
    e.add_argument("--adapter", required=True, help="exported PEFT adapter dir")
    e.add_argument("--gguf", help="output GGUF path")
    e.add_argument("--llama-cpp", help="path to a llama.cpp checkout")
    e.set_defaults(func=_cmd_export)

    v = sub.add_parser("verify", help="check exported-adapter numerical fidelity")
    v.add_argument("repo", help="git URL or local path to a repository")
    v.add_argument("--task", default="assert_rhs")
    v.add_argument("--prefix", default="assert add(2, 2) == ")
    v.add_argument("--with-4bit", action="store_true")
    v.set_defaults(func=_cmd_verify)

    sub.add_parser("tasks", help="list registered tasks").set_defaults(func=_cmd_tasks)
    sub.add_parser("config", help="print the resolved configuration").set_defaults(func=_cmd_config)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
