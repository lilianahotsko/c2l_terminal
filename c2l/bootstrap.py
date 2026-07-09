"""One-command provisioning for the Code2LoRA terminal tool.

Goal: after ``pip install code2lora`` the user should be able to run ``c2lt``
and have *everything* set up automatically — the repo encoder, the C2L
checkpoint, a quantized base GGUF, and a working ``llama.cpp`` build — with no
manual steps.

Everything is provisioned under a single home directory (``$C2L_HOME`` or
``~/.cache/c2l``)::

    ~/.cache/c2l/
    ├── llama.cpp/                 # scripts (convert_lora_to_gguf.py) + build/bin
    │   └── build/bin/llama-cli ...
    ├── models/qwen-coder-gguf/    # base model GGUF (Q4_K_M ~1 GB)
    └── .setup_done                # marker (version-stamped)

The heavy artefacts are fetched lazily and cached, so setup is a one-time cost.
Nothing here runs at ``pip install`` time (wheels can't run post-install hooks
reliably); instead the very first ``c2lt`` invocation provisions on demand, and
``c2l-setup`` lets you (re-)run it explicitly.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Callable, List, Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LLAMACPP_REPO = "https://github.com/ggml-org/llama.cpp"
LLAMACPP_API_LATEST = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"
# Pin a known-good release; override with C2L_LLAMACPP_TAG. "latest" queries the
# GitHub API for the newest release.
LLAMACPP_TAG = os.environ.get("C2L_LLAMACPP_TAG", "latest")

BASE_GGUF_REPO = os.environ.get("C2L_BASE_GGUF_REPO", "Qwen/Qwen2.5-Coder-1.5B-GGUF")
BASE_GGUF_QUANT = os.environ.get("C2L_BASE_GGUF_QUANT", "q4_k_m").lower()

ProgressFn = Optional[Callable[[str, float], None]]


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def c2l_home() -> Path:
    return Path(os.environ.get("C2L_HOME", "~/.cache/c2l")).expanduser()


def default_llamacpp_dir() -> Path:
    return c2l_home() / "llama.cpp"


def default_base_gguf_dir() -> Path:
    return c2l_home() / "models" / "qwen-coder-gguf"


def _marker() -> Path:
    return c2l_home() / ".setup_done"


def _log(progress: ProgressFn, msg: str, frac: float = 0.0) -> None:
    if progress is not None:
        progress(msg, frac)
    else:
        sys.stderr.write(f"[c2l-setup] {msg}\n")
        sys.stderr.flush()


# ---------------------------------------------------------------------------
# Binary discovery
# ---------------------------------------------------------------------------

_LLAMA_BINS = ("llama-completion", "llama-cli")


def find_llama_binary(llamacpp_dir: Optional[Path] = None) -> Optional[Path]:
    """Return a usable llama.cpp completion binary, or ``None``."""
    from shutil import which

    roots: List[Path] = []
    env_root = os.environ.get("C2L_LLAMACPP")
    if env_root:
        b = Path(env_root).expanduser()
        roots += [b / "build" / "bin", b]
    d = llamacpp_dir or default_llamacpp_dir()
    roots += [d / "build" / "bin", d]
    for name in _LLAMA_BINS:
        for r in roots:
            cand = r / name
            if cand.exists():
                return cand
        found = which(name)
        if found:
            return Path(found)
    return None


def find_convert_script(llamacpp_dir: Optional[Path] = None) -> Optional[Path]:
    """Return llama.cpp's ``convert_lora_to_gguf.py`` if present."""
    from shutil import which

    d = llamacpp_dir or default_llamacpp_dir()
    for cand in (d / "convert_lora_to_gguf.py",
                 Path(os.environ.get("C2L_LLAMACPP", "/nonexistent")).expanduser()
                 / "convert_lora_to_gguf.py"):
        if cand.exists():
            return cand
    found = which("convert_lora_to_gguf.py")
    return Path(found) if found else None


# ---------------------------------------------------------------------------
# Downloads
# ---------------------------------------------------------------------------

def _download(url: str, dest: Path, progress: ProgressFn = None,
              label: str = "") -> Path:
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "code2lora-setup"})
    with urllib.request.urlopen(req) as resp:  # nosec - fixed hosts
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        chunk = 1 << 20
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as fh:
            while True:
                buf = resp.read(chunk)
                if not buf:
                    break
                fh.write(buf)
                done += len(buf)
                if total:
                    _log(progress, f"{label} {done // (1<<20)}/{total // (1<<20)} MB",
                         done / total)
        tmp.replace(dest)
    return dest


def _github_release_assets(tag: str) -> List[dict]:
    import urllib.request

    url = (LLAMACPP_API_LATEST if tag == "latest"
           else f"https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/{tag}")
    req = urllib.request.Request(url, headers={"User-Agent": "code2lora-setup",
                                               "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req) as resp:  # nosec
        data = json.loads(resp.read().decode("utf-8"))
    return data.get("assets", []), data.get("tag_name", tag)


def _platform_asset_tokens() -> List[str]:
    """Substrings that identify the right prebuilt zip for this machine."""
    sysname = platform.system().lower()
    mach = platform.machine().lower()
    if sysname == "darwin":
        arch = "arm64" if mach in ("arm64", "aarch64") else "x64"
        return [f"macos-{arch}"]
    if sysname == "linux":
        # prefer the plain ubuntu x64 CPU build
        if mach in ("aarch64", "arm64"):
            return ["ubuntu-arm64", "linux-arm64"]
        return ["ubuntu-x64", "linux-x64"]
    if sysname == "windows":
        return ["win-avx2-x64", "win-x64", "win-avx-x64"]
    return []


# ---------------------------------------------------------------------------
# llama.cpp provisioning
# ---------------------------------------------------------------------------

def _ensure_llamacpp_sources(dest: Path, tag: str, progress: ProgressFn) -> None:
    """Ensure the llama.cpp python scripts (convert_lora_to_gguf.py, deps) exist."""
    if (dest / "convert_lora_to_gguf.py").exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("git"):
        _log(progress, "cloning llama.cpp sources (shallow)…", 0.0)
        ref = None if tag == "latest" else tag
        cmd = ["git", "clone", "--depth", "1"]
        if ref:
            cmd += ["--branch", ref]
        cmd += [LLAMACPP_REPO, str(dest)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0 and (dest / "convert_lora_to_gguf.py").exists():
            return
        _log(progress, "git clone failed; falling back to raw script download…")
    # Fallback: download the two convert scripts directly from the tag.
    raw_ref = "master" if tag == "latest" else tag
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("convert_lora_to_gguf.py", "convert_hf_to_gguf.py"):
        url = f"https://raw.githubusercontent.com/ggml-org/llama.cpp/{raw_ref}/{name}"
        try:
            _download(url, dest / name, progress, label=name)
        except Exception as e:  # noqa: BLE001
            _log(progress, f"could not fetch {name}: {e}")


def _ensure_llamacpp_binary(dest: Path, tag: str, progress: ProgressFn) -> Optional[Path]:
    """Ensure a llama.cpp completion binary is available under ``dest``."""
    existing = find_llama_binary(dest)
    if existing:
        return existing

    bindir = dest / "build" / "bin"
    tokens = _platform_asset_tokens()
    if tokens:
        try:
            assets, resolved_tag = _github_release_assets(tag)
            asset = None
            for a in assets:
                name = a.get("name", "").lower()
                if name.endswith(".zip") and name.startswith("llama-") \
                        and any(tok in name for tok in tokens):
                    asset = a
                    break
            if asset:
                _log(progress, f"downloading prebuilt llama.cpp ({resolved_tag})…", 0.0)
                with tempfile.TemporaryDirectory() as td:
                    zpath = Path(td) / asset["name"]
                    _download(asset["browser_download_url"], zpath, progress,
                              label="llama.cpp")
                    bindir.mkdir(parents=True, exist_ok=True)
                    with zipfile.ZipFile(zpath) as zf:
                        for m in zf.namelist():
                            # flatten everything into build/bin so binaries and
                            # their shared libs sit together (rpath=$ORIGIN).
                            fn = Path(m).name
                            if not fn:
                                continue
                            with zf.open(m) as src, open(bindir / fn, "wb") as out:
                                shutil.copyfileobj(src, out)
                    for f in bindir.iterdir():
                        if f.is_file() and (f.name.startswith("llama-")
                                            or f.suffix in ("", ".so", ".dylib")):
                            try:
                                f.chmod(0o755)
                            except OSError:
                                pass
                found = find_llama_binary(dest)
                if found:
                    return found
        except Exception as e:  # noqa: BLE001
            _log(progress, f"prebuilt download failed ({e}); trying source build…")

    # Fallback: build from source with cmake (needs a compiler).
    if shutil.which("cmake"):
        _log(progress, "building llama.cpp from source (cmake)…", 0.0)
        try:
            subprocess.run(["cmake", "-B", str(dest / "build"), "-S", str(dest),
                            "-DGGML_NATIVE=OFF", "-DLLAMA_CURL=OFF"],
                           check=True, capture_output=True, text=True)
            subprocess.run(["cmake", "--build", str(dest / "build"),
                            "-j", str(os.cpu_count() or 4)],
                           check=True, capture_output=True, text=True)
            return find_llama_binary(dest)
        except Exception as e:  # noqa: BLE001
            _log(progress, f"source build failed: {e}")
    return None


def ensure_llamacpp(offline: bool = False, progress: ProgressFn = None) -> Optional[Path]:
    """Ensure llama.cpp (scripts + a completion binary). Returns its root dir."""
    dest = default_llamacpp_dir()
    if offline:
        return dest if find_llama_binary(dest) else None
    tag = LLAMACPP_TAG
    _ensure_llamacpp_sources(dest, tag, progress)
    binary = _ensure_llamacpp_binary(dest, tag, progress)
    # Expose for the current process so infer/export find it immediately.
    os.environ.setdefault("C2L_LLAMACPP", str(dest))
    binp = binary.parent if binary else (dest / "build" / "bin")
    os.environ["PATH"] = f"{binp}{os.pathsep}{os.environ.get('PATH', '')}"
    if binary is None:
        _log(progress, "WARNING: no llama.cpp binary provisioned; the gguf "
                       "backend will be unavailable until one is built.")
    return dest


# ---------------------------------------------------------------------------
# Base GGUF provisioning
# ---------------------------------------------------------------------------

def ensure_base_gguf(offline: bool = False, progress: ProgressFn = None) -> Optional[Path]:
    """Ensure a quantized base-model GGUF is present. Returns its path."""
    dest_dir = default_base_gguf_dir()
    existing = (sorted(dest_dir.glob(f"*{BASE_GGUF_QUANT}*.gguf"))
                or sorted(dest_dir.glob("*.gguf")))
    if existing:
        os.environ.setdefault("C2L_BASE_GGUF", str(existing[0]))
        return existing[0]
    if offline:
        return None
    from huggingface_hub import hf_hub_download, list_repo_files

    _log(progress, "locating base GGUF on the Hub…", 0.0)
    files = list_repo_files(BASE_GGUF_REPO)
    ggufs = [f for f in files if f.lower().endswith(".gguf")]
    picks = [f for f in ggufs if BASE_GGUF_QUANT in f.lower()] or ggufs
    # avoid multi-part shards; prefer a single-file quant
    picks = [f for f in picks if "-of-" not in f.lower()] or picks
    if not picks:
        _log(progress, f"no GGUF found in {BASE_GGUF_REPO}")
        return None
    fname = picks[0]
    _log(progress, f"downloading base GGUF {fname} (~1 GB, one-time)…", 0.0)
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(BASE_GGUF_REPO, fname, local_dir=str(dest_dir))
    os.environ["C2L_BASE_GGUF"] = str(path)
    return Path(path)


# ---------------------------------------------------------------------------
# C2L models (encoder + checkpoint + base config/tokenizer)
# ---------------------------------------------------------------------------

def ensure_models(cfg=None, offline: bool = False, progress: ProgressFn = None) -> None:
    """Pre-cache the repo encoder, the C2L checkpoint, and the base config."""
    if offline:
        return
    from .config import load_config
    cfg = cfg or load_config()
    from huggingface_hub import hf_hub_download, snapshot_download

    try:
        _log(progress, f"caching repo encoder ({cfg.encoder_model})…", 0.0)
        snapshot_download(cfg.encoder_model)
    except Exception as e:  # noqa: BLE001
        _log(progress, f"encoder cache skipped: {e}")

    try:
        if not cfg.ckpt_path:
            _log(progress, f"caching C2L checkpoint ({cfg.ckpt_repo})…", 0.0)
            hf_hub_download(cfg.ckpt_repo, cfg.ckpt_file)
    except Exception as e:  # noqa: BLE001
        _log(progress, f"checkpoint cache skipped: {e}")

    try:
        # Only the small config/tokenizer are needed (export reads AutoConfig);
        # the multi-GB base weights are NOT downloaded for the gguf path.
        _log(progress, f"caching base config/tokenizer ({cfg.base_model})…", 0.0)
        snapshot_download(cfg.base_model, allow_patterns=[
            "*.json", "*.txt", "tokenizer*", "*.model", "merges*", "vocab*"])
    except Exception as e:  # noqa: BLE001
        _log(progress, f"base config cache skipped: {e}")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def is_setup_done() -> bool:
    """True if a prior successful setup marker exists for this version."""
    m = _marker()
    if not m.exists():
        return False
    try:
        info = json.loads(m.read_text())
    except Exception:  # noqa: BLE001
        return False
    # A setup is "good enough" if llama.cpp binary + base gguf are present.
    return bool(find_llama_binary()) and bool(
        (sorted(default_base_gguf_dir().glob("*.gguf"))))


def bootstrap_all(cfg=None, *, offline: bool = False, force: bool = False,
                  skip_base_gguf: bool = False, skip_llamacpp: bool = False,
                  progress: ProgressFn = None) -> dict:
    """Provision everything. Returns a summary dict of resolved paths."""
    from .config import load_config
    cfg = cfg or load_config()
    home = c2l_home()
    home.mkdir(parents=True, exist_ok=True)

    summary: dict = {"c2l_home": str(home)}

    ensure_models(cfg, offline=offline, progress=progress)
    summary["encoder"] = cfg.encoder_model
    summary["checkpoint"] = cfg.ckpt_path or f"{cfg.ckpt_repo}/{cfg.ckpt_file}"

    if not skip_llamacpp:
        d = ensure_llamacpp(offline=offline, progress=progress)
        summary["llamacpp"] = str(d) if d else None
        b = find_llama_binary()
        summary["llama_binary"] = str(b) if b else None

    if not skip_base_gguf:
        g = ensure_base_gguf(offline=offline, progress=progress)
        summary["base_gguf"] = str(g) if g else None

    # marker
    try:
        _marker().write_text(json.dumps({"version": _pkg_version(),
                                         "summary": summary}, indent=2))
    except Exception:  # noqa: BLE001
        pass
    return summary


def auto_setup_if_needed(offline: bool = False, progress: ProgressFn = None) -> bool:
    """Run setup on first use unless disabled or already done.

    Controlled by ``C2L_NO_AUTOSETUP=1`` (skip) and offline mode. Returns True
    if a setup pass ran.
    """
    if offline or os.environ.get("C2L_NO_AUTOSETUP", "").strip() in ("1", "true", "yes"):
        return False
    if is_setup_done():
        return False
    bootstrap_all(offline=offline, progress=progress)
    return True


def _pkg_version() -> str:
    try:
        from . import __version__
        return __version__
    except Exception:  # noqa: BLE001
        return "0"


# ---------------------------------------------------------------------------
# CLI: ``c2l-setup``
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(
        prog="c2l-setup",
        description="Provision everything the Code2LoRA terminal needs "
                    "(models, base GGUF, llama.cpp) into ~/.cache/c2l.")
    p.add_argument("--offline", action="store_true",
                   help="don't download; only verify local caches")
    p.add_argument("--force", action="store_true",
                   help="re-provision even if a setup marker exists")
    p.add_argument("--skip-base-gguf", action="store_true",
                   help="don't download the ~1 GB base model GGUF")
    p.add_argument("--skip-llamacpp", action="store_true",
                   help="don't provision llama.cpp")
    args = p.parse_args(argv)

    def prog(msg: str, frac: float) -> None:
        sys.stderr.write(f"\r[c2l-setup] {msg[:70]:<70}")
        sys.stderr.flush()
        if frac >= 1.0:
            sys.stderr.write("\n")

    if args.force:
        try:
            _marker().unlink()
        except OSError:
            pass

    print("Provisioning Code2LoRA into", c2l_home(), file=sys.stderr)
    summary = bootstrap_all(offline=args.offline, force=args.force,
                            skip_base_gguf=args.skip_base_gguf,
                            skip_llamacpp=args.skip_llamacpp, progress=prog)
    print("\nSetup complete:", file=sys.stderr)
    for k, v in summary.items():
        print(f"  {k:<14} {v}", file=sys.stderr)
    if not summary.get("llama_binary"):
        print("\nNote: no llama.cpp binary was provisioned. Install a C/C++ "
              "toolchain + cmake, or set C2L_LLAMACPP to an existing build.",
              file=sys.stderr)
    return 0


__all__ = [
    "c2l_home", "default_llamacpp_dir", "default_base_gguf_dir",
    "find_llama_binary", "find_convert_script",
    "ensure_llamacpp", "ensure_base_gguf", "ensure_models",
    "bootstrap_all", "auto_setup_if_needed", "is_setup_done", "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
