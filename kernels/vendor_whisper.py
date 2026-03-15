"""Helpers for importing a local OpenAI Whisper source tree."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path


def vendor_repo_root() -> Path:
    return Path(__file__).resolve().parents[1] / "vendor" / "whisper"


def custom_repo_root() -> Path:
    return Path(__file__).resolve().parents[1] / "vendor" / "whisper_custom"


def whisper_cache_dir() -> Path:
    cache_dir = os.environ.get("WHISPER_CACHE_DIR")
    if cache_dir:
        return Path(cache_dir).expanduser().resolve()
    return (Path(__file__).resolve().parents[1] / "model-cache" / "whisper").resolve()


def _normalize_repo_root(repo_root: Path | str | None = None) -> Path:
    if repo_root is None:
        repo_root = vendor_repo_root()
    return Path(repo_root).expanduser().resolve()


def ensure_vendor_whisper_on_path(repo_root: Path | str | None = None) -> Path:
    repo_root = _normalize_repo_root(repo_root)
    if not repo_root.is_dir():
        raise RuntimeError(
            f"Vendored Whisper repo not found at {repo_root}. Run: git clone https://github.com/openai/whisper.git {repo_root}"
        )
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    return repo_root


def import_vendor_whisper(repo_root: Path | str | None = None, env_var: str | None = None):
    if env_var and os.environ.get(env_var):
        repo_root = os.environ.get(env_var)
    repo_root = ensure_vendor_whisper_on_path(repo_root).resolve()

    if "whisper" in sys.modules:
        loaded = Path(getattr(sys.modules["whisper"], "__file__", "")).resolve()
        if repo_root not in loaded.parents:
            for name in list(sys.modules):
                if name == "whisper" or name.startswith("whisper."):
                    del sys.modules[name]

    whisper = importlib.import_module("whisper")
    loaded = Path(getattr(whisper, "__file__", "")).resolve()
    if repo_root not in loaded.parents:
        raise RuntimeError(
            f"Imported whisper from {loaded}, expected it under {repo_root}. Activate this repo's environment and run scripts/bootstrap_local.sh or scripts/bootstrap_cluster.sh."
        )
    return whisper
