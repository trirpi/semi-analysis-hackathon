from __future__ import annotations

import shutil
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kernels.vendor_whisper import import_vendor_whisper, vendor_repo_root, whisper_cache_dir


def main() -> int:
    print(f"Repo root: {REPO_ROOT}")
    print(f"Vendored Whisper repo: {vendor_repo_root()}")
    print(f"Whisper cache dir: {whisper_cache_dir()}")
    print(f"ffmpeg: {shutil.which('ffmpeg') or 'NOT FOUND'}")

    try:
        import torch

        print(f"torch: {torch.__version__}")
        print(f"cuda_available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"cuda_device_count: {torch.cuda.device_count()}")
            print(f"cuda_device_0: {torch.cuda.get_device_name(0)}")
    except Exception as exc:
        print(f"torch: ERROR ({exc})")

    try:
        whisper = import_vendor_whisper()
        print(f"whisper_module: {Path(whisper.__file__).resolve()}")
        print("available_models:", ", ".join(whisper.available_models()[:5]), "...")
    except Exception as exc:
        print(f"whisper: ERROR ({exc})")
        return 1

    sample = vendor_repo_root() / "tests" / "jfk.flac"
    print(f"sample_audio_exists: {sample.is_file()} ({sample})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
