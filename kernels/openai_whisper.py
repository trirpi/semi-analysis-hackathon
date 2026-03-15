"""Whisper Tiny loaded from the vendored OpenAI Whisper source tree."""

from pathlib import Path
from typing import Optional

from .base import BaseKernel
from .vendor_whisper import import_vendor_whisper, whisper_cache_dir


class OpenAIWhisperKernel(BaseKernel):
    """Uses the cloned openai/whisper source under vendor/whisper."""

    name = "openai"

    def __init__(self) -> None:
        self._model: Optional[object] = None
        self._device: Optional[str] = None

    def load(self, device: str = "cuda") -> None:
        whisper = import_vendor_whisper(env_var="OPENAI_WHISPER_SOURCE_DIR")
        self._device = device
        self._model = whisper.load_model(
            "tiny",
            device=device,
            download_root=str(whisper_cache_dir()),
        )

    def transcribe(self, audio_path: str) -> str:
        if self._model is None:
            raise RuntimeError("OpenAIWhisperKernel: call load() before transcribe()")
        result = self._model.transcribe(
            str(Path(audio_path).resolve()),
            language=None,
            task="transcribe",
            fp16=(self._device == "cuda"),
        )
        return (result.get("text") or "").strip()

    def unload(self) -> None:
        self._model = None
        self._device = None
