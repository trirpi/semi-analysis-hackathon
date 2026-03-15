"""
Abstract base for Whisper Tiny inference kernels.
Implement load() and transcribe(audio_path) to add a new kernel.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional


class BaseKernel(ABC):
    """Base class for all Whisper Tiny kernels."""

    name: str = "base"

    @abstractmethod
    def load(self, device: str = "cuda") -> None:
        """Load the model. Called once before any transcribe() calls."""
        pass

    @abstractmethod
    def transcribe(self, audio_path: str) -> str:
        """
        Transcribe a single audio file. Return the transcribed text only (no timestamps).
        audio_path: path to wav/flac/mp3 etc.
        """
        pass

    def unload(self) -> None:
        """Optional: release GPU memory. Default no-op."""
        pass

    def __enter__(self) -> "BaseKernel":
        return self

    def __exit__(self, *args: object) -> None:
        self.unload()
