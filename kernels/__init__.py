"""Small local backend registry for the refocused project."""

from typing import Dict, Type

from .base import BaseKernel
from .openai_whisper import OpenAIWhisperKernel

KERNELS: Dict[str, Type[BaseKernel]] = {
    "openai": OpenAIWhisperKernel,
}


def get_kernel(name: str) -> BaseKernel:
    if name not in KERNELS:
        raise ValueError(f"Unknown kernel: {name}. Available: {list(KERNELS)}")
    return KERNELS[name]()
