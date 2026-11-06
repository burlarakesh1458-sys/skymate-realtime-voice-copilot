"""Provider interfaces. Pipelines only ever talk to these three protocols."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

Message = dict[str, str]  # {"role": ..., "content": ...}


@runtime_checkable
class ASRProvider(Protocol):
    name: str

    async def transcribe(self, pcm16: bytes, sample_rate: int) -> str:
        """Return the transcript of one complete utterance."""
        ...

    async def warmup(self) -> None: ...


@runtime_checkable
class LLMProvider(Protocol):
    name: str

    def stream(self, messages: list[Message]) -> AsyncIterator[str]:
        """Yield content tokens as they arrive."""
        ...

    async def warmup(self) -> None: ...


@runtime_checkable
class TTSProvider(Protocol):
    name: str

    def synthesize(self, text: str) -> AsyncIterator[tuple[bytes, int]]:
        """Yield (pcm16 mono bytes, sample_rate) chunks as soon as each is available."""
        ...

    async def warmup(self) -> None: ...


class NoWarmup:
    async def warmup(self) -> None:  # noqa: D401 - trivial
        return None
