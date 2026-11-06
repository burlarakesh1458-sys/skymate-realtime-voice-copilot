"""Fault-injection wrappers. They decorate the *primary* provider of each chain so we can
demonstrate (and regression-test) timeouts, retries, breaker trips and fallbacks on demand."""

from __future__ import annotations

import asyncio
import random
from collections.abc import AsyncIterator

from ..config import FaultConfig
from ..resilience import ProviderError
from .base import Message


class FaultyASR:
    def __init__(self, inner, faults: FaultConfig, rng: random.Random | None = None):
        self.inner, self.faults, self.rng = inner, faults, rng or random.Random()
        self.name = inner.name

    async def transcribe(self, pcm16: bytes, sample_rate: int) -> str:
        if self.faults.asr_extra_latency_ms:
            await asyncio.sleep(self.faults.asr_extra_latency_ms / 1000)
        if self.faults.asr_down:
            raise ProviderError(self.name, "injected fault: ASR service unavailable (503)", code="injected_503")
        if self.faults.error_rate and self.rng.random() < self.faults.error_rate:
            raise ProviderError(self.name, "injected transient fault", code="injected_transient")
        return await self.inner.transcribe(pcm16, sample_rate)

    async def warmup(self) -> None:
        await self.inner.warmup()


class FaultyLLM:
    def __init__(self, inner, faults: FaultConfig, rng: random.Random | None = None):
        self.inner, self.faults, self.rng = inner, faults, rng or random.Random()
        self.name = inner.name

    async def stream(self, messages: list[Message]) -> AsyncIterator[str]:
        if self.faults.llm_timeout:
            await asyncio.sleep(3600)  # hang forever: only the TTFT timeout can save us
        if self.faults.llm_slow_ms:
            await asyncio.sleep(self.faults.llm_slow_ms / 1000)
        if self.faults.error_rate and self.rng.random() < self.faults.error_rate:
            raise ProviderError(self.name, "injected transient fault", code="injected_transient")
        async for tok in self.inner.stream(messages):
            yield tok

    async def warmup(self) -> None:
        await self.inner.warmup()


class FaultyTTS:
    def __init__(self, inner, faults: FaultConfig, rng: random.Random | None = None):
        self.inner, self.faults, self.rng = inner, faults, rng or random.Random()
        self.name = inner.name

    async def synthesize(self, text: str) -> AsyncIterator[tuple[bytes, int]]:
        if self.faults.tts_down:
            raise ProviderError(self.name, "injected fault: TTS service unavailable (503)", code="injected_503")
        if self.faults.error_rate and self.rng.random() < self.faults.error_rate:
            raise ProviderError(self.name, "injected transient fault", code="injected_transient")
        async for item in self.inner.synthesize(text):
            yield item

    async def warmup(self) -> None:
        await self.inner.warmup()
