"""Deterministic in-memory providers for tests and offline demos (no network, no models)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from .base import Message, NoWarmup


class MockASR(NoWarmup):
    def __init__(self, text: str = "what is the battery status", delay_s: float = 0.0, name: str = "mock:asr"):
        self.text, self.delay_s, self.name = text, delay_s, name
        self.calls = 0

    async def transcribe(self, pcm16: bytes, sample_rate: int) -> str:
        self.calls += 1
        await asyncio.sleep(self.delay_s)
        return self.text


class MockLLM(NoWarmup):
    def __init__(
        self,
        reply: str = "Battery is at sixty percent. You have about twelve minutes of flight time left.",
        ttft_s: float = 0.0,
        token_delay_s: float = 0.0,
        name: str = "mock:llm",
    ):
        self.reply, self.ttft_s, self.token_delay_s, self.name = reply, ttft_s, token_delay_s, name
        self.calls = 0
        self.last_messages: list[Message] = []

    async def stream(self, messages: list[Message]) -> AsyncIterator[str]:
        self.calls += 1
        self.last_messages = messages
        await asyncio.sleep(self.ttft_s)
        for i, word in enumerate(self.reply.split(" ")):
            if i:
                await asyncio.sleep(self.token_delay_s)
            yield (" " if i else "") + word


class MockTTS(NoWarmup):
    def __init__(self, delay_s: float = 0.0, sample_rate: int = 16000, name: str = "mock:tts"):
        self.delay_s, self.sr, self.name = delay_s, sample_rate, name
        self.sentences: list[str] = []

    async def synthesize(self, text: str) -> AsyncIterator[tuple[bytes, int]]:
        self.sentences.append(text)
        await asyncio.sleep(self.delay_s)
        yield b"\x00\x00" * int(self.sr * 0.05 * max(1, len(text) // 10)), self.sr
