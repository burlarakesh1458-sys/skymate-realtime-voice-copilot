"""Local, offline providers used for degraded mode (no network, no GPU).

* ``FasterWhisperASR`` — CTranslate2 Whisper (tiny/base, int8 on CPU).
* ``PiperTTS`` — Piper VITS voices via onnxruntime; ~60 MB model, faster than real time on CPU.
* ``DegradedLLM`` — deterministic, safety-first canned reply so the user always hears *something*.

Heavy imports are lazy so the server starts (and tests run) without these packages.
"""

from __future__ import annotations

import asyncio
import os
import threading
from collections.abc import AsyncIterator

import numpy as np

from ..audio import pcm16_to_float
from ..resilience import ProviderError
from .base import Message, NoWarmup


class FasterWhisperASR:
    def __init__(self, model_size: str = "tiny.en", threads: int = 2):
        self.model_size = model_size
        self.threads = threads
        self.name = f"faster_whisper:{model_size}"
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        with self._lock:
            if self._model is None:
                try:
                    from faster_whisper import WhisperModel
                except ImportError as e:  # pragma: no cover
                    raise ProviderError(self.name, "faster-whisper not installed", retryable=False) from e
                self._model = WhisperModel(self.model_size, device="cpu", compute_type="int8", cpu_threads=self.threads)
        return self._model

    def _run(self, audio: np.ndarray) -> str:
        model = self._load()
        segments, _ = model.transcribe(
            audio, beam_size=1, language="en", vad_filter=False, condition_on_previous_text=False
        )
        return " ".join(s.text.strip() for s in segments).strip()

    async def transcribe(self, pcm16: bytes, sample_rate: int) -> str:
        if sample_rate != 16000:
            raise ProviderError(self.name, "expects 16 kHz audio", retryable=False)
        return await asyncio.to_thread(self._run, pcm16_to_float(pcm16))

    async def warmup(self) -> None:
        await asyncio.to_thread(self._run, np.zeros(16000, dtype=np.float32))


class PiperTTS:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.name = f"piper:{os.path.basename(model_path).removesuffix('.onnx')}"
        self._voice = None
        self._lock = threading.Lock()

    def _load(self):
        with self._lock:
            if self._voice is None:
                if not os.path.exists(self.model_path):
                    raise ProviderError(
                        self.name, f"voice model missing: {self.model_path}", retryable=False, code="model_missing"
                    )
                try:
                    from piper import PiperVoice
                except ImportError as e:  # pragma: no cover
                    raise ProviderError(self.name, "piper-tts not installed", retryable=False) from e
                self._voice = PiperVoice.load(self.model_path)
        return self._voice

    def _synth_all(self, text: str) -> list[tuple[bytes, int]]:
        voice = self._load()
        return [(c.audio_int16_bytes, c.sample_rate) for c in voice.synthesize(text)]

    async def synthesize(self, text: str) -> AsyncIterator[tuple[bytes, int]]:
        # Piper yields one chunk per sentence; we already feed it one sentence at a time.
        for pcm, sr in await asyncio.to_thread(self._synth_all, text):
            yield pcm, sr

    async def warmup(self) -> None:
        # The first ONNX inference is ~10x slower (graph optimisation); pay it at boot.
        await asyncio.to_thread(self._synth_all, "Warm up.")


class DegradedLLM(NoWarmup):
    """Last-resort 'LLM': honest, safe, and instant. Never hangs, never hallucinates."""

    name = "degraded:canned"

    def __init__(self, persona: str = "drone_ops"):
        self.persona = persona

    async def stream(self, messages: list[Message]) -> AsyncIterator[str]:
        if self.persona == "drone_ops":
            text = (
                "I can't reach my reasoning service right now, so I'll keep this simple. "
                "Hold a safe hover or return to home if battery or link is marginal, "
                "and ask me again in a few seconds."
            )
        else:
            text = "Sorry, my reasoning service is unavailable right now. Please try again in a few seconds."
        for word in text.split(" "):
            await asyncio.sleep(0)
            yield word + " "
