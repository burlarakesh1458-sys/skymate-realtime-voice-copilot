"""Builds provider chains from config names. Adding a vendor = one adapter + one line here."""

from __future__ import annotations

import httpx

from ..config import FaultConfig, Settings
from .faults import FaultyASR, FaultyLLM, FaultyTTS
from .groq import GroqASR, GroqLLM, GroqTTS
from .local import DegradedLLM, FasterWhisperASR, PiperTTS
from .mock import MockASR, MockLLM, MockTTS
from .stubs import CartesiaTTS, DeepgramASR, ElevenLabsTTS


def build_asr(name: str, s: Settings, http: httpx.AsyncClient):
    if name == "groq":
        return GroqASR(http, s.groq_api_key, s.groq_base_url, s.groq_asr_model)
    if name == "faster_whisper":
        return FasterWhisperASR(s.faster_whisper_model, s.faster_whisper_threads)
    if name == "deepgram":
        return DeepgramASR(http, s.deepgram_api_key)
    if name == "mock":
        return MockASR()
    raise ValueError(f"unknown ASR provider {name!r}")


def build_llm(name: str, s: Settings, http: httpx.AsyncClient):
    if name == "groq":
        return GroqLLM(
            http,
            s.groq_api_key,
            s.groq_base_url,
            s.groq_llm_model,
            s.llm_max_tokens,
            s.llm_temperature,
            s.groq_llm_reasoning_effort,
        )
    if name.startswith("groq:"):  # e.g. groq:openai/gpt-oss-120b as a secondary model
        return GroqLLM(
            http,
            s.groq_api_key,
            s.groq_base_url,
            name.split(":", 1)[1],
            s.llm_max_tokens,
            s.llm_temperature,
            s.groq_llm_reasoning_effort,
        )
    if name == "degraded":
        return DegradedLLM(s.persona)
    if name == "mock":
        return MockLLM()
    raise ValueError(f"unknown LLM provider {name!r}")


def build_tts(name: str, s: Settings, http: httpx.AsyncClient):
    if name == "groq":
        return GroqTTS(http, s.groq_api_key, s.groq_base_url, s.groq_tts_model, s.groq_tts_voice)
    if name == "piper":
        return PiperTTS(s.piper_model_path)
    if name == "elevenlabs":
        return ElevenLabsTTS(http, s.elevenlabs_api_key, s.elevenlabs_voice_id)
    if name == "cartesia":
        return CartesiaTTS(http, s.cartesia_api_key)
    if name == "mock":
        return MockTTS()
    raise ValueError(f"unknown TTS provider {name!r}")


class ProviderSet:
    """The three ordered chains. Fault wrappers decorate only the primary of each chain."""

    def __init__(self, asr: list, llm: list, tts: list):
        self.asr, self.llm, self.tts = asr, llm, tts

    @classmethod
    def from_settings(cls, s: Settings, http: httpx.AsyncClient) -> ProviderSet:
        return cls(
            [build_asr(n, s, http) for n in s.asr_chain],
            [build_llm(n, s, http) for n in s.llm_chain],
            [build_tts(n, s, http) for n in s.tts_chain],
        )

    def with_faults(self, faults: FaultConfig) -> ProviderSet:
        if not faults.any():
            return self
        wrap = lambda chain, W: ([W(chain[0], faults)] + chain[1:]) if chain else chain  # noqa: E731
        return ProviderSet(wrap(self.asr, FaultyASR), wrap(self.llm, FaultyLLM), wrap(self.tts, FaultyTTS))

    def names(self) -> dict[str, list[str]]:
        return {k: [p.name for p in getattr(self, k)] for k in ("asr", "llm", "tts")}
