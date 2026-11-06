"""Runtime configuration, loaded from environment variables (and an optional .env file).

Every provider, timeout, retry and fault-injection knob lives here so the pipeline can be
re-wired without touching code (12-factor style).
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path

try:  # optional dependency at import time
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass

ROOT = Path(__file__).resolve().parent.parent


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    return float(v) if v not in (None, "") else default


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    return int(v) if v not in (None, "") else default


def _env_list(name: str, default: str) -> list[str]:
    return [s.strip() for s in _env(name, default).split(",") if s.strip()]


@dataclass(frozen=True)
class FaultConfig:
    """Fault-injection flags used to demonstrate graceful degradation."""

    asr_down: bool = False  # primary ASR raises a 503-style error
    llm_timeout: bool = False  # primary LLM never produces a first token
    llm_slow_ms: int = 0  # extra delay before the first LLM token
    tts_down: bool = False  # primary TTS raises
    asr_extra_latency_ms: int = 0
    error_rate: float = 0.0  # random transient failure probability on primary providers

    @classmethod
    def from_env(cls) -> FaultConfig:
        return cls(
            asr_down=_env_bool("FAULT_ASR_DOWN", False),
            llm_timeout=_env_bool("FAULT_LLM_TIMEOUT", False),
            llm_slow_ms=_env_int("FAULT_LLM_SLOW_MS", 0),
            tts_down=_env_bool("FAULT_TTS_DOWN", False),
            asr_extra_latency_ms=_env_int("FAULT_ASR_EXTRA_LATENCY_MS", 0),
            error_rate=_env_float("FAULT_ERROR_RATE", 0.0),
        )

    def any(self) -> bool:
        return any(asdict(self).values())

    def merged(self, overrides: dict | None) -> FaultConfig:
        if not overrides:
            return self
        allowed = {f.name for f in fields(self)}
        return replace(self, **{k: v for k, v in overrides.items() if k in allowed})


@dataclass(frozen=True)
class Settings:
    # --- providers (ordered fallback chains) ---
    asr_chain: list[str] = field(default_factory=lambda: ["groq", "faster_whisper"])
    llm_chain: list[str] = field(default_factory=lambda: ["groq", "degraded"])
    tts_chain: list[str] = field(default_factory=lambda: ["groq", "piper"])

    # --- Groq / OpenAI-compatible ---
    groq_api_key: str = ""
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_asr_model: str = "whisper-large-v3-turbo"
    groq_llm_model: str = "openai/gpt-oss-20b"
    groq_llm_reasoning_effort: str = "low"
    groq_tts_model: str = "canopylabs/orpheus-v1-english"
    groq_tts_voice: str = "troy"
    llm_max_tokens: int = 220
    llm_temperature: float = 0.4

    # --- local fallbacks ---
    faster_whisper_model: str = "tiny.en"
    faster_whisper_threads: int = 2
    piper_model_path: str = str(ROOT / "models" / "piper" / "en_US-lessac-low.onnx")
    warmup_local: bool = True

    # --- optional third-party adapters (stubs unless keys are set) ---
    deepgram_api_key: str = ""
    elevenlabs_api_key: str = ""
    elevenlabs_voice_id: str = ""
    cartesia_api_key: str = ""

    # --- persona ---
    persona: str = "drone_ops"

    # --- audio / VAD ---
    sample_rate: int = 16000
    vad_energy_threshold: float = 0.012  # RMS on [-1, 1] floats
    vad_silence_ms: int = 700
    vad_min_speech_ms: int = 250
    max_utterance_s: float = 30.0
    chunk_mode: str = "eager"  # full | sentence | clause | eager (see chunker.py; eager won the A/B)
    asr_partials: bool = False  # interim transcripts cost one ASR call per interval
    asr_partial_interval_ms: int = 1200

    # --- resilience ---
    asr_timeout_s: float = 8.0
    llm_ttft_timeout_s: float = 4.0  # ~5x the measured p95 TTFT (813 ms)
    llm_total_timeout_s: float = 25.0
    tts_timeout_s: float = 8.0
    retry_attempts: int = 2  # retries after the first attempt (retryable errors only)
    retry_base_delay_s: float = 0.25
    retry_max_delay_s: float = 2.0
    breaker_failure_threshold: int = 3
    breaker_cooldown_s: float = 30.0
    ack_after_s: float = 1.5  # speak "one moment" if no LLM token by then
    ack_text: str = "One moment."

    # --- observability / ops ---
    metrics_db: str = str(ROOT / "reports" / "metrics.sqlite")
    metrics_jsonl: str = str(ROOT / "reports" / "metrics.jsonl")
    record_audio_dir: str = ""  # if set, every utterance is saved here for replay
    allow_client_faults: bool = False  # let session.start override fault flags (demo only)
    faults: FaultConfig = field(default_factory=FaultConfig)

    @classmethod
    def from_env(cls) -> Settings:
        d = cls()
        return cls(
            asr_chain=_env_list("ASR_CHAIN", ",".join(d.asr_chain)),
            llm_chain=_env_list("LLM_CHAIN", ",".join(d.llm_chain)),
            tts_chain=_env_list("TTS_CHAIN", ",".join(d.tts_chain)),
            groq_api_key=_env("GROQ_API_KEY", ""),
            groq_base_url=_env("GROQ_BASE_URL", d.groq_base_url),
            groq_asr_model=_env("GROQ_ASR_MODEL", d.groq_asr_model),
            groq_llm_model=_env("GROQ_LLM_MODEL", d.groq_llm_model),
            groq_llm_reasoning_effort=_env("GROQ_LLM_REASONING_EFFORT", d.groq_llm_reasoning_effort),
            groq_tts_model=_env("GROQ_TTS_MODEL", d.groq_tts_model),
            groq_tts_voice=_env("GROQ_TTS_VOICE", d.groq_tts_voice),
            llm_max_tokens=_env_int("LLM_MAX_TOKENS", d.llm_max_tokens),
            llm_temperature=_env_float("LLM_TEMPERATURE", d.llm_temperature),
            faster_whisper_model=_env("FASTER_WHISPER_MODEL", d.faster_whisper_model),
            faster_whisper_threads=_env_int("FASTER_WHISPER_THREADS", d.faster_whisper_threads),
            piper_model_path=_env("PIPER_MODEL_PATH", d.piper_model_path),
            warmup_local=_env_bool("WARMUP_LOCAL", d.warmup_local),
            deepgram_api_key=_env("DEEPGRAM_API_KEY", ""),
            elevenlabs_api_key=_env("ELEVENLABS_API_KEY", ""),
            elevenlabs_voice_id=_env("ELEVENLABS_VOICE_ID", ""),
            cartesia_api_key=_env("CARTESIA_API_KEY", ""),
            persona=_env("PERSONA", d.persona),
            vad_energy_threshold=_env_float("VAD_ENERGY_THRESHOLD", d.vad_energy_threshold),
            vad_silence_ms=_env_int("VAD_SILENCE_MS", d.vad_silence_ms),
            vad_min_speech_ms=_env_int("VAD_MIN_SPEECH_MS", d.vad_min_speech_ms),
            max_utterance_s=_env_float("MAX_UTTERANCE_S", d.max_utterance_s),
            chunk_mode=_env("CHUNK_MODE", d.chunk_mode),
            asr_partials=_env_bool("ASR_PARTIALS", d.asr_partials),
            asr_partial_interval_ms=_env_int("ASR_PARTIAL_INTERVAL_MS", d.asr_partial_interval_ms),
            asr_timeout_s=_env_float("ASR_TIMEOUT_S", d.asr_timeout_s),
            llm_ttft_timeout_s=_env_float("LLM_TTFT_TIMEOUT_S", d.llm_ttft_timeout_s),
            llm_total_timeout_s=_env_float("LLM_TOTAL_TIMEOUT_S", d.llm_total_timeout_s),
            tts_timeout_s=_env_float("TTS_TIMEOUT_S", d.tts_timeout_s),
            retry_attempts=_env_int("RETRY_ATTEMPTS", d.retry_attempts),
            retry_base_delay_s=_env_float("RETRY_BASE_DELAY_S", d.retry_base_delay_s),
            retry_max_delay_s=_env_float("RETRY_MAX_DELAY_S", d.retry_max_delay_s),
            breaker_failure_threshold=_env_int("BREAKER_FAILURE_THRESHOLD", d.breaker_failure_threshold),
            breaker_cooldown_s=_env_float("BREAKER_COOLDOWN_S", d.breaker_cooldown_s),
            ack_after_s=_env_float("ACK_AFTER_S", d.ack_after_s),
            ack_text=_env("ACK_TEXT", d.ack_text),
            metrics_db=_env("METRICS_DB", d.metrics_db),
            metrics_jsonl=_env("METRICS_JSONL", d.metrics_jsonl),
            record_audio_dir=_env("RECORD_AUDIO_DIR", ""),
            allow_client_faults=_env_bool("ALLOW_CLIENT_FAULTS", False),
            faults=FaultConfig.from_env(),
        )

    def with_overrides(self, **kw) -> Settings:
        return replace(self, **kw)

    def public_dict(self) -> dict:
        """Config safe to show to clients (no secrets)."""
        d = asdict(self)
        for k in list(d):
            if k.endswith("api_key"):
                d[k] = bool(d[k])
        return d
