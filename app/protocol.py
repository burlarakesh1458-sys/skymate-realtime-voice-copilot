"""Typed WebSocket event protocol.

Every frame on the socket is a JSON object with a ``type`` discriminator. Client->server and
server->client events are separate unions so each side validates exactly what it may receive.
Audio travels as base64 PCM16 little-endian mono (16 kHz upstream; the sample rate is stated on
every downstream ``tts.audio`` frame).
"""

from __future__ import annotations

import time
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter


class _Event(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _now_ms() -> float:
    return time.time() * 1000.0


# ----------------------------------------------------------------------------- client -> server
class SessionStart(_Event):
    type: Literal["session.start"] = "session.start"
    sample_rate: int = 16000
    label: str = "web"  # free-form run tag, stored with every metrics row
    persona: str | None = None
    server_vad: bool = True  # False = client endpoints explicitly (push-to-talk, replay)
    faults: dict[str, Any] | None = None  # honoured only when ALLOW_CLIENT_FAULTS=true


class AudioChunk(_Event):
    type: Literal["audio.chunk"] = "audio.chunk"
    data: str  # base64 PCM16 LE mono
    seq: int = 0


class EndOfUtterance(_Event):
    """Client-side endpointing (push-to-talk release, client VAD, or replay)."""

    type: Literal["vad.end_of_utterance"] = "vad.end_of_utterance"
    client_ts: float | None = None  # ms epoch when the client decided the user stopped


class TextInput(_Event):
    """Bypass ASR (typed query) — the rest of the pipeline is identical."""

    type: Literal["text.input"] = "text.input"
    text: str
    client_ts: float | None = None


class Interrupt(_Event):
    """Barge-in: cancel the in-flight turn (LLM + TTS) immediately."""

    type: Literal["interrupt"] = "interrupt"
    reason: str = "barge_in"


class ClientMetrics(_Event):
    """Client reports what it measured (e.g. EoU -> first audio played)."""

    type: Literal["metrics.client"] = "metrics.client"
    turn_id: str
    client_e2e_ms: float


class SessionEnd(_Event):
    type: Literal["session.end"] = "session.end"


ClientEvent = Annotated[
    Union[SessionStart, AudioChunk, EndOfUtterance, TextInput, Interrupt, ClientMetrics, SessionEnd],
    Field(discriminator="type"),
]
_client_adapter: TypeAdapter = TypeAdapter(ClientEvent)


def parse_client_event(raw: str | bytes | dict) -> Any:
    """Validate an incoming frame; raises ``pydantic.ValidationError`` on bad input."""
    if isinstance(raw, dict):
        return _client_adapter.validate_python(raw)
    return _client_adapter.validate_json(raw)


# ----------------------------------------------------------------------------- server -> client
class SessionStarted(_Event):
    type: Literal["session.started"] = "session.started"
    session_id: str
    providers: dict[str, list[str]]
    faults: dict[str, Any] = {}


class SpeechStarted(_Event):
    type: Literal["vad.speech_start"] = "vad.speech_start"
    ts: float = Field(default_factory=_now_ms)


class ServerEndOfUtterance(_Event):
    type: Literal["vad.end_of_utterance"] = "vad.end_of_utterance"
    turn_id: str
    source: Literal["client", "server_vad", "max_length"]
    audio_ms: float


class TranscriptPartial(_Event):
    type: Literal["transcript.partial"] = "transcript.partial"
    text: str


class TranscriptFinal(_Event):
    type: Literal["transcript.final"] = "transcript.final"
    turn_id: str
    text: str
    provider: str


class LLMToken(_Event):
    type: Literal["llm.token"] = "llm.token"
    turn_id: str
    token: str


class LLMDone(_Event):
    type: Literal["llm.done"] = "llm.done"
    turn_id: str
    text: str
    provider: str


class TTSAudio(_Event):
    type: Literal["tts.audio"] = "tts.audio"
    turn_id: str
    data: str  # base64 PCM16 LE mono
    sample_rate: int
    seq: int
    sentence: str = ""
    provider: str = ""


class Ack(_Event):
    """Degradation notice: the system is slow / degraded and says so instead of hanging."""

    type: Literal["ack"] = "ack"
    turn_id: str
    text: str
    reason: str


class TurnDone(_Event):
    type: Literal["turn.done"] = "turn.done"
    turn_id: str
    status: Literal["ok", "degraded", "cancelled", "failed"]
    text_only: bool = False


class ErrorEvent(_Event):
    type: Literal["error"] = "error"
    code: str
    message: str
    stage: str | None = None
    turn_id: str | None = None
    recoverable: bool = True


class Metrics(_Event):
    type: Literal["metrics"] = "metrics"
    turn_id: str
    budget: dict[str, float | None]
    providers: dict[str, str]
    degraded: list[str] = []


ServerEvent = Annotated[
    Union[
        SessionStarted,
        SpeechStarted,
        ServerEndOfUtterance,
        TranscriptPartial,
        TranscriptFinal,
        LLMToken,
        LLMDone,
        TTSAudio,
        Ack,
        TurnDone,
        ErrorEvent,
        Metrics,
    ],
    Field(discriminator="type"),
]
_server_adapter: TypeAdapter = TypeAdapter(ServerEvent)


def parse_server_event(raw: str | bytes | dict) -> Any:
    if isinstance(raw, dict):
        return _server_adapter.validate_python(raw)
    return _server_adapter.validate_json(raw)


CLIENT_EVENT_TYPES = [
    "session.start",
    "audio.chunk",
    "vad.end_of_utterance",
    "text.input",
    "interrupt",
    "metrics.client",
    "session.end",
]
SERVER_EVENT_TYPES = [
    "session.started",
    "vad.speech_start",
    "vad.end_of_utterance",
    "transcript.partial",
    "transcript.final",
    "llm.token",
    "llm.done",
    "tts.audio",
    "ack",
    "turn.done",
    "error",
    "metrics",
]
