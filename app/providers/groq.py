"""Groq (OpenAI-compatible) adapters: Whisper ASR, streaming chat LLM, Orpheus TTS.

``base_url`` is configurable, so the LLM adapter works with any OpenAI-compatible server
(OpenAI, Together, vLLM, Ollama's /v1 endpoint, ...).
"""

from __future__ import annotations

import json
import re
import struct
from collections.abc import AsyncIterator

import httpx

from ..audio import pcm16_to_wav_bytes
from ..resilience import ProviderError
from .base import Message, NoWarmup

_PERMANENT_STATUS = {400, 401, 403, 404, 422}


_TRY_AGAIN = re.compile(r"try again in (?:(\d+)m)?([\d.]+)(ms|s)", re.IGNORECASE)


def parse_retry_after(headers: httpx.Headers | dict | None, body: str) -> float | None:
    """Seconds to wait, from a Retry-After header or Groq's "Please try again in 2.5s" message."""
    if headers:
        v = headers.get("retry-after")
        if v:
            try:
                return float(v)
            except ValueError:
                pass
    m = _TRY_AGAIN.search(body or "")
    if m:
        mins, val, unit = m.groups()
        secs = float(val) / (1000 if unit.lower() == "ms" else 1)
        return secs + 60 * int(mins or 0)
    return None


def _raise_for(provider: str, status: int, body: str, headers: httpx.Headers | dict | None = None) -> None:
    code = "http_error"
    try:
        err = json.loads(body).get("error", {})
        code = err.get("code") or err.get("type") or code
        msg = err.get("message", body)
    except Exception:
        msg = body
    raise ProviderError(
        provider,
        f"HTTP {status}: {msg[:300]}",
        retryable=status not in _PERMANENT_STATUS,
        code=code,
        retry_after_s=parse_retry_after(headers, msg) if status in (429, 503) else None,
    )


def _wrap_transport(provider: str, e: Exception) -> ProviderError:
    return ProviderError(provider, f"{type(e).__name__}: {e}", retryable=True, code="transport")


class _GroqBase(NoWarmup):
    def __init__(self, client: httpx.AsyncClient, api_key: str, base_url: str):
        self.client = client
        self.base_url = base_url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}


class GroqASR(_GroqBase):
    def __init__(self, client, api_key, base_url, model: str):
        super().__init__(client, api_key, base_url)
        self.model = model
        self.name = f"groq:{model}"

    async def transcribe(self, pcm16: bytes, sample_rate: int) -> str:
        wav = pcm16_to_wav_bytes(pcm16, sample_rate)
        try:
            r = await self.client.post(
                f"{self.base_url}/audio/transcriptions",
                headers=self.headers,
                files={"file": ("utterance.wav", wav, "audio/wav")},
                data={"model": self.model, "response_format": "json", "language": "en", "temperature": "0"},
            )
        except httpx.HTTPError as e:
            raise _wrap_transport(self.name, e) from e
        if r.status_code != 200:
            _raise_for(self.name, r.status_code, r.text, r.headers)
        return r.json().get("text", "").strip()


class GroqLLM(_GroqBase):
    def __init__(
        self,
        client,
        api_key,
        base_url,
        model: str,
        max_tokens: int,
        temperature: float,
        reasoning_effort: str | None = "low",
    ):
        super().__init__(client, api_key, base_url)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.name = f"groq:{model}"

    def _payload(self, messages: list[Message]) -> dict:
        p: dict = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "max_completion_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if "gpt-oss" in self.model:
            # gpt-oss is a reasoning model: keep reasoning short and out of the spoken stream.
            p["reasoning_effort"] = self.reasoning_effort or "low"
            p["include_reasoning"] = False
        return p

    async def stream(self, messages: list[Message]) -> AsyncIterator[str]:
        try:
            async with self.client.stream(
                "POST", f"{self.base_url}/chat/completions", headers=self.headers, json=self._payload(messages)
            ) as r:
                if r.status_code != 200:
                    body = (await r.aread()).decode(errors="replace")
                    _raise_for(self.name, r.status_code, body, r.headers)
                async for line in r.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        return
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    if "error" in chunk:
                        raise ProviderError(self.name, str(chunk["error"])[:300], retryable=True, code="stream_error")
                    for ch in chunk.get("choices", []):
                        tok = (ch.get("delta") or {}).get("content")
                        if tok:
                            yield tok
        except httpx.HTTPError as e:
            raise _wrap_transport(self.name, e) from e


def parse_wav_header(buf: bytes) -> tuple[int, int] | None:
    """Return (sample_rate, data_offset) once the header is complete, else None.

    Streaming WAVs often carry placeholder sizes, so we only trust fmt + the 'data' tag.
    """
    if len(buf) < 12 or buf[:4] != b"RIFF" or buf[8:12] != b"WAVE":
        if len(buf) >= 12:
            raise ValueError("not a WAV stream")
        return None
    pos, sr = 12, None
    while pos + 8 <= len(buf):
        cid, size = buf[pos : pos + 4], struct.unpack("<I", buf[pos + 4 : pos + 8])[0]
        if cid == b"fmt ":
            if pos + 8 + 16 > len(buf):
                return None
            sr = struct.unpack("<I", buf[pos + 12 : pos + 16])[0]
        if cid == b"data":
            return (sr or 24000, pos + 8)
        pos += 8 + size + (size & 1)
    return None


class GroqTTS(_GroqBase):
    """Orpheus via /audio/speech. Streams the WAV body and yields PCM as it arrives."""

    def __init__(self, client, api_key, base_url, model: str, voice: str):
        super().__init__(client, api_key, base_url)
        self.model = model
        self.voice = voice
        self.name = f"groq:{model}"

    async def synthesize(self, text: str) -> AsyncIterator[tuple[bytes, int]]:
        payload = {"model": self.model, "input": text, "voice": self.voice, "response_format": "wav"}
        try:
            async with self.client.stream(
                "POST", f"{self.base_url}/audio/speech", headers=self.headers, json=payload
            ) as r:
                if r.status_code != 200:
                    body = (await r.aread()).decode(errors="replace")
                    _raise_for(self.name, r.status_code, body, r.headers)
                buf, header, carry = b"", None, b""
                async for chunk in r.aiter_bytes():
                    if header is None:
                        buf += chunk
                        header = parse_wav_header(buf)
                        if header is None:
                            continue
                        chunk = buf[header[1] :]
                    data = carry + chunk
                    even = len(data) - (len(data) & 1)
                    carry = data[even:]
                    if even:
                        yield data[:even], header[0]
        except httpx.HTTPError as e:
            raise _wrap_transport(self.name, e) from e
