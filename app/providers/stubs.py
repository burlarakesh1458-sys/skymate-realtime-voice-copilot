"""Optional third-party adapters (Deepgram ASR, ElevenLabs TTS, Cartesia TTS).

These follow each vendor's public REST API but were NOT exercised in this repo's measurements
(no keys available). They fail fast with a permanent ``ProviderError`` when unconfigured, so
putting them in a chain without a key simply falls through to the next provider.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx

from ..audio import pcm16_to_wav_bytes
from ..resilience import ProviderError
from .base import NoWarmup
from .groq import _raise_for, _wrap_transport


class DeepgramASR(NoWarmup):
    def __init__(self, client: httpx.AsyncClient, api_key: str, model: str = "nova-3"):
        self.client, self.api_key, self.model = client, api_key, model
        self.name = f"deepgram:{model}"

    async def transcribe(self, pcm16: bytes, sample_rate: int) -> str:
        if not self.api_key:
            raise ProviderError(self.name, "DEEPGRAM_API_KEY not set", retryable=False, code="not_configured")
        try:
            r = await self.client.post(
                f"https://api.deepgram.com/v1/listen?model={self.model}&smart_format=true",
                headers={"Authorization": f"Token {self.api_key}", "Content-Type": "audio/wav"},
                content=pcm16_to_wav_bytes(pcm16, sample_rate),
            )
        except httpx.HTTPError as e:
            raise _wrap_transport(self.name, e) from e
        if r.status_code != 200:
            _raise_for(self.name, r.status_code, r.text)
        alts = r.json()["results"]["channels"][0]["alternatives"]
        return alts[0]["transcript"].strip() if alts else ""


class ElevenLabsTTS(NoWarmup):
    def __init__(self, client: httpx.AsyncClient, api_key: str, voice_id: str, model: str = "eleven_flash_v2_5"):
        self.client, self.api_key, self.voice_id, self.model = client, api_key, voice_id, model
        self.name = f"elevenlabs:{model}"

    async def synthesize(self, text: str) -> AsyncIterator[tuple[bytes, int]]:
        if not (self.api_key and self.voice_id):
            raise ProviderError(
                self.name, "ELEVENLABS_API_KEY/VOICE_ID not set", retryable=False, code="not_configured"
            )
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}/stream?output_format=pcm_16000"
        try:
            async with self.client.stream(
                "POST", url, headers={"xi-api-key": self.api_key}, json={"text": text, "model_id": self.model}
            ) as r:
                if r.status_code != 200:
                    _raise_for(self.name, r.status_code, (await r.aread()).decode(errors="replace"))
                carry = b""
                async for chunk in r.aiter_bytes():
                    data = carry + chunk
                    even = len(data) - (len(data) & 1)
                    carry = data[even:]
                    if even:
                        yield data[:even], 16000
        except httpx.HTTPError as e:
            raise _wrap_transport(self.name, e) from e


class CartesiaTTS(NoWarmup):
    def __init__(self, client: httpx.AsyncClient, api_key: str, voice_id: str = "", model: str = "sonic-2"):
        self.client, self.api_key, self.voice_id, self.model = client, api_key, voice_id, model
        self.name = f"cartesia:{model}"

    async def synthesize(self, text: str) -> AsyncIterator[tuple[bytes, int]]:
        if not (self.api_key and self.voice_id):
            raise ProviderError(self.name, "CARTESIA_API_KEY/voice not set", retryable=False, code="not_configured")
        body = {
            "model_id": self.model,
            "transcript": text,
            "voice": {"mode": "id", "id": self.voice_id},
            "output_format": {"container": "raw", "encoding": "pcm_s16le", "sample_rate": 16000},
        }
        try:
            async with self.client.stream(
                "POST",
                "https://api.cartesia.ai/tts/bytes",
                headers={"X-API-Key": self.api_key, "Cartesia-Version": "2024-06-10"},
                json=body,
            ) as r:
                if r.status_code != 200:
                    _raise_for(self.name, r.status_code, (await r.aread()).decode(errors="replace"))
                carry = b""
                async for chunk in r.aiter_bytes():
                    data = carry + chunk
                    even = len(data) - (len(data) & 1)
                    carry = data[even:]
                    if even:
                        yield data[:even], 16000
        except httpx.HTTPError as e:
            raise _wrap_transport(self.name, e) from e
