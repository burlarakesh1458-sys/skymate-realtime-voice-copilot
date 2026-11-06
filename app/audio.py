"""Audio helpers: PCM16 <-> float, WAV I/O, resampling, and a tiny energy VAD for endpointing."""

from __future__ import annotations

import base64
import io
import wave
from dataclasses import dataclass

import numpy as np


def b64_to_pcm16(data: str) -> bytes:
    return base64.b64decode(data)


def pcm16_to_b64(pcm: bytes) -> str:
    return base64.b64encode(pcm).decode("ascii")


def pcm16_to_float(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0


def float_to_pcm16(x: np.ndarray) -> bytes:
    return (np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def pcm16_to_wav_bytes(pcm: bytes, sample_rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


def wav_bytes_to_pcm16(data: bytes) -> tuple[bytes, int]:
    """Decode a PCM WAV (any channels / 16-bit) to mono PCM16 + sample rate."""
    with wave.open(io.BytesIO(data), "rb") as w:
        sr, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
        frames = w.readframes(w.getnframes())
    if sw != 2:
        raise ValueError(f"only 16-bit WAV supported, got {sw * 8}-bit")
    x = np.frombuffer(frames, dtype="<i2")
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1).astype("<i2")
    return x.tobytes(), sr


def read_wav_16k(path: str) -> bytes:
    with open(path, "rb") as f:
        pcm, sr = wav_bytes_to_pcm16(f.read())
    return resample_pcm16(pcm, sr, 16000)


def resample_pcm16(pcm: bytes, src_sr: int, dst_sr: int) -> bytes:
    if src_sr == dst_sr or not pcm:
        return pcm
    x = pcm16_to_float(pcm)
    n = int(round(len(x) * dst_sr / src_sr))
    y = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x)
    return float_to_pcm16(y.astype(np.float32))


def rms(pcm: bytes) -> float:
    if not pcm:
        return 0.0
    x = pcm16_to_float(pcm)
    return float(np.sqrt(np.mean(x * x)))


def duration_ms(pcm: bytes, sample_rate: int = 16000) -> float:
    return len(pcm) / 2 / sample_rate * 1000.0


@dataclass
class VADResult:
    speech_started: bool = False
    end_of_utterance: bool = False


class EnergyVAD:
    """Frame-energy endpointing.

    Not a neural VAD — deliberately dependency-free. It declares speech after
    ``min_speech_ms`` of frames above threshold and end-of-utterance after
    ``silence_ms`` of trailing silence. Clients can also endpoint themselves.
    """

    def __init__(self, threshold: float, silence_ms: int, min_speech_ms: int, sample_rate: int = 16000):
        self.threshold = threshold
        self.silence_ms = silence_ms
        self.min_speech_ms = min_speech_ms
        self.sr = sample_rate
        self.reset()

    def reset(self) -> None:
        self.in_speech = False
        self.speech_ms = 0.0
        self.silence_run_ms = 0.0

    def process(self, pcm: bytes) -> VADResult:
        res = VADResult()
        ms = duration_ms(pcm, self.sr)
        loud = rms(pcm) >= self.threshold
        if loud:
            self.speech_ms += ms
            self.silence_run_ms = 0.0
            if not self.in_speech and self.speech_ms >= self.min_speech_ms:
                self.in_speech = True
                res.speech_started = True
        else:
            if self.in_speech:
                self.silence_run_ms += ms
                if self.silence_run_ms >= self.silence_ms:
                    res.end_of_utterance = True
                    self.reset()
            else:
                self.speech_ms = max(0.0, self.speech_ms - ms)  # decay noise blips
        return res
