"""The voice turn: ASR -> streaming LLM -> sentence chunker -> streaming TTS, with a latency
budget recorded for every turn and resilience (timeouts, retries, breakers, fallbacks) at
every stage.

Concurrency model for one turn::

    ASR (await) ──► llm_task ──tokens──► chunker ──segments──► asyncio.Queue ──► tts_task ──► emit(tts.audio)
                       └─ ack_task: if no token after ACK_AFTER_S, speak "One moment." (never hang silently)

Barge-in cancels the turn task; ``finally`` blocks cancel the children and still persist metrics.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import wave
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field

from .audio import duration_ms, pcm16_to_b64
from .chunker import SentenceChunker
from .config import Settings
from .metrics import MetricsStore, Stopwatch, TurnRecord, compute_budget
from .persona import system_prompt
from .protocol import (
    Ack,
    ErrorEvent,
    LLMDone,
    LLMToken,
    Metrics,
    TranscriptFinal,
    TTSAudio,
    TurnDone,
)
from .providers.registry import ProviderSet
from .resilience import CircuitBreaker, CircuitOpenError, ProviderError, retry_async

log = logging.getLogger("voice.pipeline")
Emit = Callable[[object], Awaitable[None]]
ASR_APOLOGY = "Sorry, I'm having trouble hearing right now. Please say that again in a moment."
FRAME_BYTES = 16000 * 2 // 4  # 250 ms of 16 kHz PCM16 per tts.audio frame (scaled by sample rate)


# Fixed phrases ("One moment.", the ASR apology) are synthesized once per TTS voice and replayed
# from memory, so the degradation notice itself never waits on a slow/contended TTS.
_PHRASE_CACHE: dict[tuple[str, str], list[tuple[bytes, int]]] = {}


class AllProvidersFailed(Exception):
    def __init__(self, stage: str, errors: list[str]):
        super().__init__(f"{stage}: all providers failed: {errors}")
        self.stage, self.errors = stage, errors


@dataclass
class _Segment:
    text: str
    is_ack: bool = False


@dataclass
class TurnContext:
    turn_id: str
    session_id: str
    label: str
    emit: Emit
    history: list[dict]
    sw: Stopwatch = field(default_factory=Stopwatch)
    record: TurnRecord | None = None
    tts_error_reported: bool = False


class BreakerRegistry:
    """Process-wide breakers keyed by provider name (shared across sessions, as in production)."""

    def __init__(self, threshold: int, cooldown_s: float):
        self.threshold, self.cooldown_s = threshold, cooldown_s
        self._b: dict[str, CircuitBreaker] = {}

    def get(self, name: str) -> CircuitBreaker:
        if name not in self._b:
            self._b[name] = CircuitBreaker(name, self.threshold, self.cooldown_s)
        return self._b[name]

    def snapshot(self) -> dict:
        return {k: v.snapshot() for k, v in self._b.items()}


async def _first_item(agen: AsyncIterator, first_timeout_s: float):
    """Await the first item of an async iterator under a timeout; return (item, iterator)."""
    it = agen.__aiter__()
    try:
        first = await asyncio.wait_for(it.__anext__(), timeout=first_timeout_s)
    except BaseException:
        aclose = getattr(it, "aclose", None)
        if aclose:
            try:
                await aclose()
            except Exception:  # pragma: no cover
                pass
        raise
    return first, it


class VoicePipeline:
    def __init__(
        self,
        settings: Settings,
        providers: ProviderSet,
        breakers: BreakerRegistry | None = None,
        store: MetricsStore | None = None,
    ):
        self.s = settings
        self.p = providers
        self.breakers = breakers or BreakerRegistry(settings.breaker_failure_threshold, settings.breaker_cooldown_s)
        self.store = store

    # ------------------------------------------------------------------ helpers
    async def _call_with_policy(self, provider, stage: str, make_call: Callable[[], Awaitable], ctx: TurnContext):
        breaker = self.breakers.get(provider.name)

        def on_retry(n: int, e: Exception) -> None:
            ctx.record.errors.append(f"{stage}:{provider.name}: retry {n} after {e}")
            log.info("retry %s %s #%d: %s", stage, provider.name, n, e)

        return await breaker.call(
            lambda: retry_async(
                make_call,
                attempts=self.s.retry_attempts,
                base_delay=self.s.retry_base_delay_s,
                max_delay=self.s.retry_max_delay_s,
                on_retry=on_retry,
            )
        )

    def _note_failure(self, ctx: TurnContext, stage: str, provider, e: BaseException) -> None:
        kind = "timeout" if isinstance(e, asyncio.TimeoutError) else type(e).__name__
        msg = f"{stage}:{provider.name}: {kind}: {e}"
        ctx.record.errors.append(msg[:400])
        log.warning(msg)

    async def _emit_error(self, ctx: TurnContext, stage: str, code: str, message: str, recoverable=True):
        await ctx.emit(
            ErrorEvent(code=code, message=message, stage=stage, turn_id=ctx.turn_id, recoverable=recoverable)
        )

    # ------------------------------------------------------------------ ASR
    async def transcribe(self, ctx: TurnContext, pcm: bytes) -> str:
        errors: list[str] = []
        for i, provider in enumerate(self.p.asr):
            try:
                text = await self._call_with_policy(
                    provider,
                    "asr",
                    lambda provider=provider: asyncio.wait_for(
                        provider.transcribe(pcm, self.s.sample_rate), timeout=self.s.asr_timeout_s
                    ),
                    ctx,
                )
                ctx.record.providers["asr"] = provider.name
                if i > 0:
                    ctx.record.degraded.append(f"asr_fallback:{provider.name}")
                return text
            except asyncio.CancelledError:
                raise
            except (TimeoutError, ProviderError, Exception) as e:  # noqa: BLE001
                self._note_failure(ctx, "asr", provider, e)
                errors.append(f"{provider.name}: {e}")
                if not isinstance(e, CircuitOpenError):
                    await self._emit_error(
                        ctx,
                        "asr",
                        getattr(e, "code", "timeout"),
                        f"{provider.name} failed; "
                        + ("falling back" if i + 1 < len(self.p.asr) else "no fallback left"),
                    )
        raise AllProvidersFailed("asr", errors)

    # ------------------------------------------------------------------ LLM
    async def _llm_produce(
        self, ctx: TurnContext, messages: list[dict], queue: asyncio.Queue, first_token: asyncio.Event
    ) -> str:
        chunker = SentenceChunker(mode=self.s.chunk_mode)
        out: list[str] = []
        errors: list[str] = []
        sw = ctx.sw

        async def push(tok: str) -> None:
            if not first_token.is_set():
                sw.mark("llm_first_token")
                first_token.set()
            out.append(tok)
            await ctx.emit(LLMToken(turn_id=ctx.turn_id, token=tok))
            for seg in chunker.feed(tok):
                sw.mark("first_segment")
                await queue.put(_Segment(seg))

        for i, provider in enumerate(self.p.llm):
            breaker = self.breakers.get(provider.name)
            try:
                first, it = await self._call_with_policy(
                    provider,
                    "llm",
                    lambda provider=provider: _first_item(provider.stream(messages), self.s.llm_ttft_timeout_s),
                    ctx,
                )
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                self._note_failure(ctx, "llm", provider, e)
                errors.append(f"{provider.name}: {e}")
                if not isinstance(e, CircuitOpenError):
                    await self._emit_error(
                        ctx,
                        "llm",
                        getattr(e, "code", "timeout"),
                        f"{provider.name} failed; "
                        + ("falling back" if i + 1 < len(self.p.llm) else "no fallback left"),
                    )
                continue

            ctx.record.providers["llm"] = provider.name
            if i > 0:
                ctx.record.degraded.append(f"llm_fallback:{provider.name}")
            await push(first)
            deadline = time.monotonic() + self.s.llm_total_timeout_s
            try:
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("llm total timeout")
                    try:
                        tok = await asyncio.wait_for(it.__anext__(), timeout=remaining)
                    except StopAsyncIteration:
                        break
                    await push(tok)
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001 — mid-stream failure: keep what we have
                self._note_failure(ctx, "llm", provider, e)
                breaker.record_failure(permanent=False)
                ctx.record.degraded.append("llm_truncated")
                await self._emit_error(ctx, "llm", "stream_interrupted", "LLM stream interrupted; answer truncated")
            finally:
                aclose = getattr(it, "aclose", None)
                if aclose:
                    try:
                        await aclose()
                    except Exception:  # pragma: no cover
                        pass
            for seg in chunker.flush():
                sw.mark("first_segment")
                await queue.put(_Segment(seg))
            sw.mark("llm_done")
            text = "".join(out).strip()
            await ctx.emit(LLMDone(turn_id=ctx.turn_id, text=text, provider=provider.name))
            return text
        raise AllProvidersFailed("llm", errors)

    # ------------------------------------------------------------------ TTS
    async def _speak_segment(self, ctx: TurnContext, seg: _Segment, seq: list[int]) -> bool:
        """Synthesize one segment through the TTS chain. Returns False if every provider failed."""
        sw = ctx.sw
        if seg.is_ack:
            for provider in self.p.tts:
                cached = _PHRASE_CACHE.get((provider.name, seg.text))
                if cached:  # pre-rendered audio plays even if that TTS service is down right now
                    for pcm, sr in cached:
                        await self._emit_pcm(ctx, seg, seq, provider, pcm, sr)
                    return True
        for i, provider in enumerate(self.p.tts):
            try:
                first, it = await self._call_with_policy(
                    provider,
                    "tts",
                    lambda provider=provider: _first_item(provider.synthesize(seg.text), self.s.tts_timeout_s),
                    ctx,
                )
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                self._note_failure(ctx, "tts", provider, e)
                if not isinstance(e, CircuitOpenError) and not ctx.tts_error_reported:
                    await self._emit_error(
                        ctx,
                        "tts",
                        getattr(e, "code", "timeout"),
                        f"{provider.name} failed; " + ("falling back" if i + 1 < len(self.p.tts) else "text-only"),
                    )
                    ctx.tts_error_reported = True  # report a failing provider once per turn, not per sentence
                continue
            if not seg.is_ack:
                ctx.record.providers["tts"] = provider.name
                if i > 0 and f"tts_fallback:{provider.name}" not in ctx.record.degraded:
                    ctx.record.degraded.append(f"tts_fallback:{provider.name}")
                sw.mark("tts_first_byte")
            item = first
            collected: list[tuple[bytes, int]] = []
            try:
                while item is not None:
                    pcm, sr = item
                    if seg.is_ack:
                        collected.append((pcm, sr))
                    await self._emit_pcm(ctx, seg, seq, provider, pcm, sr)
                    try:
                        item = await asyncio.wait_for(it.__anext__(), timeout=self.s.tts_timeout_s)
                    except StopAsyncIteration:
                        item = None
                if collected:
                    _PHRASE_CACHE[(provider.name, seg.text)] = collected
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001 — partial audio already sent; move on
                self._note_failure(ctx, "tts", provider, e)
            return True
        return False

    async def _emit_pcm(self, ctx: TurnContext, seg: _Segment, seq: list[int], provider, pcm: bytes, sr: int):
        frame = FRAME_BYTES * sr // 16000
        for off in range(0, len(pcm), frame):
            await ctx.emit(
                TTSAudio(
                    turn_id=ctx.turn_id,
                    data=pcm16_to_b64(pcm[off : off + frame]),
                    sample_rate=sr,
                    seq=seq[0],
                    sentence=seg.text if off == 0 else "",
                    provider=provider.name,
                )
            )
            seq[0] += 1
            ctx.sw.mark("ack_audio_sent" if seg.is_ack else "first_audio_sent")

    async def prime_phrase_cache(self, phrases: list[str] | None = None) -> None:
        """Pre-synthesize fixed phrases with the first TTS provider that works (call at boot)."""
        phrases = phrases or [self.s.ack_text, ASR_APOLOGY]
        for provider in self.p.tts:
            try:
                for text in phrases:
                    if (provider.name, text) not in _PHRASE_CACHE:
                        _PHRASE_CACHE[(provider.name, text)] = [c async for c in provider.synthesize(text)]
                return
            except Exception as e:  # noqa: BLE001
                log.warning("phrase cache: %s failed: %s", provider.name, e)

    async def _tts_consume(self, ctx: TurnContext, queue: asyncio.Queue) -> bool:
        """Speak segments as they arrive. Returns True if the turn degraded to text-only.

        A single segment that fails on every provider is dropped (its text was already streamed);
        only two consecutive failures switch the rest of the turn to text-only, so one unlucky
        sentence doesn't silence an otherwise healthy reply.
        """
        seq = [0]
        text_only = False
        consecutive_failures = 0
        while True:
            seg = await queue.get()
            if seg is None:
                break
            if text_only:
                continue
            if await self._speak_segment(ctx, seg, seq):
                consecutive_failures = 0
                continue
            consecutive_failures += 1
            if consecutive_failures < 2:
                ctx.record.degraded.append("tts_segment_dropped")
                await self._emit_error(ctx, "tts", "tts_segment_dropped", "TTS failed for one sentence; skipped it")
                continue
            text_only = True
            ctx.record.degraded.append("tts_text_only")
            await self._emit_error(
                ctx, "tts", "tts_unavailable", "All TTS providers failed; continuing as text-only", recoverable=True
            )
        ctx.sw.mark("tts_done")
        # nothing was ever spoken for a non-empty reply -> the client must render it as text
        if (
            not text_only
            and ctx.sw.get("first_audio_sent") is None
            and ctx.record.degraded.count("tts_segment_dropped")
        ):
            text_only = True
            ctx.record.degraded.append("tts_text_only")
        return text_only

    # ------------------------------------------------------------------ turn
    def _save_audio(self, turn_id: str, pcm: bytes) -> None:
        if not self.s.record_audio_dir:
            return
        os.makedirs(self.s.record_audio_dir, exist_ok=True)
        with wave.open(os.path.join(self.s.record_audio_dir, f"{turn_id}.wav"), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self.s.sample_rate)
            w.writeframes(pcm)

    async def run_turn(
        self, ctx: TurnContext, *, pcm: bytes | None = None, text: str | None = None, faults: dict | None = None
    ) -> TurnRecord:
        sw = ctx.sw
        sw.mark("eou")
        rec = TurnRecord(
            turn_id=ctx.turn_id,
            session_id=ctx.session_id,
            label=ctx.label,
            input_mode="audio" if pcm is not None else "text",
            faults=faults or {},
        )
        ctx.record = rec
        status = "ok"
        text_only = False
        queue: asyncio.Queue = asyncio.Queue()
        tasks: list[asyncio.Task] = []
        try:
            # ---- ASR
            if pcm is not None:
                rec.audio_ms = round(duration_ms(pcm, self.s.sample_rate), 1)
                self._save_audio(ctx.turn_id, pcm)
                try:
                    transcript = await self.transcribe(ctx, pcm)
                except AllProvidersFailed:
                    status = "failed"
                    rec.degraded.append("asr_unavailable")
                    sw.mark("asr_done")
                    msg = ASR_APOLOGY
                    await ctx.emit(Ack(turn_id=ctx.turn_id, text=msg, reason="asr_unavailable"))
                    await queue.put(_Segment(msg, is_ack=True))
                    await queue.put(None)
                    await self._tts_consume(ctx, queue)
                    return rec
            else:
                transcript = (text or "").strip()
                rec.providers["asr"] = "text"
            sw.mark("asr_done")
            rec.transcript = transcript
            await ctx.emit(TranscriptFinal(turn_id=ctx.turn_id, text=transcript, provider=rec.providers.get("asr", "")))
            if not transcript:
                status = "ok"
                return rec

            # ---- LLM + TTS, concurrently
            messages = [
                {"role": "system", "content": system_prompt(self.s.persona)},
                *ctx.history[-12:],
                {"role": "user", "content": transcript},
            ]
            first_token = asyncio.Event()

            async def ack_watchdog() -> None:
                try:
                    await asyncio.wait_for(first_token.wait(), timeout=self.s.ack_after_s)
                except TimeoutError:
                    rec.degraded.append("ack_spoken")
                    sw.mark("ack_emitted")
                    await ctx.emit(Ack(turn_id=ctx.turn_id, text=self.s.ack_text, reason="llm_slow"))
                    await queue.put(_Segment(self.s.ack_text, is_ack=True))

            sw.mark("llm_start")
            tts_task = asyncio.create_task(self._tts_consume(ctx, queue))
            ack_task = asyncio.create_task(ack_watchdog())
            tasks = [tts_task, ack_task]
            try:
                response = await self._llm_produce(ctx, messages, queue, first_token)
            except AllProvidersFailed:
                status = "failed"
                response = "Sorry, I can't answer right now. Please try again shortly."
                await queue.put(_Segment(response))
            finally:
                ack_task.cancel()
                await queue.put(None)
            rec.response = response
            text_only = await tts_task
            if status == "ok" and rec.degraded:
                status = "degraded"
            ctx.history.extend([{"role": "user", "content": transcript}, {"role": "assistant", "content": response}])
            return rec
        except asyncio.CancelledError:
            status = "cancelled"
            raise
        finally:
            for t in tasks:
                if not t.done():
                    t.cancel()
            sw.mark("turn_done")
            rec.status = status
            rec.text_only = text_only or "tts_text_only" in rec.degraded
            rec.budget = compute_budget(sw)
            if sw.get("ack_audio_sent") is not None:
                rec.budget["ack_audio_ms"] = sw.span("eou", "ack_audio_sent")
            if self.store:
                try:
                    self.store.save(rec)
                except Exception as e:  # pragma: no cover
                    log.error("metrics save failed: %s", e)
            if status != "cancelled":
                try:
                    await ctx.emit(
                        Metrics(turn_id=ctx.turn_id, budget=rec.budget, providers=rec.providers, degraded=rec.degraded)
                    )
                    await ctx.emit(TurnDone(turn_id=ctx.turn_id, status=status, text_only=rec.text_only))
                except Exception:  # socket may be gone
                    pass
