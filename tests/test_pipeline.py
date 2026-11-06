import asyncio

import pytest

from app.config import FaultConfig
from app.metrics import MetricsStore
from app.pipeline import TurnContext, VoicePipeline
from app.protocol import Ack, ErrorEvent, LLMToken, Metrics, TranscriptFinal, TTSAudio, TurnDone
from app.providers.local import DegradedLLM
from app.providers.mock import MockASR, MockLLM, MockTTS
from app.providers.registry import ProviderSet
from app.resilience import ProviderError

PCM = b"\x01\x00" * 16000  # 1 s


class FailingASR(MockASR):
    def __init__(self, retryable=True, fail_times=99):
        super().__init__(name="fail:asr")
        self.retryable, self.fail_times = retryable, fail_times

    async def transcribe(self, pcm16, sample_rate):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise ProviderError(self.name, "503", retryable=self.retryable)
        return "recovered"


class FailingTTS(MockTTS):
    async def synthesize(self, text):
        raise ProviderError("fail:tts", "down")
        yield  # pragma: no cover


def make(settings, asr, llm, tts, store=True):
    st = MetricsStore(settings.metrics_db, settings.metrics_jsonl) if store else None
    return VoicePipeline(settings, ProviderSet(asr, llm, tts), store=st)


def ctx():
    events = []

    async def emit(e):
        events.append(e)

    return TurnContext(turn_id="t-001", session_id="s", label="test", emit=emit, history=[]), events


async def test_happy_path_event_order_and_budget(settings):
    tts = MockTTS()
    pipe = make(settings, [MockASR()], [MockLLM(token_delay_s=0.001)], [tts])
    c, events = ctx()
    rec = await pipe.run_turn(c, pcm=PCM)
    types = [type(e).__name__ for e in events]
    assert types[0] == "TranscriptFinal"
    assert types.index("LLMToken") < types.index("TTSAudio")
    assert types[-2:] == ["Metrics", "TurnDone"]
    assert rec.status == "ok" and not rec.degraded
    # chunker fed TTS sentence by sentence (two sentences in the mock reply)
    assert len(tts.sentences) == 2
    b = rec.budget
    for k in ("asr_ms", "llm_ttft_ms", "tts_ttfb_ms", "e2e_ms", "overhead_ms"):
        assert b[k] is not None and b[k] >= 0
    assert b["e2e_ms"] >= b["asr_ms"]
    assert pipe.store.recent(1)[0]["turn_id"] == "t-001"
    assert c.history[-1]["role"] == "assistant"


async def test_asr_retry_then_success(settings):
    asr = FailingASR(fail_times=1)
    pipe = make(settings, [asr], [MockLLM()], [MockTTS()])
    c, _ = ctx()
    rec = await pipe.run_turn(c, pcm=PCM)
    assert rec.transcript == "recovered" and asr.calls == 2
    assert any("retry 1" in e for e in rec.errors)


async def test_asr_fallback_to_secondary(settings):
    pipe = make(settings, [FailingASR(), MockASR(text="from fallback", name="mock:local")], [MockLLM()], [MockTTS()])
    c, events = ctx()
    rec = await pipe.run_turn(c, pcm=PCM)
    assert rec.transcript == "from fallback"
    assert "asr_fallback:mock:local" in rec.degraded and rec.status == "degraded"
    assert any(isinstance(e, ErrorEvent) and e.stage == "asr" for e in events)


async def test_all_asr_down_speaks_apology(settings):
    pipe = make(settings, [FailingASR(retryable=False)], [MockLLM()], [MockTTS()])
    c, events = ctx()
    rec = await pipe.run_turn(c, pcm=PCM)
    assert rec.status == "failed"
    assert any(isinstance(e, Ack) and e.reason == "asr_unavailable" for e in events)
    assert any(isinstance(e, TTSAudio) for e in events)  # user still hears something
    assert isinstance(events[-1], TurnDone)


async def test_llm_timeout_acks_then_falls_back(settings):
    hung = MockLLM(ttft_s=10, name="hung:llm")
    pipe = make(settings, [MockASR()], [hung, DegradedLLM()], [MockTTS()])
    c, events = ctx()
    rec = await pipe.run_turn(c, pcm=PCM)
    acks = [e for e in events if isinstance(e, Ack)]
    assert acks and acks[0].reason == "llm_slow"
    assert rec.providers["llm"] == "degraded:canned"
    assert "llm_fallback:degraded:canned" in rec.degraded and "ack_spoken" in rec.degraded
    assert rec.budget["llm_ttft_ms"] is not None
    assert rec.budget["ack_audio_ms"] < rec.budget["e2e_ms"]


async def test_tts_down_degrades_to_text_only(settings):
    pipe = make(settings, [MockASR()], [MockLLM()], [FailingTTS()])
    c, events = ctx()
    rec = await pipe.run_turn(c, pcm=PCM)
    assert rec.text_only and "tts_text_only" in rec.degraded
    assert any(isinstance(e, LLMToken) for e in events)  # text still streamed
    assert not any(isinstance(e, TTSAudio) for e in events)
    assert events[-1].text_only is True


async def test_tts_fallback_chain(settings):
    pipe = make(settings, [MockASR()], [MockLLM()], [FailingTTS(), MockTTS(name="mock:piper")])
    c, _ = ctx()
    rec = await pipe.run_turn(c, pcm=PCM)
    assert rec.providers["tts"] == "mock:piper" and not rec.text_only


async def test_breaker_skips_dead_provider_on_later_turns(settings):
    s = settings.with_overrides(breaker_failure_threshold=1, retry_attempts=0)
    bad = FailingASR()
    pipe = make(s, [bad, MockASR(name="mock:local")], [MockLLM()], [MockTTS()])
    for i in range(3):
        c, _ = ctx()
        c.turn_id = f"t-{i}"
        await pipe.run_turn(c, pcm=PCM)
    assert bad.calls == 1  # breaker open: no more doomed calls
    assert pipe.breakers.get("fail:asr").state.value == "open"


async def test_cancellation_records_cancelled(settings):
    pipe = make(settings, [MockASR()], [MockLLM(token_delay_s=0.2)], [MockTTS()])
    c, _ = ctx()
    task = asyncio.create_task(pipe.run_turn(c, pcm=PCM))
    await asyncio.sleep(0.15)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert pipe.store.recent(1)[0]["status"] == "cancelled"


async def test_fault_injection_wrappers(settings):
    ps = ProviderSet([MockASR(), MockASR(text="local", name="mock:local")], [MockLLM()], [MockTTS()])
    pipe = VoicePipeline(settings, ps.with_faults(FaultConfig(asr_down=True)))
    c, events = ctx()
    rec = await pipe.run_turn(c, pcm=PCM)
    assert rec.transcript == "local"
    assert any(isinstance(e, TranscriptFinal) for e in events)
    assert any(isinstance(e, Metrics) for e in events)


async def test_text_input_skips_asr(settings):
    asr = MockASR()
    pipe = make(settings, [asr], [MockLLM()], [MockTTS()])
    c, _ = ctx()
    rec = await pipe.run_turn(c, text="hello")
    assert asr.calls == 0 and rec.providers["asr"] == "text" and rec.response


class FlakyOnceTTS(MockTTS):
    """Fails every attempt for the first sentence only."""

    async def synthesize(self, text):
        if not self.sentences and text.startswith("Battery"):
            raise ProviderError("flaky:tts", "503")
        async for item in super().synthesize(text):
            yield item


async def test_single_failed_segment_is_dropped_not_text_only(settings):
    tts = FlakyOnceTTS(name="flaky:tts")
    pipe = make(settings, [MockASR()], [MockLLM()], [tts])
    c, events = ctx()
    rec = await pipe.run_turn(c, pcm=PCM)
    assert not rec.text_only
    assert "tts_segment_dropped" in rec.degraded
    assert any(isinstance(e, TTSAudio) for e in events)  # the second sentence was still spoken


async def test_ack_phrase_is_served_from_cache(settings):
    tts = MockTTS(name="mock:cache-tts")
    pipe = make(settings, [MockASR()], [MockLLM(ttft_s=0.4)], [tts])
    await pipe.prime_phrase_cache()
    primed = list(tts.sentences)
    c, events = ctx()
    await pipe.run_turn(c, pcm=PCM)
    assert any(isinstance(e, Ack) for e in events)
    assert tts.sentences.count(settings.ack_text) == primed.count(settings.ack_text)  # not re-synthesized
