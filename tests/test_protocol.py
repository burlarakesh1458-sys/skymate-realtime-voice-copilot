import pytest
from pydantic import ValidationError

from app.protocol import (
    CLIENT_EVENT_TYPES,
    SERVER_EVENT_TYPES,
    AudioChunk,
    EndOfUtterance,
    Metrics,
    SessionStart,
    TTSAudio,
    parse_client_event,
    parse_server_event,
)


def test_parse_each_client_event():
    samples = {
        "session.start": {"type": "session.start", "label": "t"},
        "audio.chunk": {"type": "audio.chunk", "data": "AAA=", "seq": 1},
        "vad.end_of_utterance": {"type": "vad.end_of_utterance"},
        "text.input": {"type": "text.input", "text": "hi"},
        "interrupt": {"type": "interrupt"},
        "metrics.client": {"type": "metrics.client", "turn_id": "a", "client_e2e_ms": 1.0},
        "session.end": {"type": "session.end"},
    }
    assert set(samples) == set(CLIENT_EVENT_TYPES)
    for t, payload in samples.items():
        ev = parse_client_event(payload)
        assert ev.type == t


def test_dispatch_types():
    assert isinstance(parse_client_event('{"type":"audio.chunk","data":"AAA="}'), AudioChunk)
    assert isinstance(parse_client_event({"type": "vad.end_of_utterance"}), EndOfUtterance)
    assert isinstance(parse_client_event({"type": "session.start"}), SessionStart)


@pytest.mark.parametrize(
    "bad",
    [
        {"type": "nope"},
        {"type": "audio.chunk"},  # missing data
        {"type": "text.input", "text": "x", "extra": 1},  # extra fields forbidden
        "not json",
    ],
)
def test_rejects_bad_events(bad):
    with pytest.raises(ValidationError):
        parse_client_event(bad)


def test_server_events_roundtrip():
    ev = TTSAudio(turn_id="t1", data="AAA=", sample_rate=16000, seq=0)
    back = parse_server_event(ev.model_dump_json())
    assert back == ev
    m = Metrics(turn_id="t1", budget={"e2e_ms": 812.5, "asr_ms": None}, providers={"asr": "x"})
    assert parse_server_event(m.model_dump()).budget["e2e_ms"] == 812.5
    assert len(SERVER_EVENT_TYPES) == len(set(SERVER_EVENT_TYPES))
