import base64
import json

import numpy as np
from fastapi.testclient import TestClient

from app.providers.mock import MockASR, MockLLM, MockTTS
from app.providers.registry import ProviderSet
from app.server import create_app


def client(settings, llm=None):
    ps = ProviderSet([MockASR()], [llm or MockLLM()], [MockTTS()])
    return TestClient(create_app(settings, providers=ps, warmup=False))


def collect_until_done(ws, limit=500):
    events = []
    for _ in range(limit):
        ev = json.loads(ws.receive_text())
        events.append(ev)
        if ev["type"] == "turn.done":
            return events
    raise AssertionError("no turn.done")


def tone(seconds, amp=0.3):
    t = np.arange(int(16000 * seconds)) / 16000
    return (np.sin(2 * np.pi * 220 * t) * amp * 32767).astype("<i2").tobytes()


def chunks(pcm, ms=100):
    step = 16000 * 2 * ms // 1000
    for off in range(0, len(pcm), step):
        yield {"type": "audio.chunk", "data": base64.b64encode(pcm[off : off + step]).decode()}


def test_health_and_config(settings):
    with client(settings) as c:
        h = c.get("/health").json()
        assert h["providers"]["asr"] == ["mock:asr"]
        cfg = c.get("/api/config").json()
        assert cfg["groq_api_key"] in (True, False)  # never leaks the secret


def test_text_turn_over_websocket(settings):
    with client(settings) as c, c.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "session.start", "label": "ws-test"}))
        assert json.loads(ws.receive_text())["type"] == "session.started"
        ws.send_text(json.dumps({"type": "text.input", "text": "battery?"}))
        events = collect_until_done(ws)
        types = [e["type"] for e in events]
        assert types[0] == "transcript.final"
        assert "llm.token" in types and "tts.audio" in types and "metrics" in types
        assert events[-1]["status"] == "ok"
        assert c.get("/api/metrics?label=ws-test").json()[0]["status"] == "ok"


def test_audio_turn_with_explicit_eou(settings):
    with client(settings) as c, c.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "session.start"}))
        ws.receive_text()
        for ch in chunks(tone(1.0)):
            ws.send_text(json.dumps(ch))
        ws.send_text(json.dumps({"type": "vad.end_of_utterance"}))
        events = collect_until_done(ws)
        types = [e["type"] for e in events]
        assert "vad.speech_start" in types and "vad.end_of_utterance" in types
        eou = next(e for e in events if e["type"] == "vad.end_of_utterance")
        assert eou["source"] == "client" and eou["audio_ms"] >= 900


def test_server_vad_endpoints_on_silence(settings):
    with client(settings) as c, c.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "session.start"}))
        ws.receive_text()
        for ch in chunks(tone(0.8) + b"\x00\x00" * 16000):
            ws.send_text(json.dumps(ch))
        events = collect_until_done(ws)
        eou = next(e for e in events if e["type"] == "vad.end_of_utterance")
        assert eou["source"] == "server_vad"


def test_barge_in_cancels_turn(settings):
    slow = MockLLM(reply=" ".join(["word"] * 200) + ".", token_delay_s=0.02)
    with client(settings, llm=slow) as c, c.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "session.start"}))
        ws.receive_text()
        ws.send_text(json.dumps({"type": "text.input", "text": "long answer please"}))
        while json.loads(ws.receive_text())["type"] != "llm.token":
            pass
        ws.send_text(json.dumps({"type": "interrupt"}))
        done = collect_until_done(ws)[-1]
        assert done["status"] == "cancelled"


def test_bad_event_is_rejected_without_closing(settings):
    with client(settings) as c, c.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "session.start"}))
        ws.receive_text()
        ws.send_text(json.dumps({"type": "bogus"}))
        err = json.loads(ws.receive_text())
        assert err["type"] == "error" and err["code"] == "bad_event"
        ws.send_text(json.dumps({"type": "text.input", "text": "still alive?"}))
        assert collect_until_done(ws)[-1]["status"] == "ok"


def test_serves_web_client(settings):
    with client(settings) as c:
        assert "SkyMate" in c.get("/").text
        js = c.get("/static/app.js")
        assert js.status_code == 200 and "audio.chunk" in js.text
