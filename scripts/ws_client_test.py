"""Automated end-to-end WebSocket client test against a *running* server with real providers.

Checks the protocol contract, server VAD endpointing, barge-in cancellation latency, error
isolation, and (if the server allows client fault overrides) an ASR fallback.

    python -m scripts.ws_client_test --url ws://127.0.0.1:8000/ws
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.audio import read_wav_16k  # noqa: E402
from scripts.replay import wer  # noqa: E402

RESULTS: list[dict] = []


def check(name: str, ok: bool, **info) -> None:
    RESULTS.append({"test": name, "pass": bool(ok), **info})
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {json.dumps(info) if info else ''}")


async def session(url: str, **start):
    import websockets

    ws = await websockets.connect(url, max_size=None)
    await ws.send(json.dumps({"type": "session.start", "label": "ws-client-test", **start}))
    started = json.loads(await ws.recv())
    return ws, started


async def until_done(ws, timeout=60, on_event=None) -> list[dict]:
    events = []
    while True:
        ev = json.loads(await asyncio.wait_for(ws.recv(), timeout=timeout))
        ev["_t"] = time.perf_counter()
        events.append(ev)
        if on_event:
            await on_event(ev)
        if ev["type"] == "turn.done":
            return events


async def send_wav(ws, path: str, eou: bool) -> None:
    pcm = read_wav_16k(path)
    step = 3200
    for off in range(0, len(pcm), step):
        await ws.send(json.dumps({"type": "audio.chunk", "data": base64.b64encode(pcm[off : off + step]).decode()}))
    if eou:
        await ws.send(json.dumps({"type": "vad.end_of_utterance"}))


async def main(url: str, utter_dir: str, gap: float) -> int:
    import httpx

    http_url = url.replace("ws://", "http://").replace("/ws", "")
    # trust_env=False: never route localhost through an HTTP(S)_PROXY from the environment
    # (in the sandbox the proxy answered {"status":"ok"} itself and produced a false pass).
    h = httpx.get(f"{http_url}/health", timeout=5, trust_env=False).json()
    check("health", h.get("status") == "ok" and bool(h.get("providers", {}).get("asr")), providers=h.get("providers"))

    # 1. text turn: protocol contract + ordering
    ws, started = await session(url)
    check("session.started", started.get("type") == "session.started" and "session_id" in started)
    t0 = time.perf_counter()
    await ws.send(json.dumps({"type": "text.input", "text": "Give me two quick pre-flight checks."}))
    evs = await until_done(ws)
    types = [e["type"] for e in evs]
    first = {t: types.index(t) for t in set(types)}
    audio = [e for e in evs if e["type"] == "tts.audio"]
    pcm_ok = all(len(base64.b64decode(a["data"])) % 2 == 0 and a["sample_rate"] > 0 for a in audio)
    ttfa = round((audio[0]["_t"] - t0) * 1000) if audio else None
    check(
        "text_turn_contract",
        {"transcript.final", "llm.token", "llm.done", "tts.audio", "metrics", "turn.done"} <= set(types)
        and first["llm.token"] < first["tts.audio"] < first["turn.done"]
        and pcm_ok
        and evs[-1]["status"] in ("ok", "degraded"),
        audio_frames=len(audio),
        client_ttfa_ms=ttfa,
        status=evs[-1]["status"],
    )
    await asyncio.sleep(gap)

    # 2. bad event is rejected without dropping the session
    await ws.send(json.dumps({"type": "definitely.not.valid"}))
    err = json.loads(await ws.recv())
    await ws.send(json.dumps({"type": "text.input", "text": "Still there?"}))
    evs = await until_done(ws)
    check("bad_event_isolated", err.get("code") == "bad_event" and evs[-1]["status"] in ("ok", "degraded"))
    await asyncio.sleep(gap)

    # 3. barge-in: interrupt on the first audio frame of a long answer
    cancel = {}

    async def on_event(ev):
        if ev["type"] == "tts.audio" and "sent" not in cancel:
            cancel["sent"] = time.perf_counter()
            await ws.send(json.dumps({"type": "interrupt", "reason": "barge_in"}))

    await ws.send(
        json.dumps({"type": "text.input", "text": "Explain in detail every step of a full pre-flight inspection."})
    )
    evs = await until_done(ws, on_event=on_event)
    done = evs[-1]
    cancel_ms = round((done["_t"] - cancel["sent"]) * 1000, 1) if "sent" in cancel else None
    check("barge_in_cancels", done["status"] == "cancelled", interrupt_to_turn_done_ms=cancel_ms)
    await ws.send(json.dumps({"type": "session.end"}))
    await ws.close()
    await asyncio.sleep(gap)

    # 4. audio turn endpointed by server VAD (no explicit end_of_utterance)
    ws, _ = await session(url, server_vad=True)
    manifest = {m["file"]: m["text"] for m in json.loads(Path(utter_dir, "manifest.json").read_text())}
    await send_wav(ws, os.path.join(utter_dir, "q05.wav"), eou=False)
    evs = await until_done(ws)
    eou = next((e for e in evs if e["type"] == "vad.end_of_utterance"), {})
    tr = next((e["text"] for e in evs if e["type"] == "transcript.final"), "")
    w = wer(manifest["q05.wav"], tr)
    check("server_vad_audio_turn", eou.get("source") == "server_vad" and w <= 0.3, transcript=tr, wer=round(w, 3))
    await ws.close()
    await asyncio.sleep(gap)

    # 5. client fault override (only honoured with ALLOW_CLIENT_FAULTS=true)
    ws, started = await session(url, server_vad=False, faults={"asr_down": True})
    if started.get("faults", {}).get("asr_down"):
        await send_wav(ws, os.path.join(utter_dir, "q09.wav"), eou=True)
        evs = await until_done(ws)
        m = next((e for e in evs if e["type"] == "metrics"), {})
        tr = next((e["text"] for e in evs if e["type"] == "transcript.final"), "")
        check(
            "asr_down_falls_back_to_local",
            m.get("providers", {}).get("asr", "").startswith("faster_whisper") and evs[-1]["status"] == "degraded",
            asr=m.get("providers", {}).get("asr"),
            transcript=tr,
            e2e_ms=m.get("budget", {}).get("e2e_ms"),
        )
    else:
        check("asr_down_falls_back_to_local", True, skipped="server has ALLOW_CLIENT_FAULTS=false")
    await ws.close()

    passed = sum(r["pass"] for r in RESULTS)
    print(f"\n{passed}/{len(RESULTS)} checks passed")
    os.makedirs("reports", exist_ok=True)
    with open("reports/ws_client_test.json", "w") as f:
        json.dump({"url": url, "passed": passed, "total": len(RESULTS), "results": RESULTS}, f, indent=2)
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://127.0.0.1:8000/ws")
    ap.add_argument("--dir", default="data/utterances")
    ap.add_argument("--gap", type=float, default=2.0)
    a = ap.parse_args()
    sys.exit(asyncio.run(main(a.url, a.dir, a.gap)))
