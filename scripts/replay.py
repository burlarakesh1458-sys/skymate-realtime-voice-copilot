"""Replay mode: feed recorded audio files through the pipeline for debugging and regression.

Two modes:
  * ``ws``     — stream each WAV over the real WebSocket protocol to a running server, in
                 (optionally) real time, exactly like a browser would. Measures client-side e2e.
  * ``inproc`` — call the pipeline directly (no server). Useful under a debugger.

Inputs: a directory of 16-bit WAVs (``data/utterances`` from gen_utterances.py, or the
``RECORD_AUDIO_DIR`` the server writes in production). If a ``manifest.json`` with reference
text exists, ASR word-error-rate is computed too.

    python -m scripts.replay --mode ws --url ws://127.0.0.1:8000/ws --label baseline --gap 3
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import glob
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.audio import read_wav_16k  # noqa: E402

CHUNK_MS = 100


_ONES = (
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
    "seventeen eighteen nineteen".split()
)
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def num_words(n: int) -> str:
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else " " + _ONES[n % 10])
    if n < 1000:
        return _ONES[n // 100] + " hundred" + ("" if n % 100 == 0 else " " + num_words(n % 100))
    if n < 1_000_000:
        return num_words(n // 1000) + " thousand" + ("" if n % 1000 == 0 else " " + num_words(n % 1000))
    return str(n)


def normalize(s: str) -> list[str]:
    """Lowercase, spell out integers and %, drop punctuation (so '40%' == 'forty percent')."""
    s = s.lower().replace("%", " percent ")
    s = re.sub(r"(?<=\d),(?=\d{3})", "", s)
    s = re.sub(r"\d+", lambda m: f" {num_words(int(m.group()))} ", s)
    return re.sub(r"[^a-z' ]+", " ", s).replace("-", " ").split()


def wer(ref: str, hyp: str) -> float:
    norm = normalize
    r, h = norm(ref), norm(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev = cur
    return d[len(h)] / max(1, len(r))


def load_inputs(directory: str, limit: int | None) -> list[dict]:
    manifest_path = os.path.join(directory, "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            items = json.load(f)
    else:
        items = [
            {"file": os.path.basename(p), "text": None} for p in sorted(glob.glob(os.path.join(directory, "*.wav")))
        ]
    for it in items:
        it["path"] = os.path.join(directory, it["file"])
    return items[:limit] if limit else items


async def replay_ws(args, items) -> list[dict]:
    import websockets

    results = []
    async with websockets.connect(args.url, max_size=None) as ws:
        await ws.send(
            json.dumps(
                {
                    "type": "session.start",
                    "label": args.label,
                    "sample_rate": 16000,
                    "server_vad": bool(args.server_vad),
                    "faults": json.loads(args.faults) if args.faults else None,
                }
            )
        )
        started = json.loads(await ws.recv())
        print("session:", started.get("session_id"), started.get("providers"))
        for n, it in enumerate(items, 1):
            pcm = read_wav_16k(it["path"])
            step = 16000 * 2 * CHUNK_MS // 1000
            for off in range(0, len(pcm), step):
                await ws.send(
                    json.dumps(
                        {
                            "type": "audio.chunk",
                            "seq": off // step,
                            "data": base64.b64encode(pcm[off : off + step]).decode(),
                        }
                    )
                )
                if args.realtime:
                    await asyncio.sleep(CHUNK_MS / 1000)
            t_eou = time.perf_counter()
            if not args.server_vad:
                await ws.send(json.dumps({"type": "vad.end_of_utterance", "client_ts": time.time() * 1000}))
            rec = {"file": it["file"], "ref": it.get("text"), "events": {}, "audio_frames": 0}
            first_audio = None
            while True:
                ev = json.loads(await asyncio.wait_for(ws.recv(), timeout=90))
                t = ev["type"]
                rec["events"][t] = rec["events"].get(t, 0) + 1
                if t == "vad.end_of_utterance" and args.server_vad:
                    t_eou = time.perf_counter()  # server decided; measure from its notice
                elif t == "transcript.final":
                    rec["transcript"] = ev["text"]
                    rec["turn_id"] = ev["turn_id"]
                elif t == "llm.done":
                    rec["response"] = ev["text"]
                elif t == "tts.audio":
                    rec["audio_frames"] += 1
                    if first_audio is None:
                        first_audio = (time.perf_counter() - t_eou) * 1000
                        await ws.send(
                            json.dumps(
                                {"type": "metrics.client", "turn_id": ev["turn_id"], "client_e2e_ms": first_audio}
                            )
                        )
                elif t == "ack":
                    rec.setdefault("acks", []).append(ev["text"])
                elif t == "error":
                    rec.setdefault("errors", []).append(f'{ev.get("stage")}:{ev["code"]}')
                elif t == "metrics":
                    rec["budget"], rec["providers"], rec["degraded"] = ev["budget"], ev["providers"], ev["degraded"]
                elif t == "turn.done":
                    rec["status"], rec["text_only"] = ev["status"], ev["text_only"]
                    rec["turn_id"] = ev["turn_id"]
                    break
            rec["client_e2e_ms"] = round(first_audio, 1) if first_audio else None
            if rec.get("ref") and "transcript" in rec:
                rec["wer"] = round(wer(rec["ref"], rec["transcript"]), 3)
            results.append(rec)
            b = rec.get("budget", {})
            print(
                f"[{n:02d}] {it['file']} status={rec.get('status')} e2e={b.get('e2e_ms')} "
                f"client={rec['client_e2e_ms']} "
                f"asr={b.get('asr_ms')} ttft={b.get('llm_ttft_ms')} tts={b.get('tts_ttfb_ms')} "
                f"wer={rec.get('wer')} | {rec.get('transcript', '')[:60]!r}"
            )
            if n < len(items):
                await asyncio.sleep(args.gap)
        await ws.send(json.dumps({"type": "session.end"}))
    return results


async def replay_inproc(args, items) -> list[dict]:
    import httpx

    from app.config import Settings
    from app.metrics import MetricsStore
    from app.pipeline import TurnContext, VoicePipeline
    from app.providers.registry import ProviderSet

    s = Settings.from_env()
    results = []
    async with httpx.AsyncClient(timeout=30) as http:
        providers = ProviderSet.from_settings(s, http).with_faults(
            s.faults.merged(json.loads(args.faults) if args.faults else None)
        )
        for chain in (providers.asr, providers.tts):
            for p in chain:
                if p.name.startswith(("faster_whisper", "piper")):
                    await p.warmup()
        pipe = VoicePipeline(s, providers, store=MetricsStore(s.metrics_db, s.metrics_jsonl))
        history: list[dict] = []
        for n, it in enumerate(items, 1):
            events: list = []

            async def emit(e, events=events):
                events.append(e)

            ctx = TurnContext(
                turn_id=f"{args.label}-{n:03d}", session_id=args.label, label=args.label, emit=emit, history=history
            )
            rec_ = await pipe.run_turn(ctx, pcm=read_wav_16k(it["path"]))
            rec = {
                "file": it["file"],
                "ref": it.get("text"),
                "transcript": rec_.transcript,
                "response": rec_.response,
                "status": rec_.status,
                "budget": rec_.budget,
                "providers": rec_.providers,
                "degraded": rec_.degraded,
                "errors": rec_.errors,
            }
            if rec["ref"]:
                rec["wer"] = round(wer(rec["ref"], rec["transcript"]), 3)
            results.append(rec)
            print(f"[{n:02d}] {it['file']} {rec['status']} e2e={rec_.budget.get('e2e_ms')} {rec_.transcript[:60]!r}")
            if n < len(items):
                await asyncio.sleep(args.gap)
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["ws", "inproc"], default="ws")
    ap.add_argument("--dir", default="data/utterances")
    ap.add_argument("--url", default="ws://127.0.0.1:8000/ws")
    ap.add_argument("--label", default="replay")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--gap", type=float, default=2.5, help="seconds between turns (provider rate limits)")
    ap.add_argument("--realtime", action="store_true", help="stream audio at 1x speed like a live mic")
    ap.add_argument("--server-vad", action="store_true", help="let the server's VAD endpoint (no explicit EoU)")
    ap.add_argument(
        "--faults", help="JSON fault overrides, e.g. '{\"asr_down\": true}' (server needs ALLOW_CLIENT_FAULTS)"
    )
    ap.add_argument("--out", help="write per-turn results JSON here")
    args = ap.parse_args()

    items = load_inputs(args.dir, args.limit)
    results = asyncio.run(replay_ws(args, items) if args.mode == "ws" else replay_inproc(args, items))
    out = args.out or f"reports/replay_{args.label}.json"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    ok = sum(r.get("status") in ("ok", "degraded") for r in results)
    wers = [r["wer"] for r in results if "wer" in r]
    print(
        f"\n{ok}/{len(results)} turns completed; mean WER={sum(wers) / len(wers):.3f}"
        if wers
        else f"\n{ok}/{len(results)} ok"
    )
    print("results ->", out)


if __name__ == "__main__":
    main()
