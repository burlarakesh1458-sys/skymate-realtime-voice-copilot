"""Fault-injection scenarios against the REAL providers (Groq + local fallbacks).

Each scenario builds a fresh pipeline (fresh circuit breakers) with fault flags — the same
``FaultConfig`` the server reads from ``FAULT_*`` env vars — replays a few utterances, and records
what the user would have experienced. Writes reports/fault_results.json + reports/RESILIENCE_REPORT.md.

    python -m scripts.fault_demo [--only asr_down llm_timeout] [--gap 3]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx  # noqa: E402

from app.audio import read_wav_16k  # noqa: E402
from app.config import FaultConfig, Settings  # noqa: E402
from app.metrics import MetricsStore  # noqa: E402
from app.pipeline import TurnContext, VoicePipeline  # noqa: E402
from app.protocol import Ack, ErrorEvent, TTSAudio  # noqa: E402
from app.providers.registry import ProviderSet, build_asr, build_llm, build_tts  # noqa: E402

SCENARIOS = [
    # name, description, chain overrides, faults, files
    ("baseline", "No faults (reference)", {}, FaultConfig(), ["q01", "q04"]),
    (
        "asr_down",
        "Primary ASR (Groq Whisper) returns 503 on every call",
        {},
        FaultConfig(asr_down=True),
        ["q05", "q09", "q11", "q17"],
    ),
    (
        "asr_all_down",
        "Every ASR provider unavailable (no fallback configured)",
        {"asr_chain": ["groq"]},
        FaultConfig(asr_down=True),
        ["q05"],
    ),
    (
        "llm_slow",
        "LLM first token delayed by 2.5 s (under the 4 s TTFT timeout)",
        {},
        FaultConfig(llm_slow_ms=2500),
        ["q06", "q13"],
    ),
    (
        "llm_timeout_to_canned",
        "Primary LLM hangs forever; only fallback is the canned degraded reply",
        {},
        FaultConfig(llm_timeout=True),
        ["q02", "q22"],
    ),
    (
        "llm_timeout_to_secondary",
        "Primary LLM hangs; chain falls over to a second Groq model (gpt-oss-120b)",
        {"llm_chain": ["groq", "groq:openai/gpt-oss-120b", "degraded"]},
        FaultConfig(llm_timeout=True),
        ["q02", "q22"],
    ),
    (
        "orpheus_primary",
        "Real Groq Orpheus TTS as primary (no injected fault), Piper as fallback",
        {"tts_chain": ["groq", "piper"]},
        FaultConfig(),
        ["q10", "q14", "q19"],
    ),
    (
        "tts_all_down",
        "Every TTS provider down -> text-only response",
        {"tts_chain": ["piper"]},
        FaultConfig(tts_down=True),
        ["q15", "q16"],
    ),
    (
        "flaky_30pct",
        "30% random transient failures on every primary provider",
        {},
        FaultConfig(error_rate=0.3),
        ["q03", "q07", "q08", "q12", "q18", "q20", "q21", "q23"],
    ),
]


def build(s: Settings, http, cache: dict) -> ProviderSet:
    def get(kind, name, fn):
        key = (kind, name)
        if name in ("faster_whisper", "piper"):
            if key not in cache:
                cache[key] = fn(name, s, http)
            return cache[key]
        return fn(name, s, http)

    return ProviderSet(
        [get("asr", n, build_asr) for n in s.asr_chain],
        [get("llm", n, build_llm) for n in s.llm_chain],
        [get("tts", n, build_tts) for n in s.tts_chain],
    )


async def run(only: list[str] | None, gap: float, udir: str) -> list[dict]:
    base = Settings.from_env()
    store = MetricsStore(base.metrics_db, base.metrics_jsonl)
    manifest = {m["file"]: m["text"] for m in json.loads(Path(udir, "manifest.json").read_text())}
    cache: dict = {}
    out = []
    async with httpx.AsyncClient(timeout=httpx.Timeout(30, connect=5)) as http:
        # warm local fallbacks once so fallback latency is steady-state, as on a running server
        warm = build(base.with_overrides(asr_chain=["faster_whisper"], tts_chain=["piper"]), http, cache)
        await warm.asr[0].warmup()
        await warm.tts[0].warmup()
        for name, desc, overrides, faults, files in SCENARIOS:
            if only and name not in only:
                continue
            s = base.with_overrides(**{"tts_chain": ["piper"], **overrides, "faults": faults})
            providers = build(s, http, cache).with_faults(faults)
            pipe = VoicePipeline(s, providers, store=store)
            history: list[dict] = []
            print(f"\n=== {name}: {desc}\n    chains={providers.names()} faults={faults}")
            turns = []
            for i, f in enumerate(files, 1):
                events: list = []

                async def emit(e, events=events):
                    events.append((time.perf_counter(), e))

                ctx = TurnContext(
                    turn_id=f"fault-{name}-{i:02d}-{int(time.time())}",
                    session_id=f"fault-{name}",
                    label=f"fault-{name}",
                    emit=emit,
                    history=history,
                )
                t0 = time.perf_counter()
                rec = await pipe.run_turn(ctx, pcm=read_wav_16k(os.path.join(udir, f + ".wav")), faults=vars(faults))
                wall = (time.perf_counter() - t0) * 1000
                first_audio = next((e for _, e in events if isinstance(e, TTSAudio)), None)
                acks = [e.text for _, e in events if isinstance(e, Ack)]
                errs = [f"{e.stage}:{e.code}" for _, e in events if isinstance(e, ErrorEvent)]
                turn = {
                    "file": f,
                    "ref": manifest.get(f + ".wav"),
                    "status": rec.status,
                    "text_only": rec.text_only,
                    "providers": rec.providers,
                    "degraded": rec.degraded,
                    "transcript": rec.transcript,
                    "response": rec.response,
                    "acks_spoken": acks,
                    "first_audio_sentence": first_audio.sentence if first_audio else None,
                    "error_events": errs,
                    "internal_errors": rec.errors,
                    "budget": rec.budget,
                    "wall_ms": round(wall, 1),
                    "breakers": pipe.breakers.snapshot(),
                }
                turns.append(turn)
                b = rec.budget
                print(
                    f"  [{i}] {rec.status:9s} asr={b.get('asr_ms')} ttft={b.get('llm_ttft_ms')} "
                    f"ack_audio={b.get('ack_audio_ms')} e2e={b.get('e2e_ms')} prov={rec.providers} "
                    f"deg={rec.degraded} errs={errs}"
                )
                await asyncio.sleep(gap)
            out.append(
                {"scenario": name, "description": desc, "faults": vars(faults), "overrides": overrides, "turns": turns}
            )
    return out


def write_report(results: list[dict], path: str) -> None:
    lines = [
        "# Resilience report (fault injection, real providers)",
        "",
        "Generated by `scripts/fault_demo.py`. Each scenario uses a fresh pipeline/breakers and real Groq + local "
        "fallback providers; faults are injected with the same `FaultConfig` flags the server reads from `FAULT_*` "
        "env vars. Times in ms, measured from end-of-utterance.",
        "",
        "| scenario | turns | statuses | what the user heard first | first audio p50 (ms) | ack audio (ms) | providers used | degradation flags |",  # noqa: E501
        "|---|---:|---|---|---:|---:|---|---|",
    ]
    import statistics as st

    for r in results:
        t = r["turns"]
        statuses = ", ".join(sorted({x["status"] for x in t}))
        e2e = [x["budget"]["e2e_ms"] for x in t if x["budget"].get("e2e_ms") is not None]
        ack = [x["budget"].get("ack_audio_ms") for x in t if x["budget"].get("ack_audio_ms") is not None]
        heard = t[0]["acks_spoken"][0] if t[0]["acks_spoken"] else (t[0]["first_audio_sentence"] or "(text only)")
        prov = sorted({f"{k}={v}" for x in t for k, v in x["providers"].items()})
        deg = sorted({d for x in t for d in x["degraded"]})
        lines.append(
            f"| `{r['scenario']}` | {len(t)} | {statuses} | {heard[:60]!s} | "
            f"{st.median(e2e):,.0f} | {(f'{st.median(ack):,.0f}' if ack else '–')} | {'<br>'.join(prov)} | "
            f"{', '.join(deg) or '–'} |"
            if e2e
            else f"| `{r['scenario']}` | {len(t)} | {statuses} | {heard[:60]} | – | "
            f"{(f'{st.median(ack):,.0f}' if ack else '–')} | {'<br>'.join(prov)} | {', '.join(deg) or '–'} |"
        )
    lines += ["", "## Per-turn detail", ""]
    for r in results:
        lines += [f"### `{r['scenario']}` — {r['description']}", ""]
        lines += [
            "| # | status | asr | llm ttft | ack audio | e2e | breaker states after turn | errors surfaced to client |"
        ]
        lines += ["|---:|---|---:|---:|---:|---:|---|---|"]
        for i, x in enumerate(r["turns"], 1):
            b = x["budget"]
            br = ", ".join(f"{k.split(':')[0]}:{k.split(':')[-1][:14]}={v['state']}" for k, v in x["breakers"].items())
            f = lambda v: "–" if v is None else f"{v:,.0f}"  # noqa: E731
            lines.append(
                f"| {i} | {x['status']}{' (text-only)' if x['text_only'] else ''} | {f(b.get('asr_ms'))} | "
                f"{f(b.get('llm_ttft_ms'))} | {f(b.get('ack_audio_ms'))} | {f(b.get('e2e_ms'))} | {br} | "
                f"{', '.join(x['error_events']) or '–'} |"
            )
        lines.append("")
    with open(path, "w") as fh:
        fh.write("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--gap", type=float, default=3.0)
    ap.add_argument("--dir", default="data/utterances")
    ap.add_argument("--out", default="reports/fault_results.json")
    args = ap.parse_args()
    results = asyncio.run(run(args.only, args.gap, args.dir))
    prev = []
    if args.only and os.path.exists(args.out):  # merge partial reruns
        prev = [r for r in json.loads(Path(args.out).read_text()) if r["scenario"] not in args.only]
    order = [s[0] for s in SCENARIOS]
    merged = sorted(prev + results, key=lambda r: order.index(r["scenario"]))
    with open(args.out, "w") as f:
        json.dump(merged, f, indent=2)
    write_report(merged, "reports/RESILIENCE_REPORT.md")
    print("\nwrote", args.out, "and reports/RESILIENCE_REPORT.md")


if __name__ == "__main__":
    main()
