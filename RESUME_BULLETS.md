# Resume bullets: Real-time Voice Copilot (SkyMate)

Every number below was measured in this repo (2 vCPU, no GPU, live Groq API). The sources are in `reports/`.

**Realtime Voice AI Copilot for Drone Ops** · Python, FastAPI, WebSockets, asyncio, Groq (Whisper, gpt-oss), Piper, faster-whisper, SQLite, Docker

- Built a streaming voice assistant (browser mic → WebSocket → Whisper ASR → streaming LLM → sentence-chunked TTS) with a typed 19-event protocol and swappable provider chains. It delivered **1.0 s p50 / 1.5 s p95 time-to-first-audio** over 24 replayed turns (24/24 completed, 1.2% ASR WER) on a 2-vCPU CPU-only box.
- Instrumented a per-turn latency budget (ASR, LLM TTFT, chunking, TTS TTFB, overhead, network) to SQLite/JSONL, with live stacked-bar charts and p50/p95 reports. The budget showed CPU-bound TTS was **64% of p50 latency** under load. An interleaved A/B of 4 chunking strategies then cut **TTS time-to-first-byte p50 from 560 ms to 361 ms**.
- Engineered resilience with per-stage timeouts, Retry-After-aware backoff, circuit breakers and fallback chains (cloud → local ASR/TTS → canned reply). Across **9 fault-injection scenarios (26 turns) against live APIs, no turn hung silently**: a pre-rendered apology played **379 ms** after speech when all ASR was down, and "One moment." played at **~1.8 s** on LLM stalls.
- Shipped a replay-based regression harness, **45 pytest tests** and a **7/7-passing live WebSocket client test**, including barge-in that cancels an in-flight reply in **105 ms**. Packaged with Docker, a Makefile and pinned dependencies.
