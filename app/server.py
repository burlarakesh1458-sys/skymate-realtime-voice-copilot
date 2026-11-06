"""FastAPI app: WebSocket voice sessions + health/metrics endpoints + the static web client."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from .audio import EnergyVAD, b64_to_pcm16, duration_ms, resample_pcm16, rms
from .config import Settings
from .metrics import MetricsStore
from .pipeline import BreakerRegistry, TurnContext, VoicePipeline
from .protocol import (
    AudioChunk,
    ClientMetrics,
    EndOfUtterance,
    ErrorEvent,
    Interrupt,
    ServerEndOfUtterance,
    SessionEnd,
    SessionStart,
    SessionStarted,
    SpeechStarted,
    TextInput,
    TranscriptPartial,
    TurnDone,
    parse_client_event,
)
from .providers.registry import ProviderSet

log = logging.getLogger("voice.server")
WEB_DIR = Path(__file__).resolve().parent.parent / "web"


class Session:
    def __init__(self, ws: WebSocket, app_state, start: SessionStart):
        self.ws = ws
        self.state = app_state
        s: Settings = app_state.settings
        self.id = uuid.uuid4().hex[:12]
        self.label = start.label
        self.client_sr = start.sample_rate
        self.history: list[dict] = []
        self.buf = bytearray()
        self.vad = EnergyVAD(s.vad_energy_threshold, s.vad_silence_ms, s.vad_min_speech_ms, s.sample_rate)
        self.turn_task: asyncio.Task | None = None
        self.partial_task: asyncio.Task | None = None
        self.turn_no = 0
        self.server_vad = start.server_vad
        self.buf_has_speech = False
        self._send_lock = asyncio.Lock()
        settings = s
        if start.persona:
            settings = settings.with_overrides(persona=start.persona)
        if start.chunk_mode:
            settings = settings.with_overrides(chunk_mode=start.chunk_mode)
        self.faults = s.faults
        if start.faults and s.allow_client_faults:
            self.faults = s.faults.merged(start.faults)
        providers = app_state.providers.with_faults(self.faults)
        self.pipeline = VoicePipeline(settings, providers, app_state.breakers, app_state.store)
        self.settings = settings

    async def send(self, event) -> None:
        async with self._send_lock:
            await self.ws.send_text(event.model_dump_json())

    # ----------------------------------------------------------- turns
    def _busy(self) -> bool:
        return self.turn_task is not None and not self.turn_task.done()

    async def cancel_turn(self, reason: str) -> bool:
        if not self._busy():
            return False
        self.turn_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await self.turn_task
        log.info("turn cancelled (%s)", reason)
        return True

    async def start_turn(self, *, pcm: bytes | None = None, text: str | None = None, source: str = "client"):
        await self.cancel_turn("superseded")
        self.turn_no += 1
        turn_id = f"{self.id}-{self.turn_no:03d}"
        ctx = TurnContext(turn_id=turn_id, session_id=self.id, label=self.label, emit=self.send, history=self.history)
        if pcm is not None:
            await self.send(ServerEndOfUtterance(turn_id=turn_id, source=source, audio_ms=round(duration_ms(pcm), 1)))

        async def run():
            try:
                await self.pipeline.run_turn(
                    ctx,
                    pcm=pcm,
                    text=text,
                    faults={k: v for k, v in vars(self.faults).items() if v} if self.faults.any() else None,
                )
            except asyncio.CancelledError:
                with contextlib.suppress(Exception):
                    await self.send(TurnDone(turn_id=turn_id, status="cancelled"))
                raise
            except Exception as e:  # noqa: BLE001 - never kill the socket for one bad turn
                log.exception("turn failed")
                with contextlib.suppress(Exception):
                    await self.send(
                        ErrorEvent(code="turn_failed", message=str(e)[:300], turn_id=turn_id, recoverable=True)
                    )
                    await self.send(TurnDone(turn_id=turn_id, status="failed"))

        self.turn_task = asyncio.create_task(run())

    async def end_of_utterance(self, source: str) -> None:
        if self.partial_task:
            self.partial_task.cancel()
            self.partial_task = None
        pcm = bytes(self.buf)
        had_speech = self.buf_has_speech or rms(pcm) >= self.settings.vad_energy_threshold
        self.buf.clear()
        self.buf_has_speech = False
        self.vad.reset()
        if duration_ms(pcm, self.settings.sample_rate) < self.settings.vad_min_speech_ms or not had_speech:
            # too short / silent: ignore. Prevents Whisper hallucinating "Thank you." on silence
            # and a stray client EoU superseding a turn the server VAD already started.
            return
        await self.start_turn(pcm=pcm, source=source)

    async def _partials_loop(self) -> None:
        asr = self.pipeline.p.asr[0]
        while True:
            await asyncio.sleep(self.settings.asr_partial_interval_ms / 1000)
            if len(self.buf) < self.settings.sample_rate:  # < 0.5 s of audio
                continue
            with contextlib.suppress(Exception):
                text = await asyncio.wait_for(asr.transcribe(bytes(self.buf), self.settings.sample_rate), 3.0)
                if text:
                    await self.send(TranscriptPartial(text=text))

    async def on_audio(self, ev: AudioChunk) -> None:
        pcm = b64_to_pcm16(ev.data)
        if self.client_sr != self.settings.sample_rate:
            pcm = resample_pcm16(pcm, self.client_sr, self.settings.sample_rate)
        self.buf.extend(pcm)
        res = self.vad.process(pcm)
        if res.speech_started or self.vad.in_speech:
            self.buf_has_speech = True
        if res.speech_started:
            await self.send(SpeechStarted())
            if self._busy():  # server-side barge-in: the user started talking over the reply
                await self.cancel_turn("barge_in")
            if self.settings.asr_partials and self.partial_task is None:
                self.partial_task = asyncio.create_task(self._partials_loop())
        if res.end_of_utterance and self.server_vad:
            await self.end_of_utterance("server_vad")
        elif duration_ms(bytes(self.buf)) / 1000 > self.settings.max_utterance_s:
            await self.end_of_utterance("max_length")

    async def close(self) -> None:
        if self.partial_task:
            self.partial_task.cancel()
        await self.cancel_turn("session closed")


def create_app(
    settings: Settings | None = None, providers: ProviderSet | None = None, warmup: bool | None = None
) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        http = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=5.0),
            limits=httpx.Limits(max_keepalive_connections=20, keepalive_expiry=60),
        )
        app.state.http = http
        app.state.settings = settings
        app.state.providers = providers or ProviderSet.from_settings(settings, http)
        app.state.breakers = BreakerRegistry(settings.breaker_failure_threshold, settings.breaker_cooldown_s)
        app.state.store = MetricsStore(settings.metrics_db, settings.metrics_jsonl)
        app.state.warm = {}
        do_warm = settings.warmup_local if warmup is None else warmup

        async def warm_all():
            for chain in (app.state.providers.asr, app.state.providers.tts):
                for p in chain:
                    if p.name.startswith(("faster_whisper", "piper")):
                        t = time.perf_counter()
                        try:
                            await p.warmup()
                            app.state.warm[p.name] = round((time.perf_counter() - t) * 1000)
                        except Exception as e:  # noqa: BLE001
                            app.state.warm[p.name] = f"failed: {e}"

            t = time.perf_counter()
            await VoicePipeline(settings, app.state.providers, app.state.breakers).prime_phrase_cache()
            app.state.warm["phrase_cache"] = round((time.perf_counter() - t) * 1000)

        app.state.warm_task = asyncio.create_task(warm_all()) if do_warm else None
        yield
        if app.state.warm_task:
            app.state.warm_task.cancel()
        await http.aclose()

    app = FastAPI(title="Realtime Voice Assistant", version="1.0.0", lifespan=lifespan)

    @app.get("/")
    async def index():
        return FileResponse(WEB_DIR / "index.html")

    if WEB_DIR.exists():
        app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

    @app.get("/health")
    async def health():
        st = app.state
        warm_done = st.warm_task is None or st.warm_task.done()
        return {
            "status": "ok" if warm_done else "warming",
            "providers": st.providers.names(),
            "breakers": st.breakers.snapshot(),
            "warmup_ms": st.warm,
            "faults": {k: v for k, v in vars(st.settings.faults).items() if v},
        }

    @app.get("/api/metrics")
    async def metrics(limit: int = 50, label: str | None = None):
        return JSONResponse(app.state.store.recent(limit, label))

    @app.get("/api/config")
    async def config():
        return app.state.settings.public_dict()

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        await ws.accept()
        session: Session | None = None
        try:
            while True:
                raw = await ws.receive_text()
                try:
                    ev = parse_client_event(raw)
                except ValidationError as e:
                    await ws.send_text(
                        ErrorEvent(code="bad_event", message=str(e.errors()[:1])[:300]).model_dump_json()
                    )
                    continue
                if session is None:
                    start = ev if isinstance(ev, SessionStart) else SessionStart()
                    session = Session(ws, app.state, start)
                    await session.send(
                        SessionStarted(
                            session_id=session.id,
                            providers=session.pipeline.p.names(),
                            faults={k: v for k, v in vars(session.faults).items() if v},
                        )
                    )
                    if isinstance(ev, SessionStart):
                        continue
                if isinstance(ev, SessionEnd):
                    break
                try:
                    if isinstance(ev, AudioChunk):
                        await session.on_audio(ev)
                    elif isinstance(ev, EndOfUtterance):
                        await session.end_of_utterance("client")
                    elif isinstance(ev, TextInput):
                        await session.start_turn(text=ev.text)
                    elif isinstance(ev, Interrupt):
                        await session.cancel_turn(ev.reason)
                    elif isinstance(ev, ClientMetrics):
                        app.state.store.update_client_e2e(ev.turn_id, ev.client_e2e_ms)
                except (WebSocketDisconnect, RuntimeError):
                    raise
                except Exception as e:  # noqa: BLE001 — one bad event must not drop the session
                    log.exception("event handling failed")
                    await session.send(ErrorEvent(code="internal", message=str(e)[:300]))
        except WebSocketDisconnect:
            pass
        finally:
            if session:
                await session.close()
            with contextlib.suppress(Exception):
                await ws.close()

    return app


app = create_app()  # `uvicorn app.server:app`
