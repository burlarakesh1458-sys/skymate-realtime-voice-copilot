"""Per-turn latency budget and its persistence (SQLite + JSONL)."""

from __future__ import annotations

import contextlib
import json
import os
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass, field

BUDGET_KEYS = [
    "asr_ms",  # end-of-utterance -> final transcript
    "llm_ttft_ms",  # LLM request -> first content token
    "first_segment_ms",  # first token -> first speakable segment (chunker wait)
    "tts_ttfb_ms",  # first segment -> first audio byte from TTS
    "overhead_ms",  # everything else on the critical path (queueing, encoding, event loop)
    "e2e_ms",  # end-of-utterance -> first tts.audio frame sent (server-side time-to-first-audio)
]
EXTRA_KEYS = ["llm_total_ms", "tts_total_ms", "turn_total_ms", "client_e2e_ms", "network_ms"]


class Stopwatch:
    """Monotonic milestone recorder for one turn."""

    def __init__(self) -> None:
        self.t0 = time.perf_counter()
        self.marks: dict[str, float] = {}

    def mark(self, name: str, *, once: bool = True) -> float:
        if once and name in self.marks:
            return self.marks[name]
        self.marks[name] = (time.perf_counter() - self.t0) * 1000.0
        return self.marks[name]

    def get(self, name: str) -> float | None:
        return self.marks.get(name)

    def span(self, a: str, b: str) -> float | None:
        if a in self.marks and b in self.marks:
            return round(self.marks[b] - self.marks[a], 1)
        return None


@dataclass
class TurnRecord:
    turn_id: str
    session_id: str
    label: str
    ts: float = field(default_factory=time.time)
    input_mode: str = "audio"
    audio_ms: float | None = None
    transcript: str = ""
    response: str = ""
    status: str = "ok"
    text_only: bool = False
    degraded: list[str] = field(default_factory=list)
    providers: dict[str, str] = field(default_factory=dict)
    budget: dict[str, float | None] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    faults: dict = field(default_factory=dict)


def compute_budget(sw: Stopwatch) -> dict[str, float | None]:
    """Decompose the critical path. Milestones (ms since end-of-utterance):

    eou=0 -> asr_done -> llm_start -> llm_first_token -> first_segment -> tts_first_byte -> first_audio_sent
    """
    b: dict[str, float | None] = {
        "asr_ms": sw.span("eou", "asr_done"),
        "llm_ttft_ms": sw.span("llm_start", "llm_first_token"),
        "first_segment_ms": sw.span("llm_first_token", "first_segment"),
        "tts_ttfb_ms": sw.span("first_segment", "tts_first_byte"),
        "e2e_ms": sw.span("eou", "first_audio_sent"),
        "llm_total_ms": sw.span("llm_start", "llm_done"),
        "tts_total_ms": sw.span("first_segment", "tts_done"),
        "turn_total_ms": sw.span("eou", "turn_done"),
    }
    if b["e2e_ms"] is not None:
        parts = [b[k] or 0.0 for k in ("asr_ms", "llm_ttft_ms", "first_segment_ms", "tts_ttfb_ms")]
        b["overhead_ms"] = round(max(0.0, b["e2e_ms"] - sum(parts)), 1)
    else:
        b["overhead_ms"] = None
    return b


class MetricsStore:
    def __init__(self, db_path: str, jsonl_path: str):
        self.db_path = db_path
        self.jsonl_path = jsonl_path
        self._lock = threading.Lock()
        self._pending_client: dict[str, float] = {}  # client metrics that arrived before the turn was saved
        for p in (db_path, jsonl_path):
            if p:
                os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)

    @contextlib.contextmanager
    def _conn(self):
        """Short-lived connection: schema ensured, committed and closed on exit (thread-safe use)."""
        c = sqlite3.connect(self.db_path)
        try:
            c.execute(
                """CREATE TABLE IF NOT EXISTS turns (
                    turn_id TEXT PRIMARY KEY, session_id TEXT, label TEXT, ts REAL,
                    status TEXT, e2e_ms REAL, client_e2e_ms REAL, record_json TEXT)"""
            )
            yield c
            c.commit()
        finally:
            c.close()

    @staticmethod
    def _apply_client(budget: dict, client_e2e_ms: float) -> None:
        budget["client_e2e_ms"] = round(client_e2e_ms, 1)
        if budget.get("e2e_ms") is not None:
            budget["network_ms"] = round(max(0.0, client_e2e_ms - budget["e2e_ms"]), 1)

    def save(self, rec: TurnRecord) -> None:
        with self._lock:
            if rec.turn_id in self._pending_client:
                self._apply_client(rec.budget, self._pending_client.pop(rec.turn_id))
            d = asdict(rec)
            if self.jsonl_path:
                with open(self.jsonl_path, "a") as f:
                    f.write(json.dumps(d) + "\n")
            if self.db_path:
                with self._conn() as c:
                    c.execute(
                        "INSERT OR REPLACE INTO turns VALUES (?,?,?,?,?,?,?,?)",
                        (
                            rec.turn_id,
                            rec.session_id,
                            rec.label,
                            rec.ts,
                            rec.status,
                            rec.budget.get("e2e_ms"),
                            rec.budget.get("client_e2e_ms"),
                            json.dumps(d),
                        ),
                    )

    def update_client_e2e(self, turn_id: str, client_e2e_ms: float) -> None:
        """Attach client-measured e2e (adds network + client decode) to a stored turn."""
        if not self.db_path:
            return
        with self._lock, self._conn() as c:
            row = c.execute("SELECT record_json FROM turns WHERE turn_id=?", (turn_id,)).fetchone()
            if not row:
                # first audio is sent long before the turn finishes and is persisted
                self._pending_client[turn_id] = client_e2e_ms
                return
            d = json.loads(row[0])
            self._apply_client(d["budget"], client_e2e_ms)
            c.execute(
                "UPDATE turns SET client_e2e_ms=?, record_json=? WHERE turn_id=?",
                (client_e2e_ms, json.dumps(d), turn_id),
            )

    def recent(self, limit: int = 50, label: str | None = None) -> list[dict]:
        if not self.db_path or not os.path.exists(self.db_path):
            return []
        q = "SELECT record_json FROM turns"
        args: tuple = ()
        if label:
            q += " WHERE label=?"
            args = (label,)
        q += " ORDER BY ts DESC LIMIT ?"
        with self._conn() as c:
            rows = c.execute(q, (*args, limit)).fetchall()
        return [json.loads(r[0]) for r in rows]
