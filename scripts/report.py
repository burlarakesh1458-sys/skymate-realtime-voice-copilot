"""Offline latency report: p50/p95 tables (Markdown) and PNG charts from recorded turns.

python -m scripts.report                       # all labels in reports/metrics.sqlite
python -m scripts.report --labels baseline     # just one run
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.metrics import BUDGET_KEYS  # noqa: E402

STAGES = ["asr_ms", "llm_ttft_ms", "first_segment_ms", "tts_ttfb_ms", "overhead_ms"]
COLORS = {
    "asr_ms": "#58a6ff",
    "llm_ttft_ms": "#a371f7",
    "first_segment_ms": "#d29922",
    "tts_ttfb_ms": "#3fb950",
    "overhead_ms": "#8b949e",
    "network_ms": "#f0883e",
}
TABLE_KEYS = [*BUDGET_KEYS, "network_ms", "client_e2e_ms", "llm_total_ms", "tts_total_ms", "turn_total_ms"]


def load(db: str, jsonl: str) -> list[dict]:
    if os.path.exists(db):
        with sqlite3.connect(db) as c:
            rows = c.execute("SELECT record_json FROM turns ORDER BY ts").fetchall()
        return [json.loads(r[0]) for r in rows]
    with open(jsonl) as f:
        return [json.loads(line) for line in f if line.strip()]


def pct(vals: list[float], q: float) -> float | None:
    return float(np.percentile(vals, q)) if vals else None


def fmt(v: float | None) -> str:
    return "–" if v is None else f"{v:,.0f}"


def summarize(turns: list[dict]) -> dict[str, dict]:
    out = {}
    for k in TABLE_KEYS:
        vals = [t["budget"][k] for t in turns if t["budget"].get(k) is not None]
        out[k] = {
            "n": len(vals),
            "p50": pct(vals, 50),
            "p95": pct(vals, 95),
            "mean": float(np.mean(vals)) if vals else None,
        }
    return out


def table(label: str, turns: list[dict]) -> str:
    s = summarize(turns)
    ok = [t for t in turns if t["budget"].get("e2e_ms") is not None]
    lines = [
        f"### `{label}` — {len(turns)} turns ({len(ok)} produced audio)",
        "",
        "| stage | n | p50 (ms) | p95 (ms) | mean (ms) | share of e2e (p50-based) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    e2e50 = s["e2e_ms"]["p50"] or 0
    for k in TABLE_KEYS:
        r = s[k]
        if not r["n"]:
            continue
        share = f"{100 * r['p50'] / e2e50:.0f}%" if k in STAGES and e2e50 else ""
        lines.append(f"| {k} | {r['n']} | {fmt(r['p50'])} | {fmt(r['p95'])} | {fmt(r['mean'])} | {share} |")
    statuses = defaultdict(int)
    for t in turns:
        statuses[t["status"]] += 1
    providers = defaultdict(int)
    for t in turns:
        providers[" / ".join(f"{k}={v}" for k, v in sorted(t["providers"].items()))] += 1
    lines += ["", "status: " + ", ".join(f"{k}={v}" for k, v in sorted(statuses.items()))]
    lines += ["providers: " + "; ".join(f"{k} ×{v}" for k, v in providers.items()), ""]
    return "\n".join(lines)


def chart_stacked(label: str, turns: list[dict], path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    turns = [t for t in turns if t["budget"].get("e2e_ms") is not None]
    if not turns:
        return
    fig, ax = plt.subplots(figsize=(max(8, len(turns) * 0.42), 4.2))
    x = np.arange(len(turns))
    bottom = np.zeros(len(turns))
    for k in [*STAGES, "network_ms"]:
        vals = np.array([t["budget"].get(k) or 0 for t in turns])
        ax.bar(x, vals, bottom=bottom, color=COLORS[k], label=k.replace("_ms", ""), width=0.75)
        bottom += vals
    ax.axhline(np.median(bottom), color="#d73a49", ls="--", lw=1, label=f"p50 total {np.median(bottom):,.0f} ms")
    ax.set_xticks(x, [t["turn_id"].split("-")[-1] for t in turns], fontsize=7, rotation=90)
    ax.set_ylabel("ms (end-of-utterance → first audio)")
    ax.set_title(f"Per-turn latency budget — {label}")
    ax.legend(fontsize=7, ncol=4, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def chart_percentiles(groups: dict[str, list[dict]], path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = [*STAGES, "e2e_ms"]
    labels = list(groups)
    fig, ax = plt.subplots(figsize=(10, 4.2))
    w = 0.8 / (2 * len(labels))
    for i, lab in enumerate(labels):
        s = summarize(groups[lab])
        p50 = [s[k]["p50"] or 0 for k in keys]
        p95 = [s[k]["p95"] or 0 for k in keys]
        xs = np.arange(len(keys)) + (2 * i) * w - 0.4 + w / 2
        ax.bar(xs, p50, w, label=f"{lab} p50")
        ax.bar(xs + w, p95, w, label=f"{lab} p95", alpha=0.55)
    ax.set_xticks(np.arange(len(keys)), [k.replace("_ms", "") for k in keys])
    ax.set_ylabel("ms")
    ax.set_title("Latency by stage: p50 vs p95")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="reports/metrics.sqlite")
    ap.add_argument("--jsonl", default="reports/metrics.jsonl")
    ap.add_argument("--labels", nargs="*", help="labels to include (default: all)")
    ap.add_argument("--compare", nargs="*", help="labels to put on the p50/p95 comparison chart")
    ap.add_argument("--out", default="reports/LATENCY_REPORT.md")
    args = ap.parse_args()

    turns = load(args.db, args.jsonl)
    groups: dict[str, list[dict]] = defaultdict(list)
    for t in turns:
        if not args.labels or t["label"] in args.labels:
            groups[t["label"]].append(t)
    outdir = os.path.dirname(args.out) or "."
    parts = [
        "# Latency report",
        "",
        "Generated by `scripts/report.py` from per-turn records. All values are measured, in milliseconds.",
        "`e2e_ms` = server-side end-of-utterance → first `tts.audio` frame sent; `client_e2e_ms` adds the "
        "WebSocket hop and client handling as measured by the replay client.",
        "",
    ]
    for label, ts in groups.items():
        parts.append(table(label, ts))
        png = os.path.join(outdir, f"latency_stacked_{label}.png")
        chart_stacked(label, ts, png)
        if os.path.exists(png):
            parts += [f"![{label}]({os.path.basename(png)})", ""]
    cmp_labels = [lab for lab in (args.compare or list(groups)) if lab in groups]
    if cmp_labels:
        chart_percentiles({lab: groups[lab] for lab in cmp_labels}, os.path.join(outdir, "latency_p50_p95.png"))
        parts += ["## p50 vs p95 by stage", "", "![p50/p95](latency_p50_p95.png)", ""]
    with open(args.out, "w") as f:
        f.write("\n".join(parts))
    print("\n".join(parts))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
