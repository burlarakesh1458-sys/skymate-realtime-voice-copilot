"""Generate a deterministic test set of spoken drone-ops queries (16 kHz mono WAV) with Piper.

There is no microphone in CI/sandbox environments, so regression audio is synthesized offline.
Each file gets 300 ms of leading and 900 ms of trailing silence so server-side VAD can endpoint it.

    python -m scripts.gen_utterances --out data/utterances
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.audio import pcm16_to_wav_bytes, resample_pcm16  # noqa: E402
from app.config import Settings  # noqa: E402

QUERIES = [
    "What should I check before my first flight of the day?",
    "My battery is at forty percent and the wind is twenty five knots. Should I continue the survey?",
    "How long can a five thousand milliamp hour battery keep a one kilogram quadcopter in the air?",
    "What does return to home do if I lose the radio link?",
    "Is it safe to fly in light rain?",
    "What is the maximum altitude allowed for a nano drone in India?",
    "Explain the difference between GPS mode and attitude mode.",
    "My drone is drifting to the left in hover. What could cause that?",
    "How do I calibrate the compass?",
    "Give me a quick checklist for a mapping mission.",
    "What overlap should I use for photogrammetry?",
    "The motors are getting very hot after landing. Is that a problem?",
    "What is a good failsafe setting for a long range flight?",
    "How cold is too cold for lithium polymer batteries?",
    "Can I fly near an airport for a college project?",
    "What does the red zone mean on the Digital Sky map?",
    "How should I store my batteries for a month?",
    "My video feed keeps breaking up. What should I try?",
    "What propeller size suits a five inch racing frame?",
    "How do I estimate flight time for a payload drop mission?",
    "What should the center of gravity be on a fixed wing?",
    "Should I land now if I see a bird of prey circling?",
    "What is the safe distance to keep from people when flying?",
    "Summarize the steps to arm the drone safely.",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/utterances")
    ap.add_argument("--model", default=Settings.from_env().piper_model_path)
    ap.add_argument("--length-scale", type=float, default=1.0, help=">1 = slower speech")
    args = ap.parse_args()

    from piper import PiperVoice, SynthesisConfig

    os.makedirs(args.out, exist_ok=True)
    voice = PiperVoice.load(args.model)
    cfg = SynthesisConfig(length_scale=args.length_scale)
    manifest = []
    for i, text in enumerate(QUERIES, 1):
        chunks = list(voice.synthesize(text, syn_config=cfg))
        sr = chunks[0].sample_rate
        pcm = b"".join(c.audio_int16_bytes for c in chunks)
        pcm = resample_pcm16(pcm, sr, 16000)
        lead, tail = np.zeros(int(0.3 * 16000), "<i2").tobytes(), np.zeros(int(0.9 * 16000), "<i2").tobytes()
        name = f"q{i:02d}.wav"
        with open(os.path.join(args.out, name), "wb") as f:
            f.write(pcm16_to_wav_bytes(lead + pcm + tail, 16000))
        manifest.append({"file": name, "text": text, "speech_s": round(len(pcm) / 32000, 2)})
        print(f"{name}  {len(pcm) / 32000:.2f}s  {text}")
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()
