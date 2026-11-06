#!/usr/bin/env bash
# Interleaved A/B of TTS chunking modes: for every utterance, run each mode back-to-back in a
# fresh session so shared-CPU load and provider latency drift hit all modes equally.
set -euo pipefail
URL=${URL:-ws://127.0.0.1:8000/ws}
FILES=${FILES:-q01 q02 q03 q04 q05 q06 q07 q08 q09 q10 q11 q12}
MODES=${MODES:-full sentence clause eager}
for f in $FILES; do
  for m in $MODES; do
    python -m scripts.replay --url "$URL" --label "ab-$m" --chunk-mode "$m" --files "$f" --gap 0 --append
    sleep "${GAP:-2}"
  done
done
