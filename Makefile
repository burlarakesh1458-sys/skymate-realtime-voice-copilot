.PHONY: help install models utterances run test lint fmt replay replay-realtime ws-test faults report docker clean

PY ?= python
PORT ?= 8000
PIPER_VOICE_URL = https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/low

help:            ## list targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-16s %s\n", $$1, $$2}'

install:         ## install runtime + dev deps
	$(PY) -m pip install -r requirements-dev.txt

models:          ## download the Piper fallback voice (~60 MB); faster-whisper downloads on first use
	mkdir -p models/piper
	curl -sL -o models/piper/en_US-lessac-low.onnx $(PIPER_VOICE_URL)/en_US-lessac-low.onnx
	curl -sL -o models/piper/en_US-lessac-low.onnx.json $(PIPER_VOICE_URL)/en_US-lessac-low.onnx.json

utterances:      ## synthesize the 24-query regression set into data/utterances
	$(PY) -m scripts.gen_utterances --out data/utterances

run:             ## start the server on :$(PORT) (web client at /)
	uvicorn app.server:app --host 0.0.0.0 --port $(PORT)

test:            ## unit + integration tests (mocked providers, no network)
	$(PY) -m pytest -q

lint:            ## ruff lint + format check
	ruff check . && ruff format --check .

fmt:             ## auto-format
	ruff check --fix . && ruff format .

replay:          ## replay data/utterances through a running server (LABEL=baseline)
	$(PY) -m scripts.replay --url ws://127.0.0.1:$(PORT)/ws --label $(or $(LABEL),baseline) --gap 3

replay-realtime: ## same, streaming audio at 1x speed with server-side VAD endpointing
	$(PY) -m scripts.replay --url ws://127.0.0.1:$(PORT)/ws --label $(or $(LABEL),realtime) --gap 3 --realtime --server-vad

ws-test:         ## automated end-to-end WebSocket client test against a running server
	$(PY) -m scripts.ws_client_test --url ws://127.0.0.1:$(PORT)/ws

faults:          ## run fault-injection scenarios (real providers + local fallbacks)
	$(PY) -m scripts.fault_demo

report:          ## p50/p95 tables + PNG charts from reports/metrics.jsonl
	$(PY) -m scripts.report

docker:          ## build + run with docker compose
	docker compose up --build

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__
