import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import Settings  # noqa: E402


@pytest.fixture
def settings(tmp_path):
    return Settings().with_overrides(
        metrics_db=str(tmp_path / "m.sqlite"),
        metrics_jsonl=str(tmp_path / "m.jsonl"),
        asr_timeout_s=0.5,
        llm_ttft_timeout_s=0.5,
        llm_total_timeout_s=2.0,
        tts_timeout_s=0.5,
        retry_base_delay_s=0.0,
        retry_max_delay_s=0.0,
        ack_after_s=0.2,
        warmup_local=False,
    )
