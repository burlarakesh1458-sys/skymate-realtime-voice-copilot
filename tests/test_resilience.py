import asyncio

import pytest

from app.resilience import (
    BreakerState,
    CircuitBreaker,
    CircuitOpenError,
    ProviderError,
    backoff_delay,
    retry_async,
)


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


async def _fail(retryable=True):
    raise ProviderError("p", "boom", retryable=retryable)


async def _ok():
    return "ok"


async def test_breaker_opens_after_threshold_and_fails_fast():
    clk = FakeClock()
    b = CircuitBreaker("p", failure_threshold=3, cooldown_s=10, clock=clk)
    for _ in range(3):
        with pytest.raises(ProviderError):
            await b.call(_fail)
    assert b.state == BreakerState.OPEN
    with pytest.raises(CircuitOpenError):
        await b.call(_ok)


async def test_breaker_half_open_probe_success_closes():
    clk = FakeClock()
    b = CircuitBreaker("p", failure_threshold=1, cooldown_s=5, clock=clk)
    with pytest.raises(ProviderError):
        await b.call(_fail)
    clk.t = 5.1
    assert await b.call(_ok) == "ok"
    assert b.state == BreakerState.CLOSED and b.failures == 0


async def test_breaker_half_open_probe_failure_reopens():
    clk = FakeClock()
    b = CircuitBreaker("p", failure_threshold=2, cooldown_s=5, clock=clk)
    for _ in range(2):
        with pytest.raises(ProviderError):
            await b.call(_fail)
    clk.t = 6
    with pytest.raises(ProviderError):
        await b.call(_fail)  # the probe
    assert b.state == BreakerState.OPEN
    assert b.retry_in() == pytest.approx(5)


async def test_permanent_error_opens_immediately():
    b = CircuitBreaker("p", failure_threshold=5, cooldown_s=5)
    with pytest.raises(ProviderError):
        await b.call(lambda: _fail(retryable=False))
    assert b.state == BreakerState.OPEN


def test_half_open_allows_single_probe():
    clk = FakeClock()
    b = CircuitBreaker("p", failure_threshold=1, cooldown_s=1, clock=clk)
    b.record_failure()
    clk.t = 2
    assert b.allow() is True
    assert b.allow() is False  # second concurrent caller is rejected


async def test_retry_recovers_from_transient():
    calls = {"n": 0}

    async def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ProviderError("p", "503")
        return "ok"

    async def nosleep(_):
        return None

    assert await retry_async(flaky, attempts=2, base_delay=0.1, max_delay=1, sleep=nosleep) == "ok"
    assert calls["n"] == 3


async def test_retry_skips_permanent_and_timeouts():
    calls = {"n": 0}

    async def perm():
        calls["n"] += 1
        raise ProviderError("p", "401", retryable=False)

    with pytest.raises(ProviderError):
        await retry_async(perm, attempts=3, base_delay=0, max_delay=0)
    assert calls["n"] == 1

    async def slow():
        calls["n"] += 1
        raise TimeoutError()

    calls["n"] = 0
    with pytest.raises(asyncio.TimeoutError):
        await retry_async(slow, attempts=3, base_delay=0, max_delay=0)
    assert calls["n"] == 1


def test_backoff_bounded():
    for attempt in range(10):
        assert 0 <= backoff_delay(attempt, 0.25, 2.0) <= 2.0


async def test_retry_honours_server_hint_and_fails_over_when_too_long():
    slept = []

    async def rec_sleep(d):
        slept.append(d)

    calls = {"n": 0}

    async def limited():
        calls["n"] += 1
        if calls["n"] == 1:
            raise ProviderError("p", "429", retry_after_s=1.5)
        return "ok"

    assert await retry_async(limited, attempts=2, base_delay=0.1, max_delay=2.0, sleep=rec_sleep) == "ok"
    assert slept == [1.5]  # waited for the hint, not the (shorter) jittered backoff

    async def long_limit():
        raise ProviderError("p", "429", retry_after_s=30)

    slept.clear()
    with pytest.raises(ProviderError):
        await retry_async(long_limit, attempts=2, base_delay=0.1, max_delay=2.0, sleep=rec_sleep)
    assert slept == []  # hint exceeds budget -> fail over immediately


def test_parse_retry_after():
    from app.providers.groq import parse_retry_after

    assert parse_retry_after({"retry-after": "3"}, "") == 3.0
    assert parse_retry_after(None, "Rate limit reached ... Please try again in 2s. Need more") == 2.0
    assert parse_retry_after(None, "Please try again in 1m7.5s") == 67.5
    assert parse_retry_after(None, "Please try again in 450ms") == 0.45
    assert parse_retry_after(None, "nothing here") is None
