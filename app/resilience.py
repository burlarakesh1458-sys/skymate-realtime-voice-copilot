"""Resilience primitives: error taxonomy, retry with exponential backoff + jitter, circuit breaker."""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import Awaitable, Callable
from enum import Enum
from typing import TypeVar

log = logging.getLogger("voice.resilience")
T = TypeVar("T")


class ProviderError(Exception):
    """A provider call failed.

    ``retryable`` distinguishes transient faults (timeouts, 429, 5xx) from permanent ones
    (bad auth, model requires terms acceptance) that should skip straight to fallback.
    """

    def __init__(self, provider: str, message: str, *, retryable: bool = True, code: str = "provider_error"):
        super().__init__(f"[{provider}] {message}")
        self.provider = provider
        self.retryable = retryable
        self.code = code


class CircuitOpenError(ProviderError):
    def __init__(self, provider: str, retry_in_s: float):
        super().__init__(provider, f"circuit open, retry in {retry_in_s:.1f}s", retryable=False, code="circuit_open")


class BreakerState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Classic three-state breaker.

    * CLOSED: calls flow; consecutive failures are counted.
    * OPEN: calls fail fast with ``CircuitOpenError`` until ``cooldown_s`` elapses.
    * HALF_OPEN: exactly one probe call is allowed; success closes, failure re-opens.

    A *permanent* error (e.g. HTTP 401/403, terms not accepted) opens the breaker immediately
    so we don't pay a doomed round-trip on every turn.
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 3,
        cooldown_s: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.cooldown_s = cooldown_s
        self._clock = clock
        self.state = BreakerState.CLOSED
        self.failures = 0
        self.opened_at = 0.0
        self._probe_in_flight = False
        self.transitions: list[tuple[float, str]] = []

    def _set(self, state: BreakerState) -> None:
        if state != self.state:
            log.warning("breaker %s: %s -> %s", self.name, self.state.value, state.value)
            self.transitions.append((self._clock(), state.value))
        self.state = state

    def allow(self) -> bool:
        if self.state == BreakerState.CLOSED:
            return True
        if self.state == BreakerState.OPEN:
            if self._clock() - self.opened_at >= self.cooldown_s:
                self._set(BreakerState.HALF_OPEN)
            else:
                return False
        # HALF_OPEN: single probe
        if self._probe_in_flight:
            return False
        self._probe_in_flight = True
        return True

    def retry_in(self) -> float:
        return max(0.0, self.cooldown_s - (self._clock() - self.opened_at))

    def record_success(self) -> None:
        self.failures = 0
        self._probe_in_flight = False
        self._set(BreakerState.CLOSED)

    def record_failure(self, permanent: bool = False) -> None:
        self._probe_in_flight = False
        self.failures += 1
        if permanent or self.state == BreakerState.HALF_OPEN or self.failures >= self.failure_threshold:
            self.opened_at = self._clock()
            self._set(BreakerState.OPEN)

    async def call(self, fn: Callable[[], Awaitable[T]]) -> T:
        if not self.allow():
            raise CircuitOpenError(self.name, self.retry_in())
        try:
            result = await fn()
        except asyncio.CancelledError:
            self._probe_in_flight = False
            raise
        except ProviderError as e:
            self.record_failure(permanent=not e.retryable)
            raise
        except Exception:
            self.record_failure()
            raise
        self.record_success()
        return result

    def snapshot(self) -> dict:
        return {"state": self.state.value, "failures": self.failures, "retry_in_s": round(self.retry_in(), 1)}


def backoff_delay(attempt: int, base: float, cap: float, rng: random.Random | None = None) -> float:
    """Full-jitter exponential backoff (AWS architecture blog)."""
    r = rng or random
    return r.uniform(0, min(cap, base * (2**attempt)))


async def retry_async(
    fn: Callable[[], Awaitable[T]],
    *,
    attempts: int,
    base_delay: float,
    max_delay: float,
    on_retry: Callable[[int, Exception], None] | None = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    retry_on_timeout: bool = False,
) -> T:
    """Call ``fn``; retry *fast* transient failures (retryable ``ProviderError``) with backoff.

    Timeouts are not retried by default: a timeout has already spent the stage's latency budget,
    so in a real-time system it is better to fail over to the next provider immediately.
    """
    last: Exception | None = None
    for attempt in range(attempts + 1):
        try:
            return await fn()
        except (TimeoutError, ProviderError) as e:
            last = e
            retryable = retry_on_timeout if isinstance(e, asyncio.TimeoutError) else e.retryable
            if not retryable or attempt == attempts:
                raise
            if on_retry:
                on_retry(attempt + 1, e)
            await sleep(backoff_delay(attempt, base_delay, max_delay))
    assert last is not None  # pragma: no cover
    raise last  # pragma: no cover
