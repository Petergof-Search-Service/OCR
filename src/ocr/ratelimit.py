"""Ограничение RPS к OCR API: token-bucket, backoff, Retry-After.

Yandex Vision OCR имеет РАЗДЕЛЬНЫЕ квоты: отправка `recognizeTextAsync` (≈10 rps)
и поллинг `getRecognition` (≈50 rps). Поэтому лимитеры создаются по отдельности
(см. `OcrClient`), а этот модуль — только сам token-bucket и помощники backoff.
"""
import asyncio
import random
import time
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime

# HTTP-статусы OCR API, на которых имеет смысл повторить запрос.
RETRYABLE_STATUSES = {429, 500, 502, 503, 504}


class OCRAuthError(Exception):
    """Ошибка авторизации OCR API (401/403). Прерывает всю обработку файла —
    повторять бессмысленно (битый/просроченный YANDEX_API_KEY)."""


class AsyncRateLimiter:
    """Простой token-bucket для ограничения RPS к OCR API.

    Потокобезопасен в пределах одного event loop за счёт `asyncio.Lock`.
    """

    def __init__(self, rate_per_sec: float, burst: float | None = None):
        self.rate = max(0.1, rate_per_sec)
        self.capacity = burst if burst is not None else max(1.0, rate_per_sec)
        self.tokens = self.capacity
        self.updated = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        while True:
            async with self._lock:
                now = time.monotonic()
                self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.rate)
                self.updated = now
                if self.tokens >= 1:
                    self.tokens -= 1
                    return
                wait = (1 - self.tokens) / self.rate
            await asyncio.sleep(wait)


def parse_retry_after(headers) -> float | None:
    """Вернуть задержку из заголовка Retry-After (секунды или HTTP-дата)."""
    value = headers.get("Retry-After")
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        pass
    try:
        when = parsedate_to_datetime(value)
        delta = (when - datetime.now(UTC)).total_seconds()
        return max(0.0, delta)
    except Exception:
        return None


def backoff_delay(attempt: int, base: float = 1.0, cap: float = 60.0) -> float:
    """Экспоненциальный backoff с джиттером: base*2^(attempt-1), но не больше cap."""
    return min(cap, base * (2 ** (attempt - 1))) + random.uniform(0, 1)
