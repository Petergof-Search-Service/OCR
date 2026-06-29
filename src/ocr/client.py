"""Клиент Yandex Vision OCR (async): отправка и поллинг результата.

Submit (`recognizeTextAsync`) и поллинг (`getRecognition`) имеют РАЗДЕЛЬНЫЕ квоты
RPS, поэтому у клиента два независимых лимитера.
"""
import asyncio
import json
import time
from typing import Any

import aiohttp
from aiohttp import ClientError, ConnectionTimeoutError

from logging_config import get_logger

from .fsutil import encode_pdf_bytes
from .ratelimit import (
    RETRYABLE_STATUSES,
    AsyncRateLimiter,
    OCRAuthError,
    backoff_delay,
    parse_retry_after,
)

logger = get_logger("ocr")


class OcrClient:
    RECOGNIZE_URL = "https://ocr.api.cloud.yandex.net/ocr/v1/recognizeTextAsync"
    RESULT_URL = "https://ocr.api.cloud.yandex.net/ocr/v1/getRecognition?operationId={operation_id}"

    def __init__(
        self,
        headers: dict[str, str],
        submit_limiter: AsyncRateLimiter,
        poll_limiter: AsyncRateLimiter,
    ):
        self.headers = headers
        self.submit_limiter = submit_limiter
        self.poll_limiter = poll_limiter

    async def recognize_pdf(
        self,
        session: aiohttp.ClientSession,
        pdf_bytes: bytes,
        max_retries: int = 12,
        base_delay: float = 1.0,
    ) -> str | None:
        body = {
            "mimeType": "application/pdf",
            "languageCodes": ["*"],
            "model": "page",
            "content": encode_pdf_bytes(pdf_bytes),
        }
        headers = {**self.headers, "Content-Type": "application/json"}
        for attempt in range(1, max_retries + 1):
            retry_after: float | None = None
            try:
                await self.submit_limiter.acquire()
                async with session.post(self.RECOGNIZE_URL, headers=headers, json=body) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("id")
                    response_text = await response.text()
                    if response.status in {401, 403}:
                        # Битый/просроченный ключ — повторять бессмысленно, рвём весь файл.
                        raise OCRAuthError(f"OCR API {response.status}: {response_text[:200]}")
                    if response.status in RETRYABLE_STATUSES:
                        retry_after = parse_retry_after(response.headers)
                        logger.warning(
                            f"recognize attempt {attempt}/{max_retries}: "
                            f"HTTP {response.status} {response_text[:160]}"
                        )
                    else:
                        logger.error(
                            f"Permanent HTTP error {response.status}: {response_text[:200]}"
                        )
                        return None
            except OCRAuthError:
                raise
            except (ConnectionTimeoutError, TimeoutError, ClientError) as exc:
                logger.warning(
                    f"recognize attempt {attempt}/{max_retries}: {type(exc).__name__}: {exc}"
                )
            except Exception as exc:
                logger.warning(
                    f"recognize attempt {attempt}/{max_retries}: {type(exc).__name__}: {exc}"
                )
            if attempt < max_retries:
                delay = (
                    retry_after if retry_after is not None
                    else backoff_delay(attempt, base_delay)
                )
                await asyncio.sleep(delay)
        logger.error("recognize: failed after all retries")
        return None

    async def get_operation_result(
        self,
        session: aiohttp.ClientSession,
        operation_id: str,
        max_wait: float = 600.0,
        poll_interval: float = 3.0,
    ) -> dict[str, Any] | None:
        url = self.RESULT_URL.format(operation_id=operation_id)
        deadline = time.monotonic() + max_wait
        rate_attempt = 0  # для эскалации backoff на 429/503
        while time.monotonic() < deadline:
            delay = poll_interval
            try:
                await self.poll_limiter.acquire()
                async with session.get(url, headers=self.headers) as response:
                    if response.status == 200:
                        text = await response.text()
                        results = []
                        for line in text.strip().split("\n"):
                            if line.strip():
                                try:
                                    results.append(json.loads(line))
                                except json.JSONDecodeError:
                                    continue
                        if not results:
                            await asyncio.sleep(delay)
                            continue
                        if len(results) == 1:
                            return results[0]
                        combined = {"result": {"pages": []}}
                        for result in results:
                            if "result" not in result:
                                continue
                            if "pages" in result["result"]:
                                combined["result"]["pages"].extend(result["result"]["pages"])
                            else:
                                combined["result"]["pages"].append(result["result"])
                        return combined
                    if response.status == 404:
                        # Операция ещё выполняется — обычный поллинг, не rate-limit.
                        rate_attempt = 0
                        await asyncio.sleep(poll_interval)
                        continue
                    if response.status in RETRYABLE_STATUSES:
                        rate_attempt += 1
                        retry_after = parse_retry_after(response.headers)
                        delay = retry_after if retry_after is not None else backoff_delay(
                            rate_attempt, base=poll_interval, cap=30.0
                        )
                        logger.warning(
                            f"poll HTTP {response.status} (backoff {delay:.1f}s): "
                            f"operation {operation_id}"
                        )
                    else:
                        logger.warning(f"Failed to get result: {response.status}")
            except Exception as exc:
                logger.warning(f"Error while polling OCR result: {exc}")
            await asyncio.sleep(delay)
        logger.error(f"Operation timed out after {max_wait:.0f}s: operation {operation_id}")
        return None
