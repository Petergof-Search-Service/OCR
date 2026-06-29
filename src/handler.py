import asyncio
import json
import time
from pathlib import Path

import requests

from config import (
    API_BASE_URL,
    BUCKET_MOUNT_POINT,
    CLOUD_FUNCTION_API_KEY,
    FOLDER_ID,
    INPUT_PREFIX,
    OCR_BATCH_SIZE,
    OCR_CHUNK_SIZE,
    OCR_DPI,
    OCR_JPEG_QUALITY,
    OCR_MAX_BATCH_BYTES,
    OCR_MAX_CONCURRENT,
    OCR_MAX_FAILED_PAGES,
    OCR_POLL_RPS,
    OCR_STRICT_MEMORY_MODE,
    OCR_SUBMIT_RPS,
    RESULT_PREFIX,
    TMP_PREFIX,
    YANDEX_API_KEY,
)
from logging_config import bind_context, get_logger, setup_logging

logger = get_logger("handler")


def _patch_status(key: str, status: str, error: str | None = None) -> None:
    if not API_BASE_URL or not CLOUD_FUNCTION_API_KEY:
        logger.warning(
            "Status callback skipped: API_BASE_URL/CLOUD_FUNCTION_API_KEY not configured"
        )
        return
    url = f"{API_BASE_URL}/files/by-key/status"
    try:
        resp = requests.patch(
            url,
            json={"system_key": key, "status": status, "error_message": error},
            headers={"x-service-key": CLOUD_FUNCTION_API_KEY},
            timeout=10,
        )
        if resp.status_code >= 400:
            logger.error(
                "Status callback failed: status=%s -> HTTP %s body=%s",
                status,
                resp.status_code,
                resp.text[:300],
            )
        else:
            logger.info("Status callback ok: status=%s -> HTTP %s", status, resp.status_code)
    except Exception as e:
        logger.error("Status callback error: status=%s: %s", status, e)


def _resolve_input_path(storage_root: Path, object_key: str) -> Path:
    candidate = storage_root / object_key
    if candidate.exists():
        return candidate

    normalized_key = object_key.lstrip("/")
    normalized_prefix = INPUT_PREFIX.strip("/")

    if normalized_prefix and normalized_key.startswith(normalized_prefix + "/"):
        stripped = normalized_key[len(normalized_prefix) + 1 :]
        candidate = storage_root / stripped
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Input file not found in mounted bucket. "
        f"Expected path: {storage_root / object_key}"
    )


def handler(event, context):
    setup_logging()
    request_id = getattr(context, "request_id", None)
    bind_context(request_id=request_id)

    message = event["messages"][0]
    details = message["details"]
    key = details["object_id"]
    bind_context(object_key=key)

    bucket_id = details.get("bucket_id")
    logger.info("Handler started: bucket=%s key=%s request_id=%s", bucket_id, key, request_id)
    logger.debug("Raw event: %s", json.dumps(event, ensure_ascii=False))

    if INPUT_PREFIX and not key.startswith(INPUT_PREFIX):
        logger.info("Skip non-input object (prefix %r): %s", INPUT_PREFIX, key)
        return {"statusCode": 200}

    storage_root = Path(BUCKET_MOUNT_POINT)
    if not storage_root.exists():
        logger.error("Mounted bucket path does not exist: %s", storage_root)
        raise FileNotFoundError(f"Mounted bucket path does not exist: {storage_root}")

    input_path = _resolve_input_path(storage_root, key)
    try:
        size_bytes = input_path.stat().st_size
    except OSError:
        size_bytes = -1
    logger.info("Resolved input path: %s (%s bytes)", input_path, size_bytes)
    logger.info(
        "OCR config: %s",
        json.dumps(
            {
                "strict_memory_mode": OCR_STRICT_MEMORY_MODE,
                "max_concurrent": OCR_MAX_CONCURRENT,
                "batch_size": OCR_BATCH_SIZE,
                "chunk_size": OCR_CHUNK_SIZE,
                "dpi": OCR_DPI,
                "jpeg_quality": OCR_JPEG_QUALITY,
                "submit_rps": OCR_SUBMIT_RPS,
                "poll_rps": OCR_POLL_RPS,
                "max_batch_bytes": OCR_MAX_BATCH_BYTES,
                "max_failed_pages": OCR_MAX_FAILED_PAGES,
            },
            ensure_ascii=False,
        ),
    )

    from ocr import YandexOCRAsync

    ocr = YandexOCRAsync(
        api_key=YANDEX_API_KEY,
        folder_id=FOLDER_ID,
        storage_root=storage_root,
        key=key,
        tmp_prefix=TMP_PREFIX,
        result_prefix=RESULT_PREFIX,
        dpi=OCR_DPI,
        jpeg_quality=OCR_JPEG_QUALITY,
        strict_memory_mode=OCR_STRICT_MEMORY_MODE,
        submit_rps=OCR_SUBMIT_RPS,
        poll_rps=OCR_POLL_RPS,
        max_batch_bytes=OCR_MAX_BATCH_BYTES,
        max_failed_pages=OCR_MAX_FAILED_PAGES,
    )

    _patch_status(key, "ocr_processing")
    started = time.monotonic()
    try:
        asyncio.run(
            ocr.process_pdf(
                input_pdf_path=input_path,
                max_concurrent=OCR_MAX_CONCURRENT,
                batch_size=OCR_BATCH_SIZE,
                chunk_size=OCR_CHUNK_SIZE,
                cleanup_tmp_files=True,
            )
        )
    except Exception as e:
        elapsed = time.monotonic() - started
        logger.exception("OCR failed after %.1fs: %s", elapsed, e)
        _patch_status(key, "failed", str(e))
        raise

    elapsed = time.monotonic() - started
    logger.info("OCR done in %.1fs for key=%s", elapsed, key)
    _patch_status(key, "ocr_done")
    return {"statusCode": 200}
