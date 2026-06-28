import asyncio
import json
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
    OCR_MAX_CONCURRENT,
    OCR_STRICT_MEMORY_MODE,
    RESULT_PREFIX,
    TMP_PREFIX,
    YANDEX_API_KEY,
)


def _patch_status(key: str, status: str, error: str | None = None) -> None:
    if not API_BASE_URL or not CLOUD_FUNCTION_API_KEY:
        print("_patch_status: API_BASE_URL or CLOUD_FUNCTION_API_KEY is not configured, skipping")
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
            print(
                f"_patch_status: {url} status={status} -> "
                f"HTTP {resp.status_code} body={resp.text[:300]}"
            )
        else:
            print(f"_patch_status: {url} status={status} -> HTTP {resp.status_code}")
    except Exception as e:
        print(f"_patch_status error: {e}")


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
    print("Handler started")
    print(json.dumps(event, ensure_ascii=False))

    message = event["messages"][0]
    details = message["details"]
    key = details["object_id"]

    if INPUT_PREFIX and not key.startswith(INPUT_PREFIX):
        print("Skip non-input object:", key)
        return {"statusCode": 200}

    storage_root = Path(BUCKET_MOUNT_POINT)
    if not storage_root.exists():
        raise FileNotFoundError(
            f"Mounted bucket path does not exist: {storage_root}"
        )

    input_path = _resolve_input_path(storage_root, key)
    print(f"Resolved input path: {input_path}")
    print(
        "OCR config:",
        json.dumps(
            {
                "strict_memory_mode": OCR_STRICT_MEMORY_MODE,
                "max_concurrent": OCR_MAX_CONCURRENT,
                "batch_size": OCR_BATCH_SIZE,
                "chunk_size": OCR_CHUNK_SIZE,
                "dpi": OCR_DPI,
                "jpeg_quality": OCR_JPEG_QUALITY,
            },
            ensure_ascii=False,
        ),
    )

    from OCR_async import YandexOCRAsync

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
    )

    _patch_status(key, "ocr_processing")
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
        print(f"OCR failed: {e}")
        _patch_status(key, "failed", str(e))
        raise

    _patch_status(key, "ocr_done")
    return {"statusCode": 200}
