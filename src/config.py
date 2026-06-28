import os

from dotenv import load_dotenv

load_dotenv()


def _get_int(name: str, default: str, minimum: int = 1) -> int:
    value = int(os.getenv(name, default))
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def _get_float(name: str, default: str, minimum: float = 0.0) -> float:
    value = float(os.getenv(name, default))
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def _get_bool(name: str, default: str = "1") -> bool:
    # 0 - ничгео не меняет, 1 и другие варианты ставят лимит на обработку по 1 странице
    # P. S. просто нужно для дебага
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes"}


FOLDER_ID = os.getenv("FOLDER_ID")
YANDEX_API_KEY = os.getenv("YANDEX_API_KEY")
BUCKET_MOUNT_POINT = os.getenv("BUCKET_MOUNT_POINT", "/function/storage/petergof-testing-backet")
INPUT_PREFIX = os.getenv("INPUT_PREFIX", "incoming/")
TMP_PREFIX = os.getenv("TMP_PREFIX", "tmp")
RESULT_PREFIX = os.getenv("RESULT_PREFIX", "result")
OCR_STRICT_MEMORY_MODE = _get_bool("OCR_STRICT_MEMORY_MODE", "0")
OCR_MAX_CONCURRENT = _get_int("OCR_MAX_CONCURRENT", "10")
OCR_BATCH_SIZE = _get_int("OCR_BATCH_SIZE", "2")
OCR_CHUNK_SIZE = _get_int("OCR_CHUNK_SIZE", "20")
OCR_DPI = _get_int("OCR_DPI", "100")
OCR_JPEG_QUALITY = _get_int("OCR_JPEG_QUALITY", "80")
# Лимит RPS к OCR API (отправка + поллинг) — защита от 429.
OCR_MAX_RPS = _get_float("OCR_MAX_RPS", "8")
# Сколько потерянных страниц допустимо, прежде чем файл уйдёт в failed (0 = строго).
OCR_MAX_FAILED_PAGES = _get_int("OCR_MAX_FAILED_PAGES", "0", minimum=0)

API_BASE_URL = os.getenv("API_BASE_URL", "")
CLOUD_FUNCTION_API_KEY = os.getenv("CLOUD_FUNCTION_API_KEY", "")
