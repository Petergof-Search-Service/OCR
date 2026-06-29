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
OCR_MAX_CONCURRENT = _get_int("OCR_MAX_CONCURRENT", "5")
# Страниц на один запрос к OCR (ограничение Yandex: ≤200 стр и ≤10 МБ на запрос).
OCR_BATCH_SIZE = _get_int("OCR_BATCH_SIZE", "10")
OCR_CHUNK_SIZE = _get_int("OCR_CHUNK_SIZE", "50")
OCR_DPI = _get_int("OCR_DPI", "100")
OCR_JPEG_QUALITY = _get_int("OCR_JPEG_QUALITY", "80")
# Раздельные квоты Yandex OCR: submit (recognizeTextAsync) ≈10 rps,
# поллинг (getRecognition) ≈50 rps. Поднимаешь квоту через ТП — растишь OCR_SUBMIT_RPS.
OCR_SUBMIT_RPS = _get_float("OCR_SUBMIT_RPS", "8")
OCR_POLL_RPS = _get_float("OCR_POLL_RPS", "40")
# Потолок размера одного запроса (байт PDF). Лимит Yandex — 10 МБ, держим запас.
OCR_MAX_BATCH_BYTES = _get_int("OCR_MAX_BATCH_BYTES", "9000000", minimum=100000)
# Сколько потерянных страниц допустимо, прежде чем файл уйдёт в failed (0 = строго).
OCR_MAX_FAILED_PAGES = _get_int("OCR_MAX_FAILED_PAGES", "0", minimum=0)

API_BASE_URL = os.getenv("API_BASE_URL", "")
CLOUD_FUNCTION_API_KEY = os.getenv("CLOUD_FUNCTION_API_KEY", "")
