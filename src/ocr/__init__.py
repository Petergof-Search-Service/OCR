"""Пакет OCR-пайплайна (распознавание PDF через Yandex Vision OCR)."""
from .pipeline import YandexOCRAsync
from .ratelimit import OCRAuthError

__all__ = ["YandexOCRAsync", "OCRAuthError"]
