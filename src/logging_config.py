"""Структурированное логирование для OCR-функции.

Главная цель — чтобы имя входного файла (object key) и request_id присутствовали в
**каждой** строке лога. Тогда в Yandex Cloud Logging можно отфильтроваться по имени
файла и получить весь трейс конкретного запуска (включая логи из пакета `ocr`).

Ключ и request_id хранятся в `contextvars`, поэтому автоматически подхватываются
во всех корутинах/тасках, порождённых в рамках обработки одного файла — пробрасывать
их по сигнатурам функций не нужно.
"""
import contextvars
import logging
import os
import sys

object_key_var: contextvars.ContextVar[str] = contextvars.ContextVar("object_key", default="-")
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")

_configured = False


class _ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.object_key = object_key_var.get()
        record.request_id = request_id_var.get()
        return True


def setup_logging(level: str | None = None) -> None:
    """Настроить root-логгер (идемпотентно — безопасно на тёплых стартах функции)."""
    global _configured
    if _configured:
        return
    log_level = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(_ContextFilter())
    handler.setFormatter(
        logging.Formatter(
            "%(levelname)s key=%(object_key)s req=%(request_id)s %(name)s: %(message)s"
        )
    )
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(log_level)
    _configured = True


def bind_context(object_key: str | None = None, request_id: str | None = None) -> None:
    """Привязать ключ файла и request_id к текущему контексту логирования."""
    if object_key is not None:
        object_key_var.set(object_key)
    if request_id is not None:
        request_id_var.set(request_id)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
