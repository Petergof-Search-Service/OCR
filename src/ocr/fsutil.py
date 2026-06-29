"""Мелкие файловые/байтовые помощники для OCR-пайплайна."""
import base64
from pathlib import Path

from logging_config import get_logger

logger = get_logger("ocr")


def encode_pdf_bytes(pdf_bytes: bytes) -> str:
    return base64.b64encode(pdf_bytes).decode("utf-8")


def read_bytes(path: Path) -> bytes:
    return path.read_bytes()


def write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def remove_file(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except Exception as exc:
        logger.warning(f"Failed to delete {path}: {exc}")


def cleanup_dir(root: Path) -> None:
    """Рекурсивно удалить директорию со всем содержимым (best-effort)."""
    if not root.exists():
        return
    for path in sorted(root.rglob("*"), reverse=True):
        try:
            if path.is_file() or path.is_symlink():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        except Exception as exc:
            logger.warning(f"Failed to delete {path}: {exc}")
    try:
        root.rmdir()
    except Exception:
        pass
