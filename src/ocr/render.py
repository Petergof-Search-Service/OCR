"""Рендер страниц батча в image-PDF для отправки в OCR.

Байт-aware: один батч может разбиться на несколько под-батчей так, чтобы каждый
PDF-запрос не превышал лимит Yandex OCR (файл ≤10 МБ). Возвращает список под-батчей.
"""
import gc
import io
from pathlib import Path

import fitz
from PIL import Image

from .fsutil import write_bytes
from .layout import PathLayout

# Тип под-батча: (image_pdf_path, [global_page_numbers], [per_page_image_pdf_paths])
SubBatch = tuple[Path, list[int], list[Path]]


def render_batch(
    layout: PathLayout,
    batch_pdf_path: Path,
    chunk_id: int,
    batch_id: int,
    page_numbers: list[int],
    dpi: int,
    jpeg_quality: int,
    max_batch_bytes: int,
) -> list[SubBatch]:
    """Отрендерить страницы батча. Возвращает 1+ под-батчей, каждый ≤ max_batch_bytes.

    Размер контролируется по сумме JPEG-байт страниц (доминирующая часть PDF).
    """
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    src = fitz.open(batch_pdf_path)

    sub_batches: list[SubBatch] = []
    sub_id = 0
    cur_doc: fitz.Document | None = None
    cur_pages: list[int] = []
    cur_page_paths: list[Path] = []
    cur_bytes = 0

    def _flush() -> None:
        nonlocal cur_doc, cur_pages, cur_page_paths, cur_bytes, sub_id
        if cur_doc is None:
            return
        if cur_pages:
            image_pdf_path = layout.chunk_image_batch_path(chunk_id, batch_id, sub_id)
            batch_bytes = cur_doc.tobytes(garbage=4, deflate=True, clean=True)
            write_bytes(image_pdf_path, batch_bytes)
            sub_batches.append((image_pdf_path, cur_pages, cur_page_paths))
            del batch_bytes
            sub_id += 1
        cur_doc.close()
        cur_doc = None
        cur_pages = []
        cur_page_paths = []
        cur_bytes = 0
        gc.collect()

    try:
        for local_index, page in enumerate(src):
            page_number = page_numbers[local_index]
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            img = None
            try:
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                jpeg_buffer = io.BytesIO()
                try:
                    img.save(jpeg_buffer, format="JPEG", quality=jpeg_quality, optimize=True)
                    jpeg_bytes = jpeg_buffer.getvalue()
                finally:
                    jpeg_buffer.close()
            finally:
                del pix
                if img is not None:
                    del img

            page_rect = page.rect

            # Отдельный одностраничный image-PDF (нужен для текстового слоя).
            single_doc = fitz.open()
            try:
                single_page = single_doc.new_page(
                    width=page_rect.width, height=page_rect.height
                )
                single_page.insert_image(page_rect, stream=jpeg_bytes, keep_proportion=True)
                single_bytes = single_doc.tobytes(garbage=4, deflate=True, clean=True)
            finally:
                single_doc.close()
            single_page_path = layout.chunk_image_page_path(chunk_id, batch_id, page_number)
            write_bytes(single_page_path, single_bytes)

            # Если страница не влезает в текущий под-батч по размеру — закрыть его.
            if cur_pages and (cur_bytes + len(jpeg_bytes)) > max_batch_bytes:
                _flush()

            if cur_doc is None:
                cur_doc = fitz.open()
            batch_page = cur_doc.new_page(width=page_rect.width, height=page_rect.height)
            batch_page.insert_image(page_rect, stream=jpeg_bytes, keep_proportion=True)
            cur_pages.append(page_number)
            cur_page_paths.append(single_page_path)
            cur_bytes += len(jpeg_bytes)

            del jpeg_bytes
            del single_bytes
            gc.collect()

        _flush()
        return sub_batches
    finally:
        if cur_doc is not None:
            cur_doc.close()
        src.close()
        gc.collect()
