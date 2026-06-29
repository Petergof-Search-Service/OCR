"""Нарезка исходного PDF на чанки и батчи."""
from pathlib import Path

from pypdf import PdfReader, PdfWriter

from .layout import PathLayout


def split_pdf_to_chunks(
    layout: PathLayout, input_pdf_path: Path, chunk_size: int = 50
) -> list[tuple[int, Path, list[int]]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    chunks: list[tuple[int, Path, list[int]]] = []
    with input_pdf_path.open("rb") as src:
        reader = PdfReader(src)
        total_pages = len(reader.pages)
        for chunk_id, start in enumerate(range(0, total_pages, chunk_size)):
            end = min(start + chunk_size, total_pages)
            page_numbers = list(range(start, end))
            writer = PdfWriter()
            for page_number in page_numbers:
                writer.add_page(reader.pages[page_number])
            chunk_pdf_path = layout.chunk_pdf_path(chunk_id)
            with chunk_pdf_path.open("wb") as dst:
                writer.write(dst)
            chunks.append((chunk_id, chunk_pdf_path, page_numbers))
    return chunks


def split_chunk_to_batches(
    layout: PathLayout,
    chunk_id: int,
    chunk_pdf_path: Path,
    global_page_numbers: list[int],
    batch_size: int = 1,
) -> list[tuple[Path, list[int], int]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    batches: list[tuple[Path, list[int], int]] = []
    with chunk_pdf_path.open("rb") as src:
        reader = PdfReader(src)
        total_pages = len(reader.pages)
        batch_id = 0
        for i in range(0, total_pages, batch_size):
            writer = PdfWriter()
            local_end = min(i + batch_size, total_pages)
            for local_page_index in range(i, local_end):
                writer.add_page(reader.pages[local_page_index])
            batch_path = layout.chunk_batch_path(chunk_id, batch_id)
            with batch_path.open("wb") as dst:
                writer.write(dst)
            batches.append((batch_path, global_page_numbers[i:local_end], batch_id))
            batch_id += 1
    return batches
