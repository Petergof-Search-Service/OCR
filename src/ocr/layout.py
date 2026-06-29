"""Раскладка путей во временной/результирующей структуре для одного файла."""
from pathlib import Path


class PathLayout:
    """Все пути tmp/result для одного входного файла (по storage_root + base_name).

    Чистый помощник без побочных эффектов, кроме `ensure_dirs()` и `chunk_dir()`,
    которые создают директории.
    """

    def __init__(self, storage_root: Path, base_name: str, tmp_prefix: str, result_prefix: str):
        self.storage_root = Path(storage_root)
        self.base_name = base_name
        self.tmp_dir = self.storage_root / tmp_prefix / base_name
        self.chunks_dir = self.tmp_dir / "chunks"
        self.result_dir = self.storage_root / result_prefix
        self.result_txt_dir = self.result_dir / "txt-files"
        self.result_json_dir = self.result_dir / "json-files"
        self.result_pdf_dir = self.result_dir / "pdf-files"

    def ensure_dirs(self) -> None:
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        self.result_txt_dir.mkdir(parents=True, exist_ok=True)
        self.result_json_dir.mkdir(parents=True, exist_ok=True)
        self.result_pdf_dir.mkdir(parents=True, exist_ok=True)

    def chunk_dir(self, chunk_id: int) -> Path:
        path = self.chunks_dir / f"chunk_{chunk_id:05d}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def chunk_pdf_path(self, chunk_id: int) -> Path:
        return self.chunk_dir(chunk_id) / "source.pdf"

    def chunk_batch_path(self, chunk_id: int, batch_id: int) -> Path:
        return self.chunk_dir(chunk_id) / f"batch_{batch_id:05d}.pdf"

    def chunk_image_batch_path(self, chunk_id: int, batch_id: int, sub_id: int = 0) -> Path:
        return self.chunk_dir(chunk_id) / f"batch_{batch_id:05d}.sub_{sub_id:03d}.image.pdf"

    def chunk_image_page_path(self, chunk_id: int, batch_id: int, page_number: int) -> Path:
        return self.chunk_dir(chunk_id) / f"batch_{batch_id:05d}.page_{page_number:06d}.image.pdf"

    def chunk_page_txt_path(self, chunk_id: int, page_number: int) -> Path:
        return self.chunk_dir(chunk_id) / f"page_{page_number:06d}.txt"

    def chunk_page_json_path(self, chunk_id: int, page_number: int) -> Path:
        return self.chunk_dir(chunk_id) / f"page_{page_number:06d}.json"

    def chunk_overlay_pdf_path(self, chunk_id: int, page_number: int) -> Path:
        return self.chunk_dir(chunk_id) / f"page_{page_number:06d}.pdf"

    def chunk_txt_result_path(self, chunk_id: int) -> Path:
        return self.chunk_dir(chunk_id) / f"chunk_{chunk_id:05d}.txt"

    def chunk_jsonl_result_path(self, chunk_id: int) -> Path:
        return self.chunk_dir(chunk_id) / f"chunk_{chunk_id:05d}.jsonl"

    def chunk_pdf_result_path(self, chunk_id: int) -> Path:
        return self.chunk_dir(chunk_id) / f"chunk_{chunk_id:05d}.pdf"

    def txt_result_path(self) -> Path:
        return self.result_txt_dir / f"{self.base_name}.txt"

    def json_result_path(self) -> Path:
        return self.result_json_dir / f"{self.base_name}.json"

    def json_tmp_path(self) -> Path:
        return self.tmp_dir / f"{self.base_name}.json"

    def pdf_result_path(self) -> Path:
        return self.result_pdf_dir / f"{self.base_name}.pdf"
