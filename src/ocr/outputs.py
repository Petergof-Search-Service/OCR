"""Сборка итоговых результатов (txt / json / постраничные chunk-PDF)."""
import json
import shutil

import fitz

from logging_config import get_logger

from .layout import PathLayout
from .pdf_merge import merge_pdfs

logger = get_logger("ocr")


class ResultWriter:
    """Пишет постраничные результаты и собирает итоговые txt/json/pdf по чанкам.

    Держит состояние стримингового финального JSON (`{"data":[ ... ]}`).
    """

    def __init__(self, layout: PathLayout):
        self.layout = layout
        self._final_json_first_item = True
        self._final_json_pages_written = 0
        self._final_json_initialized = False
        self._final_json_finalized = False

    def init_final_outputs(self) -> None:
        self.layout.txt_result_path().parent.mkdir(parents=True, exist_ok=True)
        self.layout.json_result_path().parent.mkdir(parents=True, exist_ok=True)
        self.layout.pdf_result_path().parent.mkdir(parents=True, exist_ok=True)

        with self.layout.txt_result_path().open("w", encoding="utf-8"):
            pass

        with self.layout.json_tmp_path().open("w", encoding="utf-8") as out:
            out.write('{"data":[\n')

        self._final_json_first_item = True
        self._final_json_pages_written = 0
        self._final_json_initialized = True
        self._final_json_finalized = False

    def finalize_final_json(self) -> None:
        if not self._final_json_initialized or self._final_json_finalized:
            return

        with self.layout.json_tmp_path().open("a", encoding="utf-8") as out:
            out.write('\n]}\n')

        self._final_json_finalized = True

        self.layout.json_result_path().parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(self.layout.json_tmp_path()), str(self.layout.json_result_path()))

        logger.info(
            f"Saved final JSON: {self.layout.json_result_path()} "
            f"(pages: {self._final_json_pages_written})"
        )

    def save_page_text(self, chunk_id: int, page_number: int, text: str) -> None:
        self.layout.chunk_page_txt_path(chunk_id, page_number).write_text(text, encoding="utf-8")

    def save_page_json(self, chunk_id: int, page_number: int, text: str) -> None:
        data = {"page": page_number + 1, "text": text or ""}
        self.layout.chunk_page_json_path(chunk_id, page_number).write_text(
            json.dumps(data, ensure_ascii=False),
            encoding="utf-8",
        )

    def build_chunk_outputs(self, chunk_id: int, page_numbers: list[int]) -> None:
        with self.layout.chunk_txt_result_path(chunk_id).open("w", encoding="utf-8") as txt_out:
            for idx, page_number in enumerate(page_numbers):
                page_path = self.layout.chunk_page_txt_path(chunk_id, page_number)
                text = page_path.read_text(encoding="utf-8") if page_path.exists() else ""
                if idx > 0:
                    txt_out.write("\n\n")
                txt_out.write(text)

        with self.layout.chunk_jsonl_result_path(chunk_id).open("w", encoding="utf-8") as jsonl_out:
            for page_number in page_numbers:
                page_path = self.layout.chunk_page_json_path(chunk_id, page_number)
                if page_path.exists():
                    jsonl_out.write(page_path.read_text(encoding="utf-8").strip())
                else:
                    jsonl_out.write(
                        json.dumps({"page": page_number + 1, "text": ""}, ensure_ascii=False)
                    )
                jsonl_out.write("\n")

        chunk_pdf_path = self.layout.chunk_pdf_result_path(chunk_id)
        pdf_page_paths = [
            self.layout.chunk_overlay_pdf_path(chunk_id, page_number)
            for page_number in page_numbers
            if self.layout.chunk_overlay_pdf_path(chunk_id, page_number).exists()
        ]

        if not pdf_page_paths:
            with fitz.open() as empty_doc:
                empty_doc.new_page(width=1, height=1)
                empty_doc.save(chunk_pdf_path, garbage=4, deflate=True, clean=True)
            return

        tool_name = merge_pdfs(pdf_page_paths, chunk_pdf_path)
        logger.info(f"Built chunk PDF {chunk_id} with {tool_name}: {chunk_pdf_path}")

    def append_chunk_txt_to_final(self, chunk_id: int, is_first_chunk: bool) -> None:
        mode = "a" if self.layout.txt_result_path().exists() else "w"
        with (
            self.layout.chunk_txt_result_path(chunk_id).open("r", encoding="utf-8") as src,
            self.layout.txt_result_path().open(mode, encoding="utf-8") as dst,
        ):
            if not is_first_chunk:
                dst.write("\n\n")
            dst.write(src.read())

    def append_chunk_json_to_final(self, chunk_id: int) -> None:
        jsonl_path = self.layout.chunk_jsonl_result_path(chunk_id)
        if not jsonl_path.exists():
            logger.warning(f"Chunk JSONL does not exist: {jsonl_path}")
            return

        with (
            jsonl_path.open("r", encoding="utf-8") as src,
            self.layout.json_tmp_path().open("a", encoding="utf-8") as dst,
        ):
            for line_number, line in enumerate(src, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning(
                        f"Invalid JSON line in chunk {chunk_id}, "
                        f"line {line_number}: {e}; content={line[:200]!r}"
                    )
                    continue

                if not self._final_json_first_item:
                    dst.write(",\n")

                json.dump(obj, dst, ensure_ascii=False)
                self._final_json_first_item = False
                self._final_json_pages_written += 1
