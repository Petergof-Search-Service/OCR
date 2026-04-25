import base64
import gc
import io
import json
import random
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import fitz
from aiohttp import ClientError, ClientTimeout, ConnectionTimeoutError, TCPConnector
from PIL import Image
from pypdf import PdfReader, PdfWriter, Transformation
from reportlab.lib.fonts import addMapping
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


class YandexOCRAsync:
    OCR_RECOGNIZE_URL = "https://ocr.api.cloud.yandex.net/ocr/v1/recognizeTextAsync"
    OCR_RESULT_URL = "https://ocr.api.cloud.yandex.net/ocr/v1/getRecognition?operationId={operation_id}"

    def __init__(
        self,
        api_key: str,
        folder_id: str,
        storage_root: Path,
        key: str,
        tmp_prefix: str = "OCR-tmp",
        result_prefix: str = "OCR-result",
        dpi: int = 200,
        jpeg_quality: int = 90,
        strict_memory_mode: bool = True,
    ):
        self.api_key = api_key
        self.folder_id = folder_id
        self.storage_root = Path(storage_root)
        self.key = key
        self.base_name = Path(key).stem
        self.dpi = dpi
        self.jpeg_quality = jpeg_quality
        self.strict_memory_mode = strict_memory_mode

        self.headers = {
            "Authorization": f"Api-Key {api_key}",
            "x-folder-id": folder_id,
        }

        self.tmp_dir = self.storage_root / tmp_prefix / self.base_name
        self.chunks_dir = self.tmp_dir / "chunks"
        self.result_dir = self.storage_root / result_prefix
        self.result_txt_dir = self.result_dir / "txt-files"
        self.result_json_dir = self.result_dir / "json-files"
        self.result_pdf_dir = self.result_dir / "pdf-files"

        self.font_name = "DejaVuSans"
        pdfmetrics.registerFont(TTFont(self.font_name, "DejaVuSans.ttf"))
        addMapping(self.font_name, 0, 0, self.font_name)

        self._final_json_first_item = True
        self._final_json_pages_written = 0
        self._final_json_initialized = False
        self._final_json_finalized = False

        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        self.result_txt_dir.mkdir(parents=True, exist_ok=True)
        self.result_json_dir.mkdir(parents=True, exist_ok=True)
        self.result_pdf_dir.mkdir(parents=True, exist_ok=True)

    def _chunk_dir(self, chunk_id: int) -> Path:
        path = self.chunks_dir / f"chunk_{chunk_id:05d}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _chunk_pdf_path(self, chunk_id: int) -> Path:
        return self._chunk_dir(chunk_id) / "source.pdf"

    def _chunk_batch_path(self, chunk_id: int, batch_id: int) -> Path:
        return self._chunk_dir(chunk_id) / f"batch_{batch_id:05d}.pdf"

    def _chunk_image_batch_path(self, chunk_id: int, batch_id: int) -> Path:
        return self._chunk_dir(chunk_id) / f"batch_{batch_id:05d}.image.pdf"

    def _chunk_image_page_path(self, chunk_id: int, batch_id: int, page_number: int) -> Path:
        return self._chunk_dir(chunk_id) / f"batch_{batch_id:05d}.page_{page_number:06d}.image.pdf"

    def _chunk_page_txt_path(self, chunk_id: int, page_number: int) -> Path:
        return self._chunk_dir(chunk_id) / f"page_{page_number:06d}.txt"

    def _chunk_page_json_path(self, chunk_id: int, page_number: int) -> Path:
        return self._chunk_dir(chunk_id) / f"page_{page_number:06d}.json"

    def _chunk_overlay_pdf_path(self, chunk_id: int, page_number: int) -> Path:
        return self._chunk_dir(chunk_id) / f"page_{page_number:06d}.pdf"

    def _chunk_txt_result_path(self, chunk_id: int) -> Path:
        return self._chunk_dir(chunk_id) / f"chunk_{chunk_id:05d}.txt"

    def _chunk_jsonl_result_path(self, chunk_id: int) -> Path:
        return self._chunk_dir(chunk_id) / f"chunk_{chunk_id:05d}.jsonl"

    def _chunk_pdf_result_path(self, chunk_id: int) -> Path:
        return self._chunk_dir(chunk_id) / f"chunk_{chunk_id:05d}.pdf"

    def _txt_result_path(self) -> Path:
        return self.result_txt_dir / f"{self.base_name}.txt"

    def _json_result_path(self) -> Path:
        return self.result_json_dir / f"{self.base_name}.json"

    def _pdf_result_path(self) -> Path:
        return self.result_pdf_dir / f"{self.base_name}.pdf"

    @staticmethod
    def encode_pdf_bytes(pdf_bytes: bytes) -> str:
        return base64.b64encode(pdf_bytes).decode("utf-8")

    @staticmethod
    def read_bytes(path: Path) -> bytes:
        return path.read_bytes()

    @staticmethod
    def write_bytes(path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    @staticmethod
    def remove_file(path: Path) -> None:
        try:
            if path.exists():
                path.unlink()
        except Exception as exc:
            print(f"Failed to delete {path}: {exc}")

    def cleanup_tmp_files(self) -> None:
        if not self.tmp_dir.exists():
            return
        for path in sorted(self.tmp_dir.rglob("*"), reverse=True):
            try:
                if path.is_file() or path.is_symlink():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
            except Exception as exc:
                print(f"Failed to delete {path}: {exc}")
        try:
            self.tmp_dir.rmdir()
        except Exception:
            pass

    def init_final_outputs(self) -> None:
        self._txt_result_path().parent.mkdir(parents=True, exist_ok=True)
        self._json_result_path().parent.mkdir(parents=True, exist_ok=True)
        self._pdf_result_path().parent.mkdir(parents=True, exist_ok=True)

        with self._txt_result_path().open("w", encoding="utf-8"):
            pass

        with self._json_result_path().open("w", encoding="utf-8") as out:
            out.write('{"data":[\n')

        self._final_json_first_item = True
        self._final_json_pages_written = 0
        self._final_json_initialized = True
        self._final_json_finalized = False

    def finalize_final_json(self) -> None:
        if not self._final_json_initialized or self._final_json_finalized:
            return

        with self._json_result_path().open("a", encoding="utf-8") as out:
            out.write('\n]}\n')

        self._final_json_finalized = True
        print(
            f"Saved final JSON: {self._json_result_path()} "
            f"(pages: {self._final_json_pages_written})"
        )

    def save_page_text(self, chunk_id: int, page_number: int, text: str) -> None:
        self._chunk_page_txt_path(chunk_id, page_number).write_text(text, encoding="utf-8")

    def save_page_json(self, chunk_id: int, page_number: int, text: str) -> None:
        data = {"page": page_number + 1, "text": text or ""}
        self._chunk_page_json_path(chunk_id, page_number).write_text(
            json.dumps(data, ensure_ascii=False),
            encoding="utf-8",
        )

    def split_pdf_to_chunks(self, input_pdf_path: Path, chunk_size: int = 50) -> List[Tuple[int, Path, List[int]]]:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        chunks: List[Tuple[int, Path, List[int]]] = []
        with input_pdf_path.open("rb") as src:
            reader = PdfReader(src)
            total_pages = len(reader.pages)
            for chunk_id, start in enumerate(range(0, total_pages, chunk_size)):
                end = min(start + chunk_size, total_pages)
                page_numbers = list(range(start, end))
                writer = PdfWriter()
                for page_number in page_numbers:
                    writer.add_page(reader.pages[page_number])
                chunk_pdf_path = self._chunk_pdf_path(chunk_id)
                with chunk_pdf_path.open("wb") as dst:
                    writer.write(dst)
                chunks.append((chunk_id, chunk_pdf_path, page_numbers))
        return chunks

    def split_chunk_to_batches(
        self,
        chunk_id: int,
        chunk_pdf_path: Path,
        global_page_numbers: List[int],
        batch_size: int = 1,
    ) -> List[Tuple[Path, List[int], int]]:
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        batches: List[Tuple[Path, List[int], int]] = []
        with chunk_pdf_path.open("rb") as src:
            reader = PdfReader(src)
            total_pages = len(reader.pages)
            batch_id = 0
            for i in range(0, total_pages, batch_size):
                writer = PdfWriter()
                local_end = min(i + batch_size, total_pages)
                for local_page_index in range(i, local_end):
                    writer.add_page(reader.pages[local_page_index])
                batch_path = self._chunk_batch_path(chunk_id, batch_id)
                with batch_path.open("wb") as dst:
                    writer.write(dst)
                batches.append((batch_path, global_page_numbers[i:local_end], batch_id))
                batch_id += 1
        return batches

    def render_batch_to_image_pdfs(
        self,
        batch_pdf_path: Path,
        chunk_id: int,
        batch_id: int,
        page_numbers: List[int],
    ) -> Tuple[Path, List[Path]]:
        zoom = self.dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        batch_output = fitz.open()
        per_page_paths: List[Path] = []
        src = fitz.open(batch_pdf_path)
        try:
            for local_index, page in enumerate(src):
                page_number = page_numbers[local_index]
                pix = page.get_pixmap(matrix=matrix, alpha=False)
                img = None
                try:
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    jpeg_buffer = io.BytesIO()
                    try:
                        img.save(
                            jpeg_buffer,
                            format="JPEG",
                            quality=self.jpeg_quality,
                            optimize=True,
                        )
                        jpeg_bytes = jpeg_buffer.getvalue()
                    finally:
                        jpeg_buffer.close()
                finally:
                    del pix
                    if img is not None:
                        del img

                page_rect = page.rect
                single_doc = fitz.open()
                try:
                    single_page = single_doc.new_page(width=page_rect.width, height=page_rect.height)
                    single_page.insert_image(page_rect, stream=jpeg_bytes, keep_proportion=True)
                    single_bytes = single_doc.tobytes(garbage=4, deflate=True, clean=True)
                finally:
                    single_doc.close()

                single_page_path = self._chunk_image_page_path(chunk_id, batch_id, page_number)
                self.write_bytes(single_page_path, single_bytes)
                per_page_paths.append(single_page_path)

                batch_page = batch_output.new_page(width=page_rect.width, height=page_rect.height)
                batch_page.insert_image(page_rect, stream=jpeg_bytes, keep_proportion=True)

                del jpeg_bytes
                del single_bytes
                gc.collect()

            batch_image_path = self._chunk_image_batch_path(chunk_id, batch_id)
            batch_bytes = batch_output.tobytes(garbage=4, deflate=True, clean=True)
            self.write_bytes(batch_image_path, batch_bytes)
            del batch_bytes
            return batch_image_path, per_page_paths
        finally:
            batch_output.close()
            src.close()
            gc.collect()

    async def recognize_pdf(
        self,
        session: aiohttp.ClientSession,
        pdf_bytes: bytes,
        max_retries: int = 10,
        base_delay: float = 2.0,
    ) -> Optional[str]:
        body = {
            "mimeType": "application/pdf",
            "languageCodes": ["*"],
            "model": "page",
            "content": self.encode_pdf_bytes(pdf_bytes),
        }
        headers = {**self.headers, "Content-Type": "application/json"}
        for attempt in range(1, max_retries + 1):
            try:
                async with session.post(self.OCR_RECOGNIZE_URL, headers=headers, json=body) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("id")
                    response_text = await response.text()
                    if response.status in {429, 500, 502, 503, 504}:
                        print(f"attempt {attempt}/{max_retries}: {response_text}")
                    else:
                        print(f"Permanent HTTP error {response.status}: {response_text}")
                        return None
            except (ConnectionTimeoutError, TimeoutError, ClientError) as exc:
                print(f"attempt {attempt}/{max_retries}: {type(exc).__name__}: {exc}")
            except Exception as exc:
                print(f"attempt {attempt}/{max_retries}: {type(exc).__name__}: {exc}")
            if attempt < max_retries:
                import asyncio
                await asyncio.sleep(base_delay + random.uniform(0, 1))
        print("Failed after all retries")
        return None

    async def get_operation_result(
        self,
        session: aiohttp.ClientSession,
        operation_id: str,
        max_retries: int = 30,
        delay: int = 3,
    ) -> Optional[Dict[str, Any]]:
        import asyncio

        url = self.OCR_RESULT_URL.format(operation_id=operation_id)
        for _ in range(max_retries):
            try:
                async with session.get(url, headers=self.headers) as response:
                    if response.status == 200:
                        text = await response.text()
                        results = []
                        for line in text.strip().split("\n"):
                            if line.strip():
                                try:
                                    results.append(json.loads(line))
                                except json.JSONDecodeError:
                                    continue
                        if not results:
                            await asyncio.sleep(delay)
                            continue
                        if len(results) == 1:
                            return results[0]
                        combined = {"result": {"pages": []}}
                        for result in results:
                            if "result" not in result:
                                continue
                            if "pages" in result["result"]:
                                combined["result"]["pages"].extend(result["result"]["pages"])
                            else:
                                combined["result"]["pages"].append(result["result"])
                        return combined
                    if response.status == 404:
                        await asyncio.sleep(delay)
                        continue
                    print(f"Failed to get result: {response.status}")
                    await asyncio.sleep(delay)
            except Exception as exc:
                print(f"Error while polling OCR result: {exc}")
                await asyncio.sleep(delay)
        print(f"Operation timed out after {max_retries * delay} seconds")
        return None

    def extract_text_from_result(
        self,
        ocr_result: Optional[Dict[str, Any]],
    ) -> Tuple[str, List[Dict[str, float | str]], Dict[str, float]]:
        if not ocr_result or "result" not in ocr_result:
            return "", [], {"ocr_width": 0.0, "ocr_height": 0.0}
        result = ocr_result["result"]
        text_annotation = result.get("textAnnotation", {})
        ocr_width = float(text_annotation.get("width", 0.0))
        ocr_height = float(text_annotation.get("height", 0.0))

        full_text: List[str] = []
        text_blocks: List[Dict[str, float | str]] = []
        for block in text_annotation.get("blocks", []):
            for line in block.get("lines", []):
                text = line.get("text")
                if not text:
                    continue
                vertices = line.get("boundingBox", {}).get("vertices", [])
                if not vertices:
                    continue
                xs = [float(v.get("x", 0.0)) for v in vertices]
                ys = [float(v.get("y", 0.0)) for v in vertices]
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
                text_blocks.append(
                    {
                        "text": text,
                        "x_min": x_min,
                        "y_min": y_min,
                        "x_max": x_max,
                        "y_max": y_max,
                    }
                )
                full_text.append(text)
        return "\n".join(full_text), text_blocks, {"ocr_width": ocr_width, "ocr_height": ocr_height}

    def parse_multi_page_result(
        self,
        ocr_result: Optional[Dict[str, Any]],
    ) -> List[Tuple[str, List[Dict[str, float | str]], Dict[str, float]]]:
        if not ocr_result or "result" not in ocr_result:
            return []
        result = ocr_result["result"]
        pages = result.get("pages", [result])
        return [self.extract_text_from_result({"result": page}) for page in pages]

    def create_text_overlay_pdf(
        self,
        image_only_pdf_bytes: bytes,
        text_blocks: List[Dict[str, float | str]],
        ocr_page_dim: Dict[str, float],
    ) -> bytes:
        reader = PdfReader(io.BytesIO(image_only_pdf_bytes))
        writer = PdfWriter()
        page = reader.pages[0]
        if not text_blocks:
            writer.add_page(page)
            output = io.BytesIO()
            writer.write(output)
            return output.getvalue()

        box = page.cropbox
        pdf_width = float(box.width)
        pdf_height = float(box.height)
        x_offset = float(box.left)
        y_offset = float(box.bottom)

        ocr_width = float(ocr_page_dim.get("ocr_width", 0.0))
        ocr_height = float(ocr_page_dim.get("ocr_height", 0.0))
        if ocr_width <= 0 or ocr_height <= 0:
            writer.add_page(page)
            output = io.BytesIO()
            writer.write(output)
            return output.getvalue()

        scale_x = pdf_width / ocr_width
        scale_y = pdf_height / ocr_height
        packet = io.BytesIO()
        pdf_canvas = canvas.Canvas(packet, pagesize=(pdf_width, pdf_height))

        for block in text_blocks:
            text = str(block["text"])
            x_min = float(block["x_min"])
            x_max = float(block["x_max"])
            y_min = float(block["y_min"])
            y_max = float(block["y_max"])

            x_pdf = x_min * scale_x
            target_width = (x_max - x_min) * scale_x
            y_bottom_pdf = y_max * scale_y
            y_pdf = pdf_height - y_bottom_pdf
            target_height = (y_max - y_min) * scale_y
            font_size = max(target_height * 0.9, 1.0)

            text_obj = pdf_canvas.beginText()
            text_obj.setFont(self.font_name, font_size)
            text_obj._code.append("3 Tr")
            text_width = pdfmetrics.stringWidth(text, self.font_name, font_size)
            if text_width > 0 and target_width > 0:
                text_obj.setHorizScale(100.0 * target_width / text_width)
            text_obj.setTextOrigin(x_pdf, y_pdf)
            text_obj.textLine(text)
            text_obj._code.append("0 Tr")
            pdf_canvas.drawText(text_obj)

        pdf_canvas.save()
        packet.seek(0)
        overlay_page = PdfReader(packet).pages[0]
        page.merge_transformed_page(overlay_page, Transformation().translate(x_offset, y_offset))
        writer.add_page(page)
        output = io.BytesIO()
        writer.write(output)
        result_bytes = output.getvalue()
        try:
            doc = fitz.open(stream=result_bytes, filetype="pdf")
            compressed = doc.tobytes(garbage=4, deflate=True, clean=True)
            doc.close()
            return compressed
        except Exception:
            return result_bytes

    async def process_batch(
        self,
        session: aiohttp.ClientSession,
        chunk_id: int,
        batch_path: Path,
        page_numbers: List[int],
        batch_id: int,
    ) -> None:
        image_batch_path: Optional[Path] = None
        image_page_paths: List[Path] = []
        try:
            image_batch_path, image_page_paths = self.render_batch_to_image_pdfs(
                batch_pdf_path=batch_path,
                chunk_id=chunk_id,
                batch_id=batch_id,
                page_numbers=page_numbers,
            )
            image_batch_bytes = self.read_bytes(image_batch_path)
            try:
                operation_id = await self.recognize_pdf(session, image_batch_bytes)
            finally:
                del image_batch_bytes
                gc.collect()

            if not operation_id:
                print(f"OCR start failed for chunk {chunk_id}, batch {batch_id}")
                for page_num in page_numbers:
                    self.save_page_text(chunk_id, page_num, "")
                    self.save_page_json(chunk_id, page_num, "")
                return

            ocr_result = await self.get_operation_result(
                session=session,
                operation_id=operation_id,
                max_retries=30,
                delay=3,
            )
            if not ocr_result:
                print(f"OCR result failed for chunk {chunk_id}, batch {batch_id}")
                for page_num in page_numbers:
                    self.save_page_text(chunk_id, page_num, "")
                    self.save_page_json(chunk_id, page_num, "")
                return

            parsed_pages = self.parse_multi_page_result(ocr_result)
            for idx, page_num in enumerate(page_numbers):
                if idx >= len(parsed_pages):
                    self.save_page_text(chunk_id, page_num, "")
                    self.save_page_json(chunk_id, page_num, "")
                    continue

                text, text_blocks, ocr_page_dim = parsed_pages[idx]
                self.save_page_text(chunk_id, page_num, text)
                self.save_page_json(chunk_id, page_num, text)

                page_image_path = image_page_paths[idx]
                image_single_page_bytes = self.read_bytes(page_image_path)
                try:
                    overlay_pdf_bytes = self.create_text_overlay_pdf(
                        image_single_page_bytes,
                        text_blocks,
                        ocr_page_dim,
                    )
                    self.write_bytes(self._chunk_overlay_pdf_path(chunk_id, page_num), overlay_pdf_bytes)
                finally:
                    del image_single_page_bytes
                    gc.collect()

                print(f"Processed page {page_num + 1} from chunk {chunk_id}, batch {batch_id}")
                del overlay_pdf_bytes
                del text_blocks
                gc.collect()

            del parsed_pages
            del ocr_result
            gc.collect()
        except Exception as exc:
            print(f"Chunk {chunk_id} batch {batch_id} failed: {type(exc).__name__}: {exc}")
            import traceback
            traceback.print_exc()
            for page_num in page_numbers:
                try:
                    self.save_page_text(chunk_id, page_num, "")
                    self.save_page_json(chunk_id, page_num, "")
                except Exception as save_exc:
                    print(f"Failed to save empty result for page {page_num + 1}: {save_exc}")
        finally:
            self.remove_file(batch_path)
            if image_batch_path is not None:
                self.remove_file(image_batch_path)
            for image_page_path in image_page_paths:
                self.remove_file(image_page_path)
            gc.collect()

    def build_chunk_outputs(self, chunk_id: int, page_numbers: List[int]) -> None:
        with self._chunk_txt_result_path(chunk_id).open("w", encoding="utf-8") as txt_out:
            for idx, page_number in enumerate(page_numbers):
                page_path = self._chunk_page_txt_path(chunk_id, page_number)
                text = page_path.read_text(encoding="utf-8") if page_path.exists() else ""
                if idx > 0:
                    txt_out.write("\n\n")
                txt_out.write(text)

        with self._chunk_jsonl_result_path(chunk_id).open("w", encoding="utf-8") as jsonl_out:
            for page_number in page_numbers:
                page_path = self._chunk_page_json_path(chunk_id, page_number)
                if page_path.exists():
                    jsonl_out.write(page_path.read_text(encoding="utf-8").strip())
                else:
                    jsonl_out.write(json.dumps({"page": page_number + 1, "text": ""}, ensure_ascii=False))
                jsonl_out.write("\n")

        chunk_pdf_path = self._chunk_pdf_result_path(chunk_id)
        pdf_page_paths = [
            self._chunk_overlay_pdf_path(chunk_id, page_number)
            for page_number in page_numbers
            if self._chunk_overlay_pdf_path(chunk_id, page_number).exists()
        ]

        if not pdf_page_paths:
            with fitz.open() as empty_doc:
                empty_doc.new_page(width=1, height=1)
                empty_doc.save(chunk_pdf_path, garbage=4, deflate=True, clean=True)
            return

        try:
            tool_name = self._run_pdf_merge_tool(pdf_page_paths, chunk_pdf_path)
            print(f"Built chunk PDF {chunk_id} with {tool_name}: {chunk_pdf_path}")
            return
        except RuntimeError:
            pass

        writer = PdfWriter()
        for page_path in pdf_page_paths:
            with page_path.open("rb") as src:
                reader = PdfReader(src)
                if reader.pages:
                    writer.add_page(reader.pages[0])

        with chunk_pdf_path.open("wb") as dst:
            writer.write(dst)

    def append_chunk_txt_to_final(self, chunk_id: int, is_first_chunk: bool) -> None:
        mode = "a" if self._txt_result_path().exists() else "w"
        with self._chunk_txt_result_path(chunk_id).open("r", encoding="utf-8") as src, self._txt_result_path().open(
            mode,
            encoding="utf-8",
        ) as dst:
            if not is_first_chunk:
                dst.write("\n\n")
            dst.write(src.read())

    def append_chunk_json_to_final(self, chunk_id: int) -> None:
        jsonl_path = self._chunk_jsonl_result_path(chunk_id)
        if not jsonl_path.exists():
            print(f"Chunk JSONL does not exist: {jsonl_path}")
            return

        with jsonl_path.open("r", encoding="utf-8") as src, self._json_result_path().open("a", encoding="utf-8") as dst:
            for line_number, line in enumerate(src, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    print(
                        f"Invalid JSON line in chunk {chunk_id}, "
                        f"line {line_number}: {e}; content={line[:200]!r}"
                    )
                    continue

                if not self._final_json_first_item:
                    dst.write(",\n")

                json.dump(obj, dst, ensure_ascii=False)
                self._final_json_first_item = False
                self._final_json_pages_written += 1


    def _run_pdf_merge_tool(self, input_paths: List[Path], output_path: Path) -> str:
        if not input_paths:
            raise ValueError("input_paths is empty")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        pdfunite_path = shutil.which("pdfunite")
        if pdfunite_path:
            cmd = [pdfunite_path, *[str(p) for p in input_paths], str(output_path)]
            subprocess.run(cmd, check=True)
            return "pdfunite"

        qpdf_path = shutil.which("qpdf")
        if qpdf_path:
            cmd = [qpdf_path, "--empty", "--pages", *[str(p) for p in input_paths], "--", str(output_path)]
            subprocess.run(cmd, check=True)
            return "qpdf"

        gs_path = shutil.which("gs")
        if gs_path:
            cmd = [
                gs_path,
                "-dBATCH",
                "-dNOPAUSE",
                "-q",
                "-sDEVICE=pdfwrite",
                f"-sOutputFile={output_path}",
                *[str(p) for p in input_paths],
            ]
            subprocess.run(cmd, check=True)
            return "ghostscript"

        raise RuntimeError("No PDF merge tool found: pdfunite, qpdf, gs")

    def _merge_pdf_files_tree(
        self,
        input_paths: List[Path],
        output_path: Path,
        fan_in: int = 4,
    ) -> None:
        if not input_paths:
            raise FileNotFoundError("No input PDF files to merge")
        if fan_in < 2:
            raise ValueError("fan_in must be >= 2")

        current_paths = input_paths[:]
        merge_tmp_dir = self.tmp_dir / "merge-rounds"
        merge_tmp_dir.mkdir(parents=True, exist_ok=True)
        round_index = 0

        while len(current_paths) > 1:
            next_paths: List[Path] = []
            print(
                f"Final PDF merge round {round_index}: "
                f"{len(current_paths)} files, fan_in={fan_in}"
            )

            for group_index in range(0, len(current_paths), fan_in):
                group = current_paths[group_index: group_index + fan_in]
                if len(group) == 1:
                    next_paths.append(group[0])
                    continue

                intermediate_path = (
                    merge_tmp_dir
                    / f"round_{round_index:04d}_group_{group_index // fan_in:04d}.pdf"
                )
                tool_name = self._run_pdf_merge_tool(group, intermediate_path)
                print(
                    f"Merged round={round_index} group={group_index // fan_in} "
                    f"with {tool_name}: {len(group)} -> {intermediate_path.name}"
                )
                next_paths.append(intermediate_path)

            for old_path in current_paths:
                try:
                    if old_path not in input_paths and old_path.exists():
                        old_path.unlink()
                except Exception as exc:
                    print(f"Failed to delete intermediate merge file {old_path}: {exc}")

            current_paths = next_paths
            round_index += 1
            gc.collect()

        final_candidate = current_paths[0]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            output_path.unlink()

        if final_candidate == output_path:
            print(f"Saved final PDF: {output_path}")
            return

        final_candidate.replace(output_path)
        print(f"Saved final PDF: {output_path}")

    def build_final_pdf_from_chunks(self, chunk_ids: List[int]) -> None:
        output_path = self._pdf_result_path()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        chunk_pdf_paths = [
            self._chunk_pdf_result_path(chunk_id)
            for chunk_id in chunk_ids
            if self._chunk_pdf_result_path(chunk_id).exists()
        ]

        if not chunk_pdf_paths:
            raise FileNotFoundError("No chunk PDF files were created")

        # Merge in a tree to keep external tool memory bounded.
        self._merge_pdf_files_tree(
            input_paths=chunk_pdf_paths,
            output_path=output_path,
            fan_in=4,
        )

    async def process_chunk(
        self,
        session: aiohttp.ClientSession,
        chunk_id: int,
        chunk_pdf_path: Path,
        page_numbers: List[int],
        max_concurrent: int,
        batch_size: int,
    ) -> None:
        batches = self.split_chunk_to_batches(
            chunk_id,
            chunk_pdf_path,
            page_numbers,
            batch_size=batch_size,
        )

        effective_parallel_batches = 1 if self.strict_memory_mode else max(1, max_concurrent)

        if effective_parallel_batches != 1:
            import asyncio

            semaphore = asyncio.Semaphore(effective_parallel_batches)

            async def _run(batch_path: Path, pages: List[int], batch_id: int) -> None:
                async with semaphore:
                    await self.process_batch(session, chunk_id, batch_path, pages, batch_id)

            tasks = [
                asyncio.create_task(_run(batch_path, pages, batch_id))
                for batch_path, pages, batch_id in batches
            ]
            await asyncio.gather(*tasks, return_exceptions=True)
        else:
            for batch_path, pages, batch_id in batches:
                await self.process_batch(session, chunk_id, batch_path, pages, batch_id)

        self.build_chunk_outputs(chunk_id, page_numbers)
        self.append_chunk_txt_to_final(chunk_id, is_first_chunk=(chunk_id == 0))
        self.append_chunk_json_to_final(chunk_id)

        self.remove_file(chunk_pdf_path)
        gc.collect()

    async def process_pdf(
        self,
        input_pdf_path: Path,
        max_concurrent: int = 1,
        batch_size: int = 1,
        chunk_size: int = 50,
        cleanup_tmp_files: bool = False,
    ) -> None:
        self.init_final_outputs()

        chunks = self.split_pdf_to_chunks(input_pdf_path=input_pdf_path, chunk_size=chunk_size)
        print(f"Split into {len(chunks)} chunks")
        timeout = ClientTimeout(total=600, connect=120, sock_connect=120, sock_read=300)
        connector = TCPConnector(limit=max(1, max_concurrent), ttl_dns_cache=300)
        processed_chunk_ids: List[int] = []

        try:
            async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
                for chunk_id, chunk_pdf_path, page_numbers in chunks:
                    print(f"Processing chunk {chunk_id} with {len(page_numbers)} pages")
                    await self.process_chunk(
                        session,
                        chunk_id,
                        chunk_pdf_path,
                        page_numbers,
                        max_concurrent=max_concurrent,
                        batch_size=batch_size,
                    )
                    processed_chunk_ids.append(chunk_id)
                    gc.collect()

            self.finalize_final_json()

            print("Building final PDF from chunk files...")
            self.build_final_pdf_from_chunks(processed_chunk_ids)

            print(f"Uploaded merged result: {self._txt_result_path()}")
            print(f"Uploaded merged result: {self._json_result_path()}")
            print(f"Uploaded merged result: {self._pdf_result_path()}")
        except Exception:
            try:
                self.finalize_final_json()
            except Exception:
                pass
            raise
        finally:
            if cleanup_tmp_files:
                self.cleanup_tmp_files()
