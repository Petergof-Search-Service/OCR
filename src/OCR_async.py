import io
import json
import base64
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import fitz
from pypdf import PdfReader, PdfWriter, Transformation
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.fonts import addMapping

from aiohttp import ClientError, ClientTimeout, TCPConnector, ConnectionTimeoutError


class YandexOCRAsync:
    OCR_RECOGNIZE_URL = "https://ocr.api.cloud.yandex.net/ocr/v1/recognizeTextAsync"
    OCR_RESULT_URL = "https://ocr.api.cloud.yandex.net/ocr/v1/getRecognition?operationId={operation_id}"

    def __init__(self, api_key: str, folder_id: str, bucket: str, key: str, s3_client):
        self.api_key = api_key
        self.folder_id = folder_id
        self.bucket = bucket
        self.key = key
        self.s3 = s3_client

        self.base_name = Path(key).stem
        self.headers = {
            "Authorization": f"Api-Key {api_key}",
            "x-folder-id": folder_id,
        }

        self.tmp_prefix = f"OCR-tmp/{self.base_name}"
        self.font_name = "DejaVuSans"

        pdfmetrics.registerFont(TTFont(self.font_name, "DejaVuSans.ttf"))
        addMapping(self.font_name, 0, 0, self.font_name)

    # =========================
    # S3 path helpers
    # =========================

    def _tmp_chunk_s3_key(self, page_number: int) -> str:
        return f"{self.tmp_prefix}/chunk_{page_number}.pdf"

    def _tmp_image_s3_key(self, page_number: int) -> str:
        return f"{self.tmp_prefix}/img_{page_number}.pdf"

    def _tmp_page_txt_s3_key(self, page_number: int) -> str:
        return f"{self.tmp_prefix}/page_{page_number}.txt"

    def _tmp_page_json_s3_key(self, page_number: int) -> str:
        return f"{self.tmp_prefix}/page_{page_number}.json"

    def _tmp_overlay_pdf_s3_key(self, page_number: int) -> str:
        return f"{self.tmp_prefix}/overlay/page_{page_number}.pdf"

    def _txt_s3_key(self) -> str:
        return f"OCR-result/txt-files/{self.base_name}.txt"

    def _json_s3_key(self) -> str:
        return f"OCR-result/json-files/{self.base_name}.json"

    def _pdf_s3_key(self) -> str:
        return f"OCR-result/pdf-files/{self.base_name}.pdf"

    # =========================
    # S3 utils
    # =========================

    def upload_bytes_to_s3(
        self,
        data: bytes,
        s3_key: str,
        content_type: Optional[str] = None,
    ) -> None:
        params = {
            "Bucket": self.bucket,
            "Key": s3_key,
            "Body": data,
        }
        if content_type:
            params["ContentType"] = content_type

        self.s3.put_object(**params)
        print(f"Uploaded: {s3_key}")

    def download_bytes_from_s3(self, s3_key: str) -> bytes:
        obj = self.s3.get_object(Bucket=self.bucket, Key=s3_key)
        return obj["Body"].read()

    def delete_s3_object(self, s3_key: str) -> None:
        try:
            self.s3.delete_object(Bucket=self.bucket, Key=s3_key)
            print(f"Deleted: {s3_key}")
        except Exception as exc:
            print(f"Failed to delete {s3_key}: {exc}")

    def cleanup_tmp_s3_prefix(self) -> None:
        continuation_token = None

        while True:
            params = {
                "Bucket": self.bucket,
                "Prefix": f"{self.tmp_prefix}/",
            }
            if continuation_token:
                params["ContinuationToken"] = continuation_token

            response = self.s3.list_objects_v2(**params)
            contents = response.get("Contents", [])

            if contents:
                objects = [{"Key": item["Key"]} for item in contents]
                self.s3.delete_objects(
                    Bucket=self.bucket,
                    Delete={"Objects": objects},
                )
                print(f"Deleted {len(objects)} temp objects from {self.tmp_prefix}/")

            if not response.get("IsTruncated"):
                break

            continuation_token = response.get("NextContinuationToken")

    # =========================
    # Bytes / serialization utils
    # =========================

    @staticmethod
    def encode_pdf_bytes(pdf_bytes: bytes) -> str:
        return base64.b64encode(pdf_bytes).decode("utf-8")

    def save_page_text(self, page_number: int, text: str) -> None:
        self.upload_bytes_to_s3(
            text.encode("utf-8"),
            self._tmp_page_txt_s3_key(page_number),
            content_type="text/plain; charset=utf-8",
        )

    def save_page_json(self, page_number: int, text: str) -> None:
        data = {
            "page": page_number + 1,
            "text": text,
        }
        self.upload_bytes_to_s3(
            json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8"),
            self._tmp_page_json_s3_key(page_number),
            content_type="application/json",
        )

    # =========================
    # Split / merge
    # =========================

    def split_pdf(self, input_pdf_bytes: bytes) -> List[Tuple[str, int]]:
        """
        Split input PDF into single-page PDFs and upload them to S3 OCR-tmp
        Returns list of tuples: (chunk_s3_key, page_number)
        """
        chunks: List[Tuple[str, int]] = []
        reader = PdfReader(io.BytesIO(input_pdf_bytes))

        for page_number, page in enumerate(reader.pages):
            writer = PdfWriter()
            writer.add_page(page)

            buffer = io.BytesIO()
            writer.write(buffer)
            chunk_bytes = buffer.getvalue()

            chunk_s3_key = self._tmp_chunk_s3_key(page_number)
            self.upload_bytes_to_s3(
                chunk_bytes,
                chunk_s3_key,
                content_type="application/pdf",
            )

            chunks.append((chunk_s3_key, page_number))

        return chunks

    def merge_txt_files(self, total_pages: int) -> bytes:
        parts: List[str] = []

        for page_number in range(total_pages):
            s3_key = self._tmp_page_txt_s3_key(page_number)
            try:
                page_text = self.download_bytes_from_s3(s3_key).decode("utf-8")
                parts.append(page_text)
            except Exception as exc:
                print(f"Failed to read TXT page {page_number + 1}: {exc}")
                parts.append("")

        return "\n\n".join(parts).encode("utf-8")

    def merge_json_files(self, total_pages: int) -> bytes:
        pages_data: List[Dict[str, Any]] = []

        for page_number in range(total_pages):
            s3_key = self._tmp_page_json_s3_key(page_number)
            try:
                page_json = json.loads(self.download_bytes_from_s3(s3_key).decode("utf-8"))
                pages_data.append(page_json)
            except Exception as exc:
                print(f"Failed to read JSON page {page_number + 1}: {exc}")

        result = {"data": pages_data}
        return json.dumps(result, ensure_ascii=False, indent=2).encode("utf-8")

    def merge_pdfs(self, total_pages: int) -> bytes:
        writer = PdfWriter()
        readers: List[PdfReader] = []

        for page_number in range(total_pages):
            s3_key = self._tmp_overlay_pdf_s3_key(page_number)
            try:
                pdf_bytes = self.download_bytes_from_s3(s3_key)
                reader = PdfReader(io.BytesIO(pdf_bytes))
                readers.append(reader)
                writer.add_page(reader.pages[0])
            except Exception as exc:
                print(f"Failed to read overlay PDF page {page_number + 1}: {exc}")

        output = io.BytesIO()
        writer.write(output)
        return output.getvalue()

    # =========================
    # OCR API
    # =========================

    async def recognize_pdf(
        self,
        session: aiohttp.ClientSession,
        pdf_bytes: bytes,
        max_retries: int = 5,
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
                async with session.post(
                    self.OCR_RECOGNIZE_URL,
                    headers=headers,
                    json=body,
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        operation_id = data.get("id")
                        if not operation_id:
                            print(f"[recognize_pdf] No operation id, attempt {attempt}")
                            return None
                        return operation_id

                    response_text = await response.text()

                    if response.status in {429, 500, 502, 503, 504}:
                        print(
                            f"[recognize_pdf] Temporary HTTP error {response.status}, "
                            f"attempt {attempt}/{max_retries}: {response_text}"
                        )
                    else:
                        print(
                            f"[recognize_pdf] Permanent HTTP error {response.status}: {response_text}"
                        )
                        return None

            except ConnectionTimeoutError as exc:
                print(
                    f"[recognize_pdf] Connection timeout, "
                    f"attempt {attempt}/{max_retries}: {exc}"
                )

            except asyncio.TimeoutError as exc:
                print(
                    f"[recognize_pdf] Async timeout, "
                    f"attempt {attempt}/{max_retries}: {exc}"
                )

            except ClientError as exc:
                print(
                    f"[recognize_pdf] Client error, "
                    f"attempt {attempt}/{max_retries}: {type(exc).__name__}: {exc}"
                )

            except Exception as exc:
                print(
                    f"[recognize_pdf] Unexpected error, "
                    f"attempt {attempt}/{max_retries}: {type(exc).__name__}: {exc}"
                )

            if attempt < max_retries:
                sleep_time = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 1)
                await asyncio.sleep(sleep_time)

        print("[recognize_pdf] Failed after all retries")
        return None


    async def get_operation_result(
        self,
        session: aiohttp.ClientSession,
        operation_id: str,
        max_retries: int = 5,
        delay: int = 5,
    ) -> Optional[Dict[str, Any]]:
        url = self.OCR_RESULT_URL.format(operation_id=operation_id)

        for _ in range(max_retries):
            try:
                async with session.get(url, headers=self.headers) as response:
                    if response.status == 200:
                        return await response.json()

                    if response.status == 404:
                        await asyncio.sleep(delay)
                        continue

                    print(f"Failed to get result: {response.status} - {await response.text()}")

            except Exception as exc:
                print(f"Error while polling OCR result: {exc}")

            await asyncio.sleep(delay)

        print(f"Operation timed out after {max_retries * delay} seconds")
        return None

    # =========================
    # OCR result parsing
    # =========================

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

                text_blocks.append({
                    "text": text,
                    "x_min": x_min,
                    "y_min": y_min,
                    "x_max": x_max,
                    "y_max": y_max,
                    "width_ocr": x_max - x_min,
                    "height_ocr": y_max - y_min,
                })

                full_text.append(text)

        return (
            "\n".join(full_text),
            text_blocks,
            {"ocr_width": ocr_width, "ocr_height": ocr_height},
        )

    # =========================
    # PDF processing in memory
    # =========================

    @staticmethod
    def pdf_to_image_only_pdf(input_pdf_bytes: bytes, dpi: int = 300) -> bytes:
        """
        Convert PDF bytes to image-only PDF bytes
        """
        src = fitz.open(stream=input_pdf_bytes, filetype="pdf")
        out = fitz.open()

        try:
            zoom = dpi / 72.0
            matrix = fitz.Matrix(zoom, zoom)

            for page in src:
                pix = page.get_pixmap(matrix=matrix, alpha=False)
                new_page = out.new_page(width=page.rect.width, height=page.rect.height)
                new_page.insert_image(page.rect, stream=pix.tobytes("png"))

            return out.tobytes()
        finally:
            out.close()
            src.close()

    def create_text_overlay_pdf(
        self,
        original_pdf_bytes: bytes,
        text_blocks: List[Dict[str, float | str]],
        ocr_page_dim: Dict[str, float],
    ) -> bytes:
        """
        Create PDF bytes with invisible text layer over original page
        """
        if not text_blocks:
            return original_pdf_bytes

        reader = PdfReader(io.BytesIO(original_pdf_bytes))
        writer = PdfWriter()
        page = reader.pages[0]

        box = page.cropbox
        pdf_width = float(box.width)
        pdf_height = float(box.height)
        x_offset = float(box.left)
        y_offset = float(box.bottom)

        ocr_width = float(ocr_page_dim.get("ocr_width", 0.0))
        ocr_height = float(ocr_page_dim.get("ocr_height", 0.0))

        if ocr_width <= 0 or ocr_height <= 0:
            return original_pdf_bytes

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
        page.merge_transformed_page(
            overlay_page,
            Transformation().translate(x_offset, y_offset)
        )

        writer.add_page(page)

        output = io.BytesIO()
        writer.write(output)
        return output.getvalue()

    # =========================
    # Page processing
    # =========================

    async def process_chunk(
        self,
        session: aiohttp.ClientSession,
        chunk_s3_key: str,
        page_number: int,
        semaphore: asyncio.Semaphore,
    ) -> Optional[Dict[str, int]]:
        async with semaphore:
            image_s3_key = self._tmp_image_s3_key(page_number)
            overlay_s3_key = self._tmp_overlay_pdf_s3_key(page_number)

            try:
                chunk_bytes = self.download_bytes_from_s3(chunk_s3_key)

                # Сильно уменьшает размер и количество таймаутов
                image_pdf_bytes = self.pdf_to_image_only_pdf(chunk_bytes, dpi=100)

                print(
                    f"Page {page_number + 1}: image PDF size = "
                    f"{len(image_pdf_bytes) / 1024 / 1024:.2f} MB"
                )

                self.upload_bytes_to_s3(
                    image_pdf_bytes,
                    image_s3_key,
                    content_type="application/pdf",
                )

                operation_id = await self.recognize_pdf(session, image_pdf_bytes)

                if not operation_id:
                    print(f"OCR start failed for page {page_number + 1}")
                    self.save_page_text(page_number, "")
                    self.save_page_json(page_number, "")
                    self.upload_bytes_to_s3(
                        image_pdf_bytes,
                        overlay_s3_key,
                        content_type="application/pdf",
                    )
                    return None

                ocr_result = await self.get_operation_result(
                    session=session,
                    operation_id=operation_id,
                    max_retries=10,
                    delay=3,
                )

                if not ocr_result:
                    print(f"OCR result failed for page {page_number + 1}")
                    self.save_page_text(page_number, "")
                    self.save_page_json(page_number, "")
                    self.upload_bytes_to_s3(
                        image_pdf_bytes,
                        overlay_s3_key,
                        content_type="application/pdf",
                    )
                    return None

                text, text_blocks, ocr_page_dim = self.extract_text_from_result(ocr_result)

                self.save_page_text(page_number, text)
                self.save_page_json(page_number, text)

                overlay_pdf_bytes = self.create_text_overlay_pdf(
                    original_pdf_bytes=image_pdf_bytes,
                    text_blocks=text_blocks,
                    ocr_page_dim=ocr_page_dim,
                )

                self.upload_bytes_to_s3(
                    overlay_pdf_bytes,
                    overlay_s3_key,
                    content_type="application/pdf",
                )

                return {"page": page_number + 1}

            except Exception as exc:
                print(f"[process_chunk] Page {page_number + 1} failed: {type(exc).__name__}: {exc}")

                try:
                    self.save_page_text(page_number, "")
                    self.save_page_json(page_number, "")
                except Exception as save_exc:
                    print(f"[process_chunk] Failed to save empty result for page {page_number + 1}: {save_exc}")

                return None

            finally:
                self.delete_s3_object(chunk_s3_key)
                self.delete_s3_object(image_s3_key)


    # =========================
    # Full document processing
    # =========================

    async def process_pdf(
        self,
        input_pdf_bytes: bytes,
        max_concurrent: int = 1,
        cleanup_tmp_s3: bool = False,
    ) -> None:
        chunks = self.split_pdf(input_pdf_bytes)
        semaphore = asyncio.Semaphore(max_concurrent)

        timeout = ClientTimeout(
            total=600,
            connect=120,
            sock_connect=120,
            sock_read=300,
        )

        connector = TCPConnector(
            limit=max_concurrent,
            ttl_dns_cache=300,
        )

        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            tasks = [
                asyncio.create_task(
                    self.process_chunk(session, chunk_s3_key, page_number, semaphore)
                )
                for chunk_s3_key, page_number in chunks
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            for idx, result in enumerate(results):
                if isinstance(result, Exception):
                    print(f"[process_pdf] Task for page {idx + 1} crashed: {result}")

        total_pages = len(chunks)

        print("Starting final merge...")

        txt_bytes = self.merge_txt_files(total_pages)
        json_bytes = self.merge_json_files(total_pages)
        pdf_bytes = self.merge_pdfs(total_pages)

        self.upload_bytes_to_s3(
            txt_bytes,
            self._txt_s3_key(),
            content_type="text/plain; charset=utf-8",
        )
        self.upload_bytes_to_s3(
            json_bytes,
            self._json_s3_key(),
            content_type="application/json",
        )
        self.upload_bytes_to_s3(
            pdf_bytes,
            self._pdf_s3_key(),
            content_type="application/pdf",
        )

        print(f"Uploaded merged result: {self._txt_s3_key()}")
        print(f"Uploaded merged result: {self._json_s3_key()}")
        print(f"Uploaded merged result: {self._pdf_s3_key()}")

        if cleanup_tmp_s3:
            self.cleanup_tmp_s3_prefix()
