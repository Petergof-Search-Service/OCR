"""Парсинг ответа OCR и построение невидимого текстового слоя (searchable PDF)."""
import io
from typing import Any

import fitz
from pypdf import PdfReader, PdfWriter, Transformation
from reportlab.lib.fonts import addMapping
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

DEFAULT_FONT_NAME = "DejaVuSans"
DEFAULT_FONT_FILE = "DejaVuSans.ttf"


def register_font(font_name: str = DEFAULT_FONT_NAME, font_file: str = DEFAULT_FONT_FILE) -> str:
    """Зарегистрировать TTF-шрифт в reportlab. Путь относительный — резолвится от
    cwd функции (корень src в Yandex Cloud). Возвращает имя шрифта."""
    pdfmetrics.registerFont(TTFont(font_name, font_file))
    addMapping(font_name, 0, 0, font_name)
    return font_name


def extract_text_from_result(
    ocr_result: dict[str, Any] | None,
) -> tuple[str, list[dict[str, float | str]], dict[str, float]]:
    if not ocr_result or "result" not in ocr_result:
        return "", [], {"ocr_width": 0.0, "ocr_height": 0.0}
    result = ocr_result["result"]
    text_annotation = result.get("textAnnotation", {})
    ocr_width = float(text_annotation.get("width", 0.0))
    ocr_height = float(text_annotation.get("height", 0.0))

    full_text: list[str] = []
    text_blocks: list[dict[str, float | str]] = []
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
    ocr_result: dict[str, Any] | None,
) -> list[tuple[str, list[dict[str, float | str]], dict[str, float]]]:
    if not ocr_result or "result" not in ocr_result:
        return []
    result = ocr_result["result"]
    pages = result.get("pages", [result])
    return [extract_text_from_result({"result": page}) for page in pages]


def create_text_overlay_pdf(
    image_only_pdf_bytes: bytes,
    text_blocks: list[dict[str, float | str]],
    ocr_page_dim: dict[str, float],
    font_name: str = DEFAULT_FONT_NAME,
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
        text_obj.setFont(font_name, font_size)
        text_obj._code.append("3 Tr")
        text_width = pdfmetrics.stringWidth(text, font_name, font_size)
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
