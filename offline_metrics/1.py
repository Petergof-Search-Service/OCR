from __future__ import annotations

import argparse
import asyncio
import io
import json
import re
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from config import FOLDER_ID, YANDEX_API_KEY

DATASET_DIR = Path("dataset")
UPLOADS_MANIFEST = Path("uploaded_files.json")
BASE_URL = "https://ai.api.cloud.yandex.net/v1"
MAX_CHUNK_LEN = 8192


def validate_config() -> None:
    if not FOLDER_ID:
        raise ValueError("Не задан FOLDER_ID в .env")
    if not YANDEX_API_KEY:
        raise ValueError("Не задан YANDEX_API_KEY в .env")


def make_client() -> AsyncOpenAI:
    validate_config()
    return AsyncOpenAI(
        api_key=YANDEX_API_KEY,
        base_url=BASE_URL,
        project=FOLDER_ID,
    )


def parse_pages_from_bytes(data: bytes, filename: str) -> list[dict[str, Any]]:
    try:
        obj = json.loads(data.decode("utf-8"))
    except Exception as exc:
        raise ValueError(f"Не удалось распарсить JSON из {filename}: {exc}") from exc

    if not (isinstance(obj, dict) and isinstance(obj.get("data"), list)):
        raise ValueError(
            f"Неожиданный формат файла {filename}. Ожидался объект с полем 'data' (list)."
        )

    by_page: dict[int, list[str]] = {}
    for item in obj["data"]:
        if not isinstance(item, dict):
            continue
        if "page" not in item:
            continue

        try:
            page = int(item["page"])
        except Exception:
            continue

        text = str(item.get("text", "") or "").strip()
        if text:
            by_page.setdefault(page, []).append(text)

    pages = [
        {"page": page, "text": "\n".join(parts).strip()}
        for page, parts in by_page.items()
        if parts
    ]
    pages.sort(key=lambda item: int(item["page"]))
    return pages


def build_marked_text(pages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for item in pages:
        parts.append(f"[PAGE {item['page']}]\n{item['text']}\n")
    return "\n".join(parts).strip() + "\n"


def _extract_page_markers_with_pos(text: str) -> list[tuple[int, int]]:
    return [(match.start(), int(match.group(1))) for match in re.finditer(r"\[PAGE\s+(\d+)\]", text)]


def _pages_for_slice(markers: list[tuple[int, int]], start: int, end: int) -> list[int]:
    pages_in_slice: list[int] = []
    seen: set[int] = set()

    for pos, page in markers:
        if start <= pos < end and page not in seen:
            seen.add(page)
            pages_in_slice.append(page)

    if pages_in_slice:
        return pages_in_slice

    last_page: int | None = None
    for pos, page in markers:
        if pos < start:
            last_page = page
        else:
            break

    return [last_page] if last_page is not None else []


def _pages_header(pages: list[int]) -> str:
    if not pages:
        return "PAGES: unknown"
    if len(pages) == 1:
        return f"PAGES: {pages[0]}"
    return f"PAGES: {pages[0]}-{pages[-1]}"


def _strip_page_markers(fragment: str) -> str:
    cleaned = re.sub(r"\s*\[PAGE\s+\d+\]\s*\n?", "\n", fragment)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def chunk_text_window_overlap(
    marked_text: str,
    window_chars: int,
    overlap_chars: int,
) -> list[dict[str, str]]:
    if window_chars <= 0:
        raise ValueError("window_chars должен быть > 0")
    if overlap_chars < 0:
        raise ValueError("overlap_chars должен быть >= 0")
    if overlap_chars >= window_chars:
        raise ValueError("overlap_chars должен быть < window_chars")

    window_chars = min(window_chars, MAX_CHUNK_LEN)
    step = window_chars - overlap_chars
    markers = _extract_page_markers_with_pos(marked_text)

    chunks: list[dict[str, str]] = []
    start = 0
    text_len = len(marked_text)

    while start < text_len:
        end = min(start + window_chars, text_len)
        fragment = _strip_page_markers(marked_text[start:end])

        if fragment:
            pages = _pages_for_slice(markers, start, end)
            header = _pages_header(pages)
            body = f"{header}\n{fragment}".strip()

            if len(body) > MAX_CHUNK_LEN:
                allowed = MAX_CHUNK_LEN - len(header) - 1
                body = f"{header}\n{fragment[:max(0, allowed)].rstrip()}".strip()

            chunks.append({"body": body})

        if end == text_len:
            break
        start += step

    return chunks


def chunks_to_jsonl_bytes(chunks: list[dict[str, str]]) -> bytes:
    payload = "\n".join(json.dumps(chunk, ensure_ascii=False) for chunk in chunks) + "\n"
    return payload.encode("utf-8")


async def upload_chunks_jsonl_bytes(
    client: AsyncOpenAI,
    jsonl_data: bytes,
    upload_name: str,
    expires_seconds: int = 3600,
) -> str:
    bio = io.BytesIO(jsonl_data)
    bio.name = upload_name

    uploaded = await client.files.create(
        file=(upload_name, bio, "application/jsonlines"),
        purpose="assistants",
        expires_after={"anchor": "created_at", "seconds": expires_seconds},
        extra_body={"format": "chunks"},
    )
    return str(uploaded.id)


def collect_dataset_files(dataset_dir: Path) -> list[Path]:
    return sorted(path for path in dataset_dir.rglob("*.json") if path.is_file())


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"dataset_dir": None, "files": []}
    return json.loads(path.read_text(encoding="utf-8"))


def save_manifest(path: Path, dataset_dir: Path, files: list[dict[str, Any]]) -> None:
    payload = {
        "dataset_dir": str(dataset_dir.resolve()),
        "files": files,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


async def process_dataset(
    dataset_dir: Path,
    manifest_path: Path,
    window_chars: int,
    overlap_chars: int,
    expires_seconds: int,
) -> list[dict[str, Any]]:
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Директория не найдена: {dataset_dir}")

    dataset_files = collect_dataset_files(dataset_dir)
    if not dataset_files:
        raise FileNotFoundError(f"В директории {dataset_dir} не найдено JSON-файлов")

    client = make_client()
    uploaded_files: list[dict[str, Any]] = []

    for file_path in dataset_files:
        print(f"Обрабатывается файл: {file_path}")
        raw = file_path.read_bytes()
        pages = parse_pages_from_bytes(raw, file_path.name)
        marked_text = build_marked_text(pages)
        chunks = chunk_text_window_overlap(marked_text, window_chars, overlap_chars)

        if not chunks:
            print(f"Пропущен файл без текста: {file_path}")
            continue

        upload_name = f"{file_path.stem}.chunks.jsonl"
        file_id = await upload_chunks_jsonl_bytes(
            client=client,
            jsonl_data=chunks_to_jsonl_bytes(chunks),
            upload_name=upload_name,
            expires_seconds=expires_seconds,
        )

        info = {
            "source_file": str(file_path.resolve()),
            "upload_name": upload_name,
            "file_id": file_id,
            "chunks_count": len(chunks),
        }
        uploaded_files.append(info)
        print(f"Готово: {file_path.name} -> file_id={file_id}, chunks={len(chunks)}")

    save_manifest(manifest_path, dataset_dir, uploaded_files)
    return uploaded_files


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Подготовка и загрузка JSON-файлов из локальной директории dataset в Yandex Vector Store Files"
    )
    parser.add_argument("--dataset-dir", default=str(DATASET_DIR), help="Путь к директории с JSON-файлами")
    parser.add_argument("--manifest", default=str(UPLOADS_MANIFEST), help="Путь к файлу с результатами загрузки")
    parser.add_argument("--window", type=int, default=400, help="Размер чанка в символах")
    parser.add_argument("--overlap", type=int, default=50, help="Перекрытие чанков в символах")
    parser.add_argument("--expires-seconds", type=int, default=3600, help="Срок жизни временных uploaded files")
    return parser


async def async_main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    uploaded = await process_dataset(
        dataset_dir=Path(args.dataset_dir),
        manifest_path=Path(args.manifest),
        window_chars=args.window,
        overlap_chars=args.overlap,
        expires_seconds=args.expires_seconds,
    )

    print("\nЗагрузка завершена.")
    print(f"Загружено файлов: {len(uploaded)}")
    print(f"Манифест сохранен в: {args.manifest}")


if __name__ == "__main__":
    asyncio.run(async_main())
