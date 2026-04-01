from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from config import FOLDER_ID, YANDEX_API_KEY

BASE_URL = "https://ai.api.cloud.yandex.net/v1"
UPLOADS_MANIFEST = Path("uploaded_files.json")
INDEX_REGISTRY = Path("index_registry.json")


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


def load_upload_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Файл {path} не найден. Сначала запустите 1.py для загрузки dataset."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def load_registry(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def save_registry(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


async def create_index(name: str, input_file_ids: list[str], expires_days: int = 30) -> dict[str, Any]:
    client = make_client()

    print("Создаем поисковый индекс...")
    vector_store = await client.vector_stores.create(
        name=name,
        expires_after={"anchor": "last_active_at", "days": expires_days},
        file_ids=input_file_ids,
    )

    vector_store_id = str(vector_store.id)
    print("Vector store создан:", vector_store_id)

    
    while True:
        vector_store = await client.vector_stores.retrieve(vector_store_id)
        print("Статус vector store:", vector_store.status)
        if vector_store.status != "in_progress":
            break
        await asyncio.sleep(3)

    return {
        "name": name,
        "vector_store_id": vector_store_id,
        "status": vector_store.status,
    }


async def async_main() -> None:
    parser = argparse.ArgumentParser(description="Создание индекса по ранее загруженным файлам dataset")
    parser.add_argument("--name", required=True, help="Имя создаваемого индекса")
    parser.add_argument("--manifest", default=str(UPLOADS_MANIFEST), help="Путь к uploaded_files.json")
    parser.add_argument("--registry", default=str(INDEX_REGISTRY), help="Путь к локальному реестру индексов")
    parser.add_argument("--expires-days", type=int, default=30, help="Срок жизни индекса по неактивности")
    args = parser.parse_args()

    manifest = load_upload_manifest(Path(args.manifest))
    file_ids = [item["file_id"] for item in manifest.get("files", []) if item.get("file_id")]
    if not file_ids:
        raise ValueError("В uploaded_files.json нет file_id. Сначала выполните загрузку файлов через 1.py")

    result = await create_index(args.name, file_ids, expires_days=args.expires_days)

    registry_path = Path(args.registry)
    records = load_registry(registry_path)
    records.append(
        {
            "dataset_dir": manifest.get("dataset_dir"),
            "manifest": str(Path(args.manifest).resolve()),
            **result,
        }
    )
    save_registry(registry_path, records)

    print("\nИндекс сохранен в локальный реестр:", registry_path)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(async_main())
