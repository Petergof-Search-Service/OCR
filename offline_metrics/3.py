from __future__ import annotations

import argparse
import asyncio
import json

from openai import AsyncOpenAI

from config import FOLDER_ID, YANDEX_API_KEY

BASE_URL = "https://ai.api.cloud.yandex.net/v1"


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


async def get_indexes(to_sort: bool = False) -> list[str]:
    client = make_client()
    vector_stores = await client.vector_stores.list()
    names = [item.name for item in vector_stores.data]
    return sorted(names) if to_sort else names


async def get_indexes_names2ids(to_sort: bool = False) -> dict[str, str]:
    client = make_client()
    vector_stores = await client.vector_stores.list()
    mapping = {str(item.name): str(item.id) for item in vector_stores.data}
    if to_sort:
        return dict(sorted(mapping.items(), key=lambda pair: pair[0]))
    return mapping


async def async_main() -> None:
    parser = argparse.ArgumentParser(description="Просмотр доступных индексов")
    parser.add_argument("--sorted", action="store_true", help="Отсортировать по имени")
    parser.add_argument(
        "--format",
        choices=["names", "json"],
        default="json",
        help="names — только список имен, json — словарь name -> id",
    )
    args = parser.parse_args()

    if args.format == "names":
        result = await get_indexes(to_sort=args.sorted)
    else:
        result = await get_indexes_names2ids(to_sort=args.sorted)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(async_main())
