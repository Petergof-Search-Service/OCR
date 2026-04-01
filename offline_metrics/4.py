from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from config import FOLDER_ID, YANDEX_API_KEY

BASE_URL = "https://ai.api.cloud.yandex.net/v1"
INDEX_REGISTRY = Path("index_registry.json")
DEFAULT_MODEL = os.getenv("YANDEX_MODEL", "yandexgpt-lite/latest")
DEFAULT_PROMPT = (
    "Ты ассистируешь исследователю. "
    "Отвечай строго только на основе текста из блока КОНТЕКСТ. "
    "Если ответа в контексте нет, так и скажи. "
    "В конце ответа обязательно перечисли файлы и страницы, на которые ты опирался."
)


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


def load_registry(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_index_id(
    *,
    index_id: str | None,
    index_name: str | None,
    dataset_dir: str | None,
    registry_path: Path,
) -> str:
    if index_id:
        return index_id

    records = load_registry(registry_path)
    if index_name:
        for record in reversed(records):
            if record.get("name") == index_name:
                return str(record["vector_store_id"])
        raise ValueError(f"Индекс с именем '{index_name}' не найден в {registry_path}")

    if dataset_dir:
        dataset_abs = str(Path(dataset_dir).resolve())
        for record in reversed(records):
            if record.get("dataset_dir") == dataset_abs:
                return str(record["vector_store_id"])
        raise ValueError(
            f"Для dataset '{dataset_abs}' индекс не найден в {registry_path}. Сначала создайте его через 2.py"
        )

    raise ValueError("Нужно передать --index-id, --index-name или --dataset-dir")


async def get_answer(
    question: str,
    vector_store_id: str,
    temp: float = 0.2,
    k: int = 30,
    score_threshold: float = 0.0,
    prompt: str | None = None,
    model: str = DEFAULT_MODEL,
) -> tuple[str, str] | str:
    if prompt is None:
        prompt = DEFAULT_PROMPT

    client = make_client()

    search_result = await client.vector_stores.search(
        vector_store_id=vector_store_id,
        query=question,
    )

    hits = [
        hit
        for hit in list(search_result.data or [])
        if getattr(hit, "score", None) is not None and hit.score >= score_threshold
    ]
    hits.sort(key=lambda item: item.score, reverse=True)
    hits = hits[:k]

    if not hits:
        return "В базе нет релевантной информации по этому вопросу."

    context_parts: list[str] = []
    for i, hit in enumerate(hits, 1):
        text = hit.content[0].text if getattr(hit, "content", None) else ""
        context_parts.append(
            f"Источник {i} (score={hit.score:.4f}, файл={hit.filename}):\n{text}"
        )
    context = "\n\n".join(context_parts)

    response = await client.responses.create(
        model=f"gpt://{FOLDER_ID}/{model}",
        instructions=prompt,
        input=f"КОНТЕКСТ:\n{context}\n\nВОПРОС:\n{question}",
        temperature=temp,
        store=False,
    )

    return response.output_text, context


async def async_main() -> None:
    parser = argparse.ArgumentParser(description="Получение ответа по готовому индексу")
    parser.add_argument("question", help="Вопрос к индексу")
    parser.add_argument("--index-id", help="ID индекса")
    parser.add_argument("--index-name", help="Имя индекса из index_registry.json")
    parser.add_argument("--dataset-dir", help="Путь к dataset, для которого ранее создавался индекс")
    parser.add_argument("--registry", default=str(INDEX_REGISTRY), help="Путь к index_registry.json")
    parser.add_argument("--temperature", type=float, default=0.2, help="Температура генерации")
    parser.add_argument("--k", type=int, default=10, help="Сколько чанков брать в контекст")
    parser.add_argument("--score-threshold", type=float, default=0.0, help="Минимальный score чанка")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Имя модели в Yandex Cloud")
    parser.add_argument("--show-context", action="store_true", help="Показать собранный контекст")
    args = parser.parse_args()

    resolved_index_id = resolve_index_id(
        index_id=args.index_id,
        index_name=args.index_name,
        dataset_dir=args.dataset_dir,
        registry_path=Path(args.registry),
    )

    result = await get_answer(
        question=args.question,
        vector_store_id=resolved_index_id,
        temp=args.temperature,
        k=args.k,
        score_threshold=args.score_threshold,
        model=args.model,
    )

    if isinstance(result, tuple):
        answer, context = result
        print(answer)
        if args.show_context:
            print("\nКОНТЕКСТ:\n")
            print(context)
    else:
        print(result)


if __name__ == "__main__":
    asyncio.run(async_main())
