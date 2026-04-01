from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Запускает 4.py для каждого вопроса из golden_questions.json "
            "и сохраняет ответ модели в каждом элементе."
        )
    )
    parser.add_argument(
        "--input",
        default="golden_questions.json",
        help="Путь к golden_questions.json",
    )
    parser.add_argument(
        "--output",
        default="golden_questions_with_model_answers.json",
        help="Куда сохранить результат",
    )
    parser.add_argument(
        "--script-4",
        default="4.py",
        help="Путь к скрипту 4.py",
    )
    parser.add_argument(
        "--python",
        dest="python_bin",
        default=sys.executable,
        help="Какой интерпретатор Python использовать для вызова 4.py",
    )
    parser.add_argument("--index-id", help="ID индекса для 4.py")
    parser.add_argument("--index-name", help="Имя индекса для 4.py")
    parser.add_argument("--dataset-dir", help="dataset_dir для 4.py")
    parser.add_argument(
        "--registry",
        default="index_registry.json",
        help="Путь к registry для 4.py",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Прокидывается в 4.py",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Прокидывается в 4.py",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=None,
        help="Прокидывается в 4.py",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Прокидывается в 4.py",
    )
    parser.add_argument(
        "--show-context",
        action="store_true",
        help="Прокидывается в 4.py; полезно для отладки",
    )
    return parser.parse_args()


def load_questions(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("golden_questions.json должен быть списком объектов")

    for i, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Элемент #{i} должен быть объектом")
        if "question" not in item:
            raise ValueError(f"В элементе #{i} нет поля 'question'")

    return data


def build_base_command(args: argparse.Namespace) -> list[str]:
    selectors = [bool(args.index_id), bool(args.index_name), bool(args.dataset_dir)]
    if sum(selectors) != 1:
        raise ValueError(
            "Нужно указать ровно один из параметров: "
            "--index-id, --index-name или --dataset-dir"
        )

    cmd = [args.python_bin, args.script_4]

    if args.index_id:
        cmd.extend(["--index-id", args.index_id])
    if args.index_name:
        cmd.extend(["--index-name", args.index_name])
    if args.dataset_dir:
        cmd.extend(["--dataset-dir", args.dataset_dir])

    if args.registry:
        cmd.extend(["--registry", args.registry])
    if args.temperature is not None:
        cmd.extend(["--temperature", str(args.temperature)])
    if args.k is not None:
        cmd.extend(["--k", str(args.k)])
    if args.score_threshold is not None:
        cmd.extend(["--score-threshold", str(args.score_threshold)])
    if args.model is not None:
        cmd.extend(["--model", args.model])
    if args.show_context:
        cmd.append("--show-context")

    return cmd


def run_question(base_cmd: list[str], question: str) -> tuple[bool, str, str]:
    cmd = list(base_cmd)
    cmd.append(question)

    completed = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    stdout = (completed.stdout or "").strip()
    stderr = (completed.stderr or "").strip()

    if completed.returncode != 0:
        return False, stdout, stderr

    return True, stdout, stderr


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    script_4_path = Path(args.script_4)

    if not script_4_path.exists():
        raise FileNotFoundError(f"Скрипт 4.py не найден: {script_4_path}")

    questions = load_questions(input_path)
    base_cmd = build_base_command(args)

    results: list[dict[str, Any]] = []

    total = len(questions)
    for idx, item in enumerate(questions, start=1):
        question = str(item["question"])
        print(f"[{idx}/{total}] {question}")

        ok, stdout, stderr = run_question(base_cmd, question)

        new_item = dict(item)
        if ok:
            new_item["model_answer"] = stdout
            new_item["status"] = "ok"
        else:
            new_item["model_answer"] = None
            new_item["status"] = "error"
            new_item["error"] = stderr or stdout or "Неизвестная ошибка"

        if stderr:
            new_item["stderr"] = stderr

        results.append(new_item)

    output_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    ok_count = sum(1 for x in results if x.get("status") == "ok")
    err_count = len(results) - ok_count

    print()
    print(f"Готово. Результат сохранен в: {output_path}")
    print(f"Успешно: {ok_count}")
    print(f"С ошибкой: {err_count}")


if __name__ == "__main__":
    main()
