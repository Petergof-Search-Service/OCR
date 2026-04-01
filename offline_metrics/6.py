from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import nltk
import pandas as pd
from bert_score import score as bertscore
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

DEFAULT_INPUT = Path("golden_questions_with_model_answers.json")
DEFAULT_OUTPUT = Path("golden_questions_metrics.json")
DEFAULT_CSV = Path("golden_questions_metrics.csv")


def download_nltk_resources() -> None:
    resources = [
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
    ]
    for resource_path, resource_name in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(resource_name)


def load_dataset(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Ожидался JSON-массив с вопросами и ответами")

    return data


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def compute_metrics(
    dataset: list[dict[str, Any]],
    bert_lang: str = "ru",
    sentence_model: str = "paraphrase-multilingual-MiniLM-L12-v2",
) -> tuple[pd.DataFrame, dict[str, float]]:
    rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    embedder = SentenceTransformer(sentence_model)

    results: list[dict[str, Any]] = []

    for item in dataset:
        question = safe_text(item.get("question"))
        correct_answer = safe_text(item.get("answer") or item.get("correct_answer"))
        model_answer = safe_text(item.get("model_answer"))
        status = safe_text(item.get("status"))

        row: dict[str, Any] = {
            "question": question,
            "correct_answer": correct_answer,
            "model_answer": model_answer,
            "status": status,
        }

        if not correct_answer or not model_answer:
            row.update(
                {
                    "bleu": None,
                    "rouge1": None,
                    "rouge2": None,
                    "rougeL": None,
                    "meteor": None,
                    "bertscore_f1": None,
                    "semantic_similarity": None,
                    "metrics_error": "Пустой correct_answer или model_answer",
                }
            )
            results.append(row)
            continue

        try:
            bleu = sentence_bleu(
                [correct_answer.split()],
                model_answer.split(),
                smoothing_function=SmoothingFunction().method1,
            )

            rouge_scores = rouge.score(correct_answer, model_answer)

            meteor = meteor_score(
                [nltk.word_tokenize(correct_answer)],
                nltk.word_tokenize(model_answer),
            )

            _, _, f1 = bertscore([model_answer], [correct_answer], lang=bert_lang, verbose=False)
            bertscore_f1 = float(f1.mean().item())

            emb1 = embedder.encode(correct_answer)
            emb2 = embedder.encode(model_answer)
            semantic_similarity = float(cosine_similarity([emb1], [emb2])[0][0])

            row.update(
                {
                    "bleu": float(bleu),
                    "rouge1": float(rouge_scores["rouge1"].fmeasure),
                    "rouge2": float(rouge_scores["rouge2"].fmeasure),
                    "rougeL": float(rouge_scores["rougeL"].fmeasure),
                    "meteor": float(meteor),
                    "bertscore_f1": bertscore_f1,
                    "semantic_similarity": semantic_similarity,
                    "metrics_error": None,
                }
            )
        except Exception as exc:
            row.update(
                {
                    "bleu": None,
                    "rouge1": None,
                    "rouge2": None,
                    "rougeL": None,
                    "meteor": None,
                    "bertscore_f1": None,
                    "semantic_similarity": None,
                    "metrics_error": str(exc),
                }
            )

        results.append(row)

    df = pd.DataFrame(results)
    mean_metrics = df.mean(numeric_only=True).to_dict()
    return df, {k: float(v) for k, v in mean_metrics.items()}


def save_outputs(df: pd.DataFrame, mean_metrics: dict[str, float], output_json: Path, output_csv: Path) -> None:
    payload = {
        "rows": df.to_dict(orient="records"),
        "mean_metrics": mean_metrics,
    }
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    df.to_csv(output_csv, index=False, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Подсчет BLEU, ROUGE, METEOR, BERTScore и semantic similarity по golden_questions_with_model_answers.json"
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="Путь к golden_questions_with_model_answers.json",
    )
    parser.add_argument(
        "--output-json",
        default=str(DEFAULT_OUTPUT),
        help="Путь к JSON-файлу с метриками",
    )
    parser.add_argument(
        "--output-csv",
        default=str(DEFAULT_CSV),
        help="Путь к CSV-файлу с метриками",
    )
    parser.add_argument(
        "--bert-lang",
        default="ru",
        help="Язык для BERTScore, по умолчанию ru",
    )
    parser.add_argument(
        "--sentence-model",
        default="paraphrase-multilingual-MiniLM-L12-v2",
        help="Модель sentence-transformers для semantic similarity",
    )
    args = parser.parse_args()

    download_nltk_resources()
    dataset = load_dataset(Path(args.input))
    df, mean_metrics = compute_metrics(
        dataset,
        bert_lang=args.bert_lang,
        sentence_model=args.sentence_model,
    )

    save_outputs(df, mean_metrics, Path(args.output_json), Path(args.output_csv))

    print(df)
    print("\nСредние значения:")
    print(pd.Series(mean_metrics))
    print(f"\nJSON сохранен в: {args.output_json}")
    print(f"CSV сохранен в: {args.output_csv}")


if __name__ == "__main__":
    main()
