#!/usr/bin/env python
"""
Бенчмарк небольшой модели для дополнительной части домашнего задания:
- full precision (FP32 на CPU) против quantized (dynamic int8)
- batch_size 1 против 4
- один и тот же sample CUAD для всех прогонов

Зачем нужен этот скрипт
-----------------------
Основной Ollama-бенчмарк в этом репозитории использует локальные quantized-модели
и последовательный inference. Этот вспомогательный скрипт даёт небольшой,
дружественный к CPU эксперимент, который помогает закрыть два недостающих пункта ДЗ:
1) сравнить quantized и full precision
2) сравнить batch_size=1 и более крупный batch processing

Модель по умолчанию
-------------------
Используется google/flan-t5-small, потому что она достаточно маленькая для локальных
экспериментов на CPU. Качество, скорее всего, будет ниже, чем у Ollama-моделей,
но цель здесь — показать методологию и собрать сопоставимые числа по
скорости/качеству/памяти.

Примеры команд
--------------
1) Установить дополнительные зависимости в существующее venv проекта:
   uv pip install --python .venv/Scripts/python.exe torch transformers sentencepiece

2) Запустить бенчмарк на полном подготовленном sample CUAD:
   .venv/Scripts/python.exe cuad_eval_smallmodel_fp_batch.py \
     --sample-csv results_restart20/sample_20ex_seed42.csv \
     --out-dir results_smallmodel_fp_batch \
     --batch-sizes 1 4

3) Прочитать итоговый summary CSV:
   results_smallmodel_fp_batch/summary_all.csv
"""

from __future__ import annotations

import argparse
import os
import re
import string
import sys
import time
from pathlib import Path
from typing import Iterable

import pandas as pd
import psutil


torch = None
AutoModelForSeq2SeqLM = None
AutoTokenizer = None


def ensure_ml_dependencies():
    global torch, AutoModelForSeq2SeqLM, AutoTokenizer
    if torch is not None:
        return
    try:
        import torch as _torch
        from transformers import AutoModelForSeq2SeqLM as _AutoModelForSeq2SeqLM, AutoTokenizer as _AutoTokenizer
    except Exception as exc:  # pragma: no cover
        message = str(exc)
        if "WinError 4551" in message or "Политика управления приложениями заблокировала этот файл" in message:
            raise RuntimeError(
                "PyTorch найден, но Windows временно заблокировал загрузку torch DLL.\n"
                "Это не ошибка отсутствующих зависимостей. Обычно помогает просто повторить запуск через 10-30 секунд,\n"
                "когда Defender / App Control заканчивает проверку файла.\n"
                "Если ошибка повторяется:\n"
                "  1) закройте Python/Git Bash\n"
                "  2) откройте Git Bash заново\n"
                "  3) повторите ту же команду\n"
                "Если не поможет, попробуйте переустановить wheel:\n"
                "  uv pip uninstall torch -y\n"
                "  uv pip install --python .venv/Scripts/python.exe torch\n"
                f"Original import error: {exc}"
            ) from exc
        raise RuntimeError(
            "Missing dependencies. Install them first:\n"
            "  uv pip install --python .venv/Scripts/python.exe torch transformers sentencepiece\n"
            f"Original import error: {exc}"
        ) from exc

    torch = _torch
    AutoModelForSeq2SeqLM = _AutoModelForSeq2SeqLM
    AutoTokenizer = _AutoTokenizer


DEFAULT_MODEL = "google/flan-t5-small"
DEFAULT_SAMPLE = Path("results_restart20/sample_20ex_seed42.csv")
DEFAULT_OUT_DIR = Path("results_smallmodel_fp_batch")
DEFAULT_MAX_INPUT_LENGTH = 768
DEFAULT_MAX_NEW_TOKENS = 96
DEFAULT_LIMIT = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark small HF model: FP32 vs dynamic int8 quantization, batch_size 1 vs N."
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--sample-csv", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help="Limit number of examples. Omit to use the full sample CSV.",
    )
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--max-input-length", type=int, default=DEFAULT_MAX_INPUT_LENGTH)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--skip-quantized", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    return parser.parse_args()


def ensure_columns(df: pd.DataFrame) -> None:
    required = {"question", "gold_answer", "context_short"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Sample CSV missing required columns: {sorted(missing)}")


def normalize_answer(text: str) -> str:
    text = "" if text is None else str(text)
    text = text.lower()
    text = text.replace("\n", " ")
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def token_list(text: str) -> list[str]:
    norm = normalize_answer(text)
    return norm.split() if norm else []


def compute_em_precision_recall_f1(prediction: str, gold: str) -> dict:
    pred_norm = normalize_answer(prediction)
    gold_norm = normalize_answer(gold)
    exact_match = 1 if pred_norm == gold_norm else 0

    pred_tokens = token_list(prediction)
    gold_tokens = token_list(gold)

    if len(pred_tokens) == 0 and len(gold_tokens) == 0:
        return {"exact_match": 1, "precision": 1.0, "recall": 1.0, "f1": 1.0}
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return {"exact_match": exact_match, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    pred_counter: dict[str, int] = {}
    gold_counter: dict[str, int] = {}
    for t in pred_tokens:
        pred_counter[t] = pred_counter.get(t, 0) + 1
    for t in gold_tokens:
        gold_counter[t] = gold_counter.get(t, 0) + 1

    common = 0
    for token, count in pred_counter.items():
        common += min(count, gold_counter.get(token, 0))

    precision = common / len(pred_tokens) if pred_tokens else 0.0
    recall = common / len(gold_tokens) if gold_tokens else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "exact_match": exact_match,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def get_process_ram_gb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)


def get_system_ram_used_gb() -> float:
    vm = psutil.virtual_memory()
    return vm.used / (1024 ** 3)


def build_prompt(question: str, context: str) -> str:
    return f"""Extract the exact answer span from the contract.

Rules:
- Copy the answer exactly from the contract when possible.
- Do not summarize.
- Do not explain.
- If there is no answer, return NONE.

Question: {question}
Contract: {context}
Answer:"""


def clean_prediction(text: str) -> str:
    text = "" if text is None else str(text)
    text = text.strip()
    if text.upper() == "NONE":
        return ""
    return text


def batch_iter(items: list[dict], batch_size: int) -> Iterable[list[dict]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def load_sample(path: Path, limit: int | None) -> pd.DataFrame:
    df = pd.read_csv(path)
    ensure_columns(df)
    if limit is not None:
        df = df.head(limit).copy().reset_index(drop=True)
    else:
        df = df.copy().reset_index(drop=True)
    return df


def load_model_pair(model_name: str):
    ensure_ml_dependencies()
    print(f"Загрузка tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("Загрузка full precision модели (FP32 на CPU)...")
    full_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    full_model.eval()

    print("Создание quantized модели (dynamic int8 для слоёв Linear)...")
    quantized_model = torch.quantization.quantize_dynamic(
        AutoModelForSeq2SeqLM.from_pretrained(model_name).eval(),
        {torch.nn.Linear},
        dtype=torch.qint8,
    )
    return tokenizer, full_model, quantized_model


def run_single_benchmark(
    *,
    tokenizer,
    model,
    model_name: str,
    precision_mode: str,
    df_input: pd.DataFrame,
    batch_size: int,
    max_input_length: int,
    max_new_tokens: int,
    temperature: float,
    out_dir: Path,
) -> tuple[pd.DataFrame, dict]:
    rows = df_input.to_dict("records")
    prompts = [build_prompt(r["question"], r["context_short"]) for r in rows]

    all_predictions: list[str] = []
    batch_latencies: list[float] = []
    total_input_tokens = 0
    total_output_tokens = 0

    start_all = time.perf_counter()
    with torch.no_grad():
        for batch_idx, batch_rows in enumerate(batch_iter(rows, batch_size), start=1):
            batch_prompts = prompts[(batch_idx - 1) * batch_size : (batch_idx - 1) * batch_size + len(batch_rows)]
            enc = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_input_length,
            )
            total_input_tokens += int(enc["attention_mask"].sum().item())

            gen_start = time.perf_counter()
            outputs = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
            )
            latency = time.perf_counter() - gen_start
            batch_latencies.append(latency)

            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            all_predictions.extend(clean_prediction(t) for t in decoded)

            if tokenizer.pad_token_id is not None:
                total_output_tokens += int((outputs != tokenizer.pad_token_id).sum().item())
            else:
                total_output_tokens += int(outputs.numel())

            print(
                f"[{precision_mode} | batch={batch_size}] "
                f"батч {batch_idx}: {len(batch_rows)} примеров за {latency:.2f}s"
            )

    elapsed = time.perf_counter() - start_all

    metrics_list = []
    for i, row in enumerate(rows):
        metrics_list.append(compute_em_precision_recall_f1(all_predictions[i], row["gold_answer"]))

    pred_df = df_input[["question", "gold_answer"]].copy()
    pred_df["model_name"] = model_name
    pred_df["precision_mode"] = precision_mode
    pred_df["batch_size"] = batch_size
    pred_df["prediction"] = all_predictions
    for key in ["exact_match", "precision", "recall", "f1"]:
        pred_df[key] = [m[key] for m in metrics_list]

    summary = {
        "model_name": model_name,
        "precision_mode": precision_mode,
        "batch_size": batch_size,
        "n_examples": len(df_input),
        "elapsed_sec": round(elapsed, 2),
        "avg_batch_latency_sec": round(sum(batch_latencies) / len(batch_latencies), 2),
        "examples_per_sec": round(len(df_input) / elapsed, 4) if elapsed > 0 else None,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_input_tokens + total_output_tokens,
        "tokens_per_sec": round((total_input_tokens + total_output_tokens) / elapsed, 2) if elapsed > 0 else None,
        "exact_match": round(pred_df["exact_match"].mean(), 4),
        "precision": round(pred_df["precision"].mean(), 4),
        "recall": round(pred_df["recall"].mean(), 4),
        "f1": round(pred_df["f1"].mean(), 4),
        "python_ram_after_gb": round(get_process_ram_gb(), 3),
        "system_ram_used_after_gb": round(get_system_ram_used_gb(), 3),
    }

    pred_path = out_dir / f"predictions_{precision_mode}_bs{batch_size}.csv"
    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")
    print(f"Сохранены predictions: {pred_path}")

    return pred_df, summary


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Sample CSV: {args.sample_csv}")
    print(f"Папка вывода: {args.out_dir}")
    df_input = load_sample(args.sample_csv, args.limit)
    print(f"Загружено примеров: {len(df_input)}")

    tokenizer, full_model, quantized_model = load_model_pair(args.model_name)

    summaries: list[dict] = []

    for batch_size in args.batch_sizes:
        _, summary = run_single_benchmark(
            tokenizer=tokenizer,
            model=full_model,
            model_name=args.model_name,
            precision_mode="fp32",
            df_input=df_input,
            batch_size=batch_size,
            max_input_length=args.max_input_length,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            out_dir=args.out_dir,
        )
        summaries.append(summary)

    if not args.skip_quantized:
        for batch_size in args.batch_sizes:
            _, summary = run_single_benchmark(
                tokenizer=tokenizer,
                model=quantized_model,
                model_name=args.model_name,
                precision_mode="dynamic_int8",
                df_input=df_input,
                batch_size=batch_size,
                max_input_length=args.max_input_length,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                out_dir=args.out_dir,
            )
            summaries.append(summary)

    summary_df = pd.DataFrame(summaries)
    summary_path = args.out_dir / "summary_all.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"Сохранён summary: {summary_path}")

    print("\nГотово. Краткий просмотр:")
    print(summary_df[["precision_mode", "batch_size", "tokens_per_sec", "examples_per_sec", "precision", "recall", "f1"]])


if __name__ == "__main__":
    main()
