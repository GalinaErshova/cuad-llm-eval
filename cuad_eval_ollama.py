#!/usr/bin/env python3
"""
CUAD Legal Information Extraction — адаптировано под Ollama (CPU).

Адаптация: локальный Ollama (Windows + WSL, Intel Arc/CPU).

Запуск:
    python cuad_eval_ollama.py

Что делает:
    1. Загружает датасет theatticusproject/cuad-qa
    2. Формирует подвыборку из N_EXAMPLES примеров
    3. Для КАЖДОЙ модели из MODEL_CONFIGS:
       - отправляет prompt в Ollama
       - парсит JSON-ответ
       - считает метрики (Exact Match, Precision, Recall, F1)
    4. Сохраняет результаты в CSV + выводит сводку
"""

import gc
import json
import os
import random
import re
import time
import urllib.request
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from datasets import load_dataset
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

# ====================================================================
# КОНФИГУРАЦИЯ
# ====================================================================

# Ollama endpoint (Windows → WSL localhost проброс)
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"

# Режим запуска
N_EXAMPLES = 2  # кол-во примеров для теста (было 2)
SEED = 42
MAX_CONTEXT_CHARS = 2000
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.0

# Три модели из оригинального ноутбука (адаптированы под Ollama)
MODEL_CONFIGS = [
    {
        "model_alias": "qwen2.5_7b",
        "ollama_model": "qwen2.5:7b",
        "quantization": "Q4_K_M (Ollama)",
    },
    {
        "model_alias": "llama3.1_8b",
        "ollama_model": "llama3.1:8b",
        "quantization": "Q4_K_M (Ollama)",
    },
    {
        "model_alias": "llama3.2_3b",
        "ollama_model": "llama3.2:3b",
        "quantization": "Q4_K_M (Ollama)",
    },
]

# Папка результатов
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# ====================================================================
# ФУНКЦИИ РАБОТЫ С ПАМЯТЬЮ
# ====================================================================


def get_ram_gb():
    """RAM, занятая текущим Python-процессом (RSS)."""
    return psutil.Process(os.getpid()).memory_info().rss / 1024**3


def print_memory(prefix=""):
    print(f"{prefix} RAM: {get_ram_gb():.2f} GB")


# ====================================================================
# ФУНКЦИИ ДЛЯ ПРОМПТОВ
# ====================================================================


def make_answer_centered_context(context, gold_answer, max_chars=MAX_CONTEXT_CHARS):
    """Укорачивает контекст вокруг gold_answer."""
    if not isinstance(context, str):
        return ""
    context = context.strip()
    gold_answer = str(gold_answer).strip() if gold_answer else ""

    if len(context) <= max_chars:
        return context
    if not gold_answer:
        return context[:max_chars]

    answer_pos = context.find(gold_answer)
    if answer_pos == -1:
        answer_pos = context.lower().find(gold_answer.lower())

    if answer_pos != -1:
        center = answer_pos + len(gold_answer) // 2
        start = max(0, center - max_chars // 2)
        end = start + max_chars
        if end > len(context):
            end = len(context)
            start = max(0, end - max_chars)
        return context[start:end]

    return context[:max_chars]


def build_prompt(question, context):
    """Создаёт prompt для LLM — идентично оригинальному ноутбуку."""
    prompt = f"""
You are a legal information extraction system.

Your task is to answer the question using ONLY the contract text below.

Return ONLY valid JSON with the following schema:
{{
  "answer": ""
}}

Rules:
- Return exactly one JSON object.
- Do not write "Answer:" before JSON.
- Do not use markdown.
- Do not include explanations before or after JSON.
- The answer must be copied exactly from the contract text.
- Do not summarize or paraphrase.
- Do not describe why the answer is relevant.
- Return the minimal exact text span or clause that directly answers the question.
- If several related clauses are present, return only the clause that best matches the question details.
- Do not include section numbers unless they are part of the answer span.
- Do not return only section numbers, clause numbers, or references.
- Return the full text span, not a reference to a section.
- If the answer is not present in the contract text, return {{"answer": ""}}.
- The output must be valid JSON only.


Question:
{question}

Contract text:
\"\"\"
{context}
\"\"\"
""".strip()
    return prompt


# ====================================================================
# ВЫЗОВ OLLAMA
# ====================================================================


def call_ollama(model_name, prompt, max_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE):
    """
    Отправляет prompt в Ollama и возвращает (generated_text, input_tokens, output_tokens).
    """
    payload = json.dumps(
        {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
    ).encode("utf-8")

    req = urllib.request.Request(
        OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"}
    )

    with urllib.request.urlopen(req, timeout=600) as resp:
        result = json.loads(resp.read().decode("utf-8"))

    generated = result.get("response", "").strip()
    input_tokens = result.get("prompt_eval_count", 0)
    output_tokens = result.get("eval_count", 0)

    return generated, input_tokens, output_tokens


# ====================================================================
# ПАРСИНГ ОТВЕТА
# ====================================================================


def extract_json_from_text(text):
    """Пытается извлечь JSON из ответа модели."""
    if not isinstance(text, str):
        return None

    raw = text.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    raw = re.sub(r"^\s*Answer\s*:\s*", "", raw, flags=re.IGNORECASE)

    # Попытка 1: весь текст как JSON
    try:
        return json.loads(raw)
    except Exception:
        pass

    # Попытка 2: от первой { до последней }
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(raw[start : end + 1])
        except Exception:
            pass

    return None


def parse_model_answer(generated_text):
    """Извлекает поле answer из JSON-ответа."""
    parsed = extract_json_from_text(generated_text)
    if parsed is None:
        return "", False
    answer = parsed.get("answer", "")
    if answer is None:
        answer = ""
    if isinstance(answer, list):
        answer = "; ".join([str(x) for x in answer])
    return str(answer).strip(), True


# ====================================================================
# МЕТРИКИ
# ====================================================================


def normalize_answer(text):
    """Нормализация текста для сравнения."""
    if text is None:
        return ""
    text = str(text).lower().strip()
    text = re.sub(r"[\n\r\t]", " ", text)
    text = re.sub(r"[^a-z0-9а-яё$€£.,%/-]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def token_set(text):
    norm = normalize_answer(text)
    return norm.split() if norm else []


def compute_em_precision_recall_f1(prediction, gold):
    """Метрики Exact Match, Precision, Recall, F1."""
    pred_norm = normalize_answer(prediction)
    gold_norm = normalize_answer(gold)

    exact_match = int(pred_norm == gold_norm)

    pred_tokens = token_set(prediction)
    gold_tokens = token_set(gold)

    if len(pred_tokens) == 0 and len(gold_tokens) == 0:
        return {"exact_match": 1, "precision": 1.0, "recall": 1.0, "f1": 1.0}
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return {"exact_match": exact_match, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    pred_counter = {}
    gold_counter = {}
    for t in pred_tokens:
        pred_counter[t] = pred_counter.get(t, 0) + 1
    for t in gold_tokens:
        gold_counter[t] = gold_counter.get(t, 0) + 1

    common = 0
    for t, c in pred_counter.items():
        common += min(c, gold_counter.get(t, 0))

    precision = common / len(pred_tokens) if pred_tokens else 0.0
    recall = common / len(gold_tokens) if gold_tokens else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "exact_match": exact_match,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def clean_pred_answer(answer):
    """Убирает номера разделов/пунктов из начала ответа."""
    if answer is None:
        return ""
    answer = str(answer).strip()
    answer = re.sub(
        r"^(Section|Clause|Article)\s+\d+(\.\d+)*\.?\s*",
        "",
        answer,
        flags=re.IGNORECASE,
    )
    answer = re.sub(r"^\d+(\.\d+)*\.?\s+", "", answer)
    return answer.strip()


# ====================================================================
# ПОДГОТОВКА ДАННЫХ
# ====================================================================


def extract_first_gold_answer(answers):
    if answers is None:
        return ""
    if isinstance(answers, dict):
        texts = answers.get("text", [])
        if texts is None:
            return ""
        for t in texts:
            if isinstance(t, str) and t.strip():
                return t.strip()
        return ""
    if isinstance(answers, str):
        return answers.strip()
    return ""


def prepare_sample(df_source, n_examples, seed, max_context_chars):
    """Формирует подвыборку с укороченным контекстом."""
    df_sample = df_source.sample(
        n=min(n_examples, len(df_source)), random_state=seed
    ).reset_index(drop=True)

    df_sample["context_short"] = df_sample.apply(
        lambda row: make_answer_centered_context(
            context=row["context"],
            gold_answer=row["gold_answer"],
            max_chars=max_context_chars,
        ),
        axis=1,
    )

    df_sample["gold_in_context_short"] = df_sample.apply(
        lambda row: (
            str(row["gold_answer"]).strip().lower() in str(row["context_short"]).lower()
        ),
        axis=1,
    )

    print(f"Sample: {df_sample.shape[0]} rows")
    print("Gold in context_short:")
    print(df_sample["gold_in_context_short"].value_counts(dropna=False))
    return df_sample


# ====================================================================
# ЗАПУСК ЭКСПЕРИМЕНТА ДЛЯ ОДНОЙ МОДЕЛИ
# ====================================================================


def run_experiment_for_model(model_config, df_input):
    """Запускает inference и evaluation для одной модели через Ollama."""
    model_alias = model_config["model_alias"]
    ollama_model = model_config["ollama_model"]

    print("=" * 80)
    print(f"MODEL: {model_alias}  |  Ollama: {ollama_model}")
    print(f"Quantization: {model_config['quantization']}")
    print("=" * 80)

    print_memory("Before inference")

    # Готовим промпты
    prompts = []
    for _, row in df_input.iterrows():
        prompt = build_prompt(question=row["question"], context=row["context_short"])
        prompts.append(prompt)

    all_generated_texts = []
    all_pred_answers = []
    all_is_valid_json = []
    total_input_tokens = 0
    total_output_tokens = 0

    start_time = time.perf_counter()

    for prompt in tqdm(prompts, desc=f"  {model_alias}"):
        try:
            generated, in_tok, out_tok = call_ollama(ollama_model, prompt)
            all_generated_texts.append(generated)
            total_input_tokens += in_tok
            total_output_tokens += out_tok

            pred_answer, is_valid = parse_model_answer(generated)
            all_pred_answers.append(pred_answer)
            all_is_valid_json.append(is_valid)
        except Exception as e:
            print(f"  ERROR: {e}")
            all_generated_texts.append(f"ERROR: {e}")
            all_pred_answers.append("")
            all_is_valid_json.append(False)

    elapsed = time.perf_counter() - start_time

    # Считаем метрики для каждого примера
    metrics_list = []
    for i in range(len(df_input)):
        gold = df_input.iloc[i]["gold_answer"]
        pred_clean = clean_pred_answer(all_pred_answers[i])
        m = compute_em_precision_recall_f1(pred_clean, gold)
        metrics_list.append(m)

    # Собираем predictions DataFrame
    predictions_df = df_input[["question", "gold_answer"]].copy()
    predictions_df["model_alias"] = model_alias
    predictions_df["pred_answer"] = all_pred_answers
    predictions_df["pred_answer_clean"] = [
        clean_pred_answer(a) for a in all_pred_answers
    ]
    predictions_df["is_valid_json"] = all_is_valid_json
    predictions_df["generated_text"] = all_generated_texts
    for key in ["exact_match", "precision", "recall", "f1"]:
        predictions_df[key] = [m[key] for m in metrics_list]

    # Summary
    n = len(predictions_df)
    summary = {
        "model_alias": model_alias,
        "ollama_model": ollama_model,
        "quantization": model_config["quantization"],
        "n_examples": n,
        "elapsed_sec": round(elapsed, 1),
        "avg_latency_sec": round(elapsed / n, 2) if n > 0 else None,
        "examples_per_sec": round(n / elapsed, 4) if elapsed > 0 else None,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_input_tokens + total_output_tokens,
        "tokens_per_sec": round((total_input_tokens + total_output_tokens) / elapsed, 2)
        if elapsed > 0
        else None,
        "output_tokens_per_sec": round(total_output_tokens / elapsed, 2)
        if elapsed > 0
        else None,
        "exact_match": round(predictions_df["exact_match"].mean(), 4),
        "precision": round(predictions_df["precision"].mean(), 4),
        "recall": round(predictions_df["recall"].mean(), 4),
        "f1": round(predictions_df["f1"].mean(), 4),
        "invalid_json_rate": round(1 - predictions_df["is_valid_json"].mean(), 4),
        "ram_after_gb": round(get_ram_gb(), 3),
    }

    print(f"\n  Elapsed: {elapsed:.1f}s  |  Avg latency: {elapsed / n:.1f}s/example")
    print(
        f"  Tokens/sec: {summary['tokens_per_sec']}  |  F1: {summary['f1']}  |  Invalid JSON: {summary['invalid_json_rate']}"
    )
    print_memory("  After inference")

    return predictions_df, summary


# ====================================================================
# MAIN
# ====================================================================


def main():
    print("=" * 80)
    print("CUAD Legal IE — Ollama Local Benchmark")
    print(f"Models: {[m['model_alias'] for m in MODEL_CONFIGS]}")
    print(f"N_EXAMPLES: {N_EXAMPLES}  |  MAX_CONTEXT_CHARS: {MAX_CONTEXT_CHARS}")
    print(f"Ollama: {OLLAMA_URL}")
    print("=" * 80)

    # --- 1. Загрузка датасета ---
    print("\n[1/4] Loading dataset theatticusproject/cuad-qa ...")
    dataset = load_dataset("theatticusproject/cuad-qa", trust_remote_code=True)
    print(f"  Splits: {list(dataset.keys())}")
    print(f"  Train rows: {len(dataset['train'])}")

    # --- 2. Подготовка данных ---
    print("\n[2/4] Preparing data ...")
    df_all = dataset["train"].to_pandas()
    df_all["gold_answer"] = df_all["answers"].apply(extract_first_gold_answer)
    df_all = df_all[df_all["context"].notna() & df_all["question"].notna()]
    df_non_empty = df_all[df_all["gold_answer"].str.len() > 0].copy()
    print(f"  Rows with gold answer: {len(df_non_empty)}")

    random.seed(SEED)
    np.random.seed(SEED)

    df_sample = prepare_sample(df_non_empty, N_EXAMPLES, SEED, MAX_CONTEXT_CHARS)

    # --- 3. Запуск экспериментов ---
    print(f"\n[3/4] Running {len(MODEL_CONFIGS)} models ...")
    print("=" * 80)

    all_predictions = []
    all_summaries = []

    overall_start = time.perf_counter()

    for model_config in MODEL_CONFIGS:
        preds, summary = run_experiment_for_model(model_config, df_sample)
        all_predictions.append(preds)
        all_summaries.append(summary)

        # Сохраняем сразу после каждой модели (на случай сбоя)
        alias = model_config["model_alias"]
        preds.to_csv(
            RESULTS_DIR / f"predictions_{alias}_{N_EXAMPLES}ex.csv", index=False
        )
        pd.DataFrame([summary]).to_csv(
            RESULTS_DIR / f"summary_{alias}_{N_EXAMPLES}ex.csv", index=False
        )

        print()  # пустая строка между моделями

    overall_elapsed = time.perf_counter() - overall_start

    # --- 4. Итоговая таблица ---
    print("\n[4/4] Final results")
    print("=" * 80)

    summary_df = pd.DataFrame(all_summaries)

    # Колонки для вывода
    display_cols = [
        "model_alias",
        "quantization",
        "n_examples",
        "elapsed_sec",
        "avg_latency_sec",
        "tokens_per_sec",
        "exact_match",
        "precision",
        "recall",
        "f1",
        "invalid_json_rate",
        "ram_after_gb",
    ]
    existing = [c for c in display_cols if c in summary_df.columns]

    print(f"\nTotal time: {overall_elapsed:.1f}s ({overall_elapsed / 60:.1f} min)")
    print(f"Results saved to: {RESULTS_DIR}/")
    print()
    print(summary_df[existing].to_string(index=False))

    # Сохраняем сводную таблицу
    summary_df.to_csv(RESULTS_DIR / f"summary_all_{N_EXAMPLES}ex.csv", index=False)

    # Прогноз для 50 примеров
    print("\n" + "=" * 80)
    print("ПРОГНОЗ ДЛЯ 50 ПРИМЕРОВ")
    print("=" * 80)
    for _, row in summary_df.iterrows():
        est_50 = row["avg_latency_sec"] * 50
        print(f"  {row['model_alias']}: ~{est_50:.0f}s = {est_50 / 60:.1f} мин")
    total_est = summary_df["avg_latency_sec"].sum() * 50
    print(f"  ВСЕ 3 модели × 50 примеров: ~{total_est:.0f}s = {total_est / 60:.1f} мин")


if __name__ == "__main__":
    main()
