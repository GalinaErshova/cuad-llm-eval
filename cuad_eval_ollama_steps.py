#!/usr/bin/env python3
"""
CUAD Legal Information Extraction — пошаговый benchmark через Ollama.

Этот скрипт разделяет работу на независимые этапы:

    1) prepare-sample
       Один раз загружает CUAD, формирует фиксированную выборку и сохраняет ее в CSV.

    2) run-model
       Запускает одну выбранную модель на уже сохраненной выборке и сохраняет:
       - predictions_<model>_<N>ex.csv
       - summary_<model>_<N>ex.csv

    3) analyze
       Собирает все summary/predictions, которые уже успели сохраниться,
       и формирует итоговые файлы:
       - summary_all_<N>ex.csv
       - predictions_all_<N>ex.csv
       - report_notes_<N>ex.md

Примеры запуска:

    # 1. Один раз подготовить выборку
    python cuad_eval_ollama_steps.py prepare-sample --n-examples 20

    # 2. Запустить модели по одной, можно в разные дни
    python cuad_eval_ollama_steps.py run-model --model-alias llama3.2_3b --n-examples 20
    python cuad_eval_ollama_steps.py run-model --model-alias qwen2.5_7b --n-examples 20
    python cuad_eval_ollama_steps.py run-model --model-alias llama3.1_8b --n-examples 20

    # 3. Собрать итоговый отчет из уже сохраненных файлов
    python cuad_eval_ollama_steps.py analyze --n-examples 20

Важно для ДЗ:
    - Ollama-модели обычно запускаются в quantized-виде, например Q4_K_M.
      Поэтому это quantized local benchmark, а не полноценное сравнение full precision vs quantized.
    - Классический batch inference в /api/generate Ollama не используется.
      Здесь реализован sequential local inference benchmark: warm-up, JSON mode,
      ограничение контекста, latency/throughput/tokens/sec, сохранение промежуточных CSV.
"""

import argparse
import json
import os
import random
import re
import socket
import time
import urllib.error
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
# КОНФИГУРАЦИЯ ПО УМОЛЧАНИЮ
# ====================================================================

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
RESULTS_DIR = Path("results")

DEFAULT_N_EXAMPLES = 20
DEFAULT_SEED = 42
DEFAULT_MAX_CONTEXT_CHARS = 2000
DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TIMEOUT_SEC = 900
DEFAULT_WARMUP_MAX_TOKENS = 32

# По вашим промежуточным результатам llama3.2_3b была самой практичной для CPU/Ollama,
# поэтому она оставлена первой: ее удобно запустить и получить результат быстрее.
MODEL_CONFIGS = {
    "llama3.2_3b": {
        "model_alias": "llama3.2_3b",
        "ollama_model": "llama3.2:3b",
        "quantization": "Q4_K_M (Ollama)",
    },
    "qwen2.5_7b": {
        "model_alias": "qwen2.5_7b",
        "ollama_model": "qwen2.5:7b",
        "quantization": "Q4_K_M (Ollama)",
    },
    "llama3.1_8b": {
        "model_alias": "llama3.1_8b",
        "ollama_model": "llama3.1:8b",
        "quantization": "Q4_K_M (Ollama)",
    },
}

# ====================================================================
# ПУТИ К ФАЙЛАМ
# ====================================================================


def ensure_results_dir(results_dir=RESULTS_DIR):
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def sample_path(n_examples, seed, results_dir=RESULTS_DIR):
    return Path(results_dir) / f"sample_{n_examples}ex_seed{seed}.csv"


def predictions_path(model_alias, n_examples, results_dir=RESULTS_DIR):
    return Path(results_dir) / f"predictions_{model_alias}_{n_examples}ex.csv"


def summary_path(model_alias, n_examples, results_dir=RESULTS_DIR):
    return Path(results_dir) / f"summary_{model_alias}_{n_examples}ex.csv"


def summary_all_path(n_examples, results_dir=RESULTS_DIR):
    return Path(results_dir) / f"summary_all_{n_examples}ex.csv"


def predictions_all_path(n_examples, results_dir=RESULTS_DIR):
    return Path(results_dir) / f"predictions_all_{n_examples}ex.csv"


def report_path(n_examples, results_dir=RESULTS_DIR):
    return Path(results_dir) / f"report_notes_{n_examples}ex.md"


# ====================================================================
# ПАМЯТЬ
# ====================================================================


def get_process_ram_gb():
    """RAM, занятая текущим Python-процессом."""
    return psutil.Process(os.getpid()).memory_info().rss / 1024**3


def get_system_ram_used_gb():
    """Общая занятая RAM по системе. Ollama работает отдельным процессом, поэтому это полезнее Python RSS."""
    return psutil.virtual_memory().used / 1024**3


def print_memory(prefix=""):
    print(
        f"{prefix} Python RAM: {get_process_ram_gb():.2f} GB | "
        f"System used RAM: {get_system_ram_used_gb():.2f} GB"
    )


# ====================================================================
# ДАННЫЕ И PROMPT
# ====================================================================


def extract_first_gold_answer(answers):
    if answers is None:
        return ""
    if isinstance(answers, dict):
        texts = answers.get("text", [])
        if texts is None:
            return ""
        for text in texts:
            if isinstance(text, str) and text.strip():
                return text.strip()
        return ""
    if isinstance(answers, str):
        return answers.strip()
    return ""


def make_answer_centered_context(
    context, gold_answer, max_chars=DEFAULT_MAX_CONTEXT_CHARS
):
    """
    Укорачивает contract context вокруг gold_answer.

    """
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
    """Prompt для строгого legal span extraction в JSON."""
    return f"""
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
- If the answer contains quotation marks, escape them correctly inside JSON.
- The JSON must be parseable by Python json.loads().
- If the answer is not present in the contract text, return {{"answer": ""}}.
- The output must be valid JSON only.

Question:
{question}

Contract text:
\"\"\"
{context}
\"\"\"
""".strip()


# ====================================================================
# OLLAMA
# ====================================================================


def call_ollama(
    model_name,
    prompt,
    max_tokens=DEFAULT_MAX_NEW_TOKENS,
    temperature=DEFAULT_TEMPERATURE,
    use_json_mode=True,
    timeout_sec=DEFAULT_TIMEOUT_SEC,
    ollama_url=OLLAMA_URL,
):
    """Отправляет один prompt в Ollama и возвращает generated_text, input_tokens, output_tokens."""
    payload_dict = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }

    # JSON mode снижает долю invalid JSON, но его можно отключить флагом --no-json-mode.
    if use_json_mode:
        payload_dict["format"] = "json"

    payload = json.dumps(payload_dict).encode("utf-8")
    req = urllib.request.Request(
        ollama_url,
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        result = json.loads(resp.read().decode("utf-8"))

    generated = result.get("response", "").strip()
    input_tokens = result.get("prompt_eval_count", 0)
    output_tokens = result.get("eval_count", 0)
    return generated, input_tokens, output_tokens


def classify_error(exc):
    """Короткая классификация технических ошибок инференса."""
    if isinstance(exc, TimeoutError) or isinstance(exc, socket.timeout):
        return "timeout"
    if isinstance(exc, urllib.error.HTTPError):
        return f"http_error_{exc.code}"
    if isinstance(exc, urllib.error.URLError):
        reason = getattr(exc, "reason", None)
        if isinstance(reason, socket.timeout):
            return "timeout"
        return "url_error"
    return type(exc).__name__


# ====================================================================
# ПАРСИНГ И МЕТРИКИ
# ====================================================================


def extract_json_from_text(text):
    """Пытается извлечь JSON из ответа модели."""
    if not isinstance(text, str):
        return None

    raw = text.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    raw = re.sub(r"^\s*Answer\s*:\s*", "", raw, flags=re.IGNORECASE)

    try:
        return json.loads(raw)
    except Exception:
        pass

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
        answer = "; ".join(str(x) for x in answer)
    return str(answer).strip(), True


def normalize_answer(text):
    """Нормализация для token-level сравнения."""
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
    """Exact Match, Precision, Recall, F1 для span extraction."""
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
    for token in pred_tokens:
        pred_counter[token] = pred_counter.get(token, 0) + 1
    for token in gold_tokens:
        gold_counter[token] = gold_counter.get(token, 0) + 1

    common = 0
    for token, count in pred_counter.items():
        common += min(count, gold_counter.get(token, 0))

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
    """Убирает номера разделов/пунктов из начала ответа, если модель их добавила."""
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
# ЭТАП 1: ПОДГОТОВКА ВЫБОРКИ
# ====================================================================


def prepare_sample_file(
    n_examples, seed, max_context_chars, results_dir=RESULTS_DIR, overwrite=False
):
    """Создает sample CSV. Если файл уже есть, повторно датасет не загружает."""
    results_dir = ensure_results_dir(results_dir)
    out_path = sample_path(n_examples, seed, results_dir)

    if out_path.exists() and not overwrite:
        print(f"Sample already exists: {out_path}")
        print("Use --overwrite-sample if you want to recreate it.")
        return out_path

    print("=" * 80)
    print("[prepare-sample] Loading dataset theatticusproject/cuad-qa ...")
    dataset = load_dataset("theatticusproject/cuad-qa", trust_remote_code=True)
    print(f"Splits: {list(dataset.keys())}")
    print(f"Train rows: {len(dataset['train'])}")

    df_all = dataset["train"].to_pandas()
    df_all["gold_answer"] = df_all["answers"].apply(extract_first_gold_answer)
    df_all = df_all[df_all["context"].notna() & df_all["question"].notna()].copy()
    df_non_empty = df_all[df_all["gold_answer"].str.len() > 0].copy()
    print(f"Rows with non-empty gold answer: {len(df_non_empty)}")

    random.seed(seed)
    np.random.seed(seed)

    df_sample = df_non_empty.sample(
        n=min(n_examples, len(df_non_empty)),
        random_state=seed,
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

    cols = [
        "question",
        "gold_answer",
        "context_short",
        "gold_in_context_short",
    ]
    df_sample[cols].to_csv(out_path, index=False)

    print(f"Sample saved: {out_path}")
    print(f"Sample size: {len(df_sample)}")
    print("Gold in context_short:")
    print(df_sample["gold_in_context_short"].value_counts(dropna=False))
    return out_path


def load_sample(n_examples, seed, results_dir=RESULTS_DIR):
    path = sample_path(n_examples, seed, results_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"Sample file not found: {path}\n"
            f"Run first: python cuad_eval_ollama_steps.py prepare-sample --n-examples {n_examples} --seed {seed}"
        )
    df = pd.read_csv(path).fillna("")
    required = {"question", "gold_answer", "context_short", "gold_in_context_short"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Sample file is missing columns: {sorted(missing)}")
    return df


# ====================================================================
# ЭТАП 2: ЗАПУСК ОДНОЙ МОДЕЛИ
# ====================================================================


def warmup_model(
    model_alias,
    ollama_model,
    prompt,
    max_tokens,
    use_json_mode,
    timeout_sec,
    ollama_url,
):
    """Один короткий warm-up запрос до основного benchmark."""
    print(f"Warm-up for {model_alias} ...")
    start = time.perf_counter()
    try:
        _ = call_ollama(
            ollama_model,
            prompt,
            max_tokens=max_tokens,
            use_json_mode=use_json_mode,
            timeout_sec=timeout_sec,
            ollama_url=ollama_url,
        )
        elapsed = time.perf_counter() - start
        print(f"Warm-up completed: {elapsed:.1f}s")
        return {"warmup_ok": True, "warmup_sec": round(elapsed, 1), "warmup_error": ""}
    except Exception as exc:
        elapsed = time.perf_counter() - start
        err = classify_error(exc)
        print(f"Warm-up failed: {err} ({exc})")
        return {
            "warmup_ok": False,
            "warmup_sec": round(elapsed, 1),
            "warmup_error": err,
        }


def run_model_experiment(
    model_alias,
    n_examples,
    seed,
    max_new_tokens,
    temperature,
    use_json_mode,
    do_warmup,
    warmup_max_tokens,
    timeout_sec,
    results_dir=RESULTS_DIR,
    ollama_url=OLLAMA_URL,
    overwrite_model=False,
):
    """Запускает одну модель на уже сохраненной выборке."""
    results_dir = ensure_results_dir(results_dir)

    if model_alias not in MODEL_CONFIGS:
        known = ", ".join(MODEL_CONFIGS.keys())
        raise ValueError(f"Unknown model_alias: {model_alias}. Available: {known}")

    pred_path = predictions_path(model_alias, n_examples, results_dir)
    sum_path = summary_path(model_alias, n_examples, results_dir)
    if pred_path.exists() and sum_path.exists() and not overwrite_model:
        print(f"Results already exist for {model_alias}:")
        print(f"  {pred_path}")
        print(f"  {sum_path}")
        print("Use --overwrite-model if you want to rerun this model.")
        return pred_path, sum_path

    df_input = load_sample(n_examples, seed, results_dir)
    model_config = MODEL_CONFIGS[model_alias]
    ollama_model = model_config["ollama_model"]

    print("=" * 80)
    print(f"[run-model] MODEL: {model_alias} | Ollama: {ollama_model}")
    print(f"Quantization: {model_config['quantization']}")
    print(f"N examples: {len(df_input)}")
    print(f"JSON mode: {use_json_mode} | Warm-up: {do_warmup}")
    print("=" * 80)
    print_memory("Before inference")

    # Prompt cache: подготовка prompt не попадает в основной замер inference.
    prompts = [
        build_prompt(row["question"], row["context_short"])
        for _, row in df_input.iterrows()
    ]

    if do_warmup:
        warmup_info = warmup_model(
            model_alias=model_alias,
            ollama_model=ollama_model,
            prompt=prompts[0],
            max_tokens=warmup_max_tokens,
            use_json_mode=use_json_mode,
            timeout_sec=timeout_sec,
            ollama_url=ollama_url,
        )
    else:
        warmup_info = {
            "warmup_ok": False,
            "warmup_sec": 0.0,
            "warmup_error": "disabled",
        }

    all_generated_texts = []
    all_pred_answers = []
    all_is_valid_json = []
    all_error_type = []
    total_input_tokens = 0
    total_output_tokens = 0

    start_time = time.perf_counter()

    for prompt in tqdm(prompts, desc=f"{model_alias}"):
        try:
            generated, in_tok, out_tok = call_ollama(
                ollama_model,
                prompt,
                max_tokens=max_new_tokens,
                temperature=temperature,
                use_json_mode=use_json_mode,
                timeout_sec=timeout_sec,
                ollama_url=ollama_url,
            )
            total_input_tokens += in_tok
            total_output_tokens += out_tok

            pred_answer, is_valid = parse_model_answer(generated)
            all_generated_texts.append(generated)
            all_pred_answers.append(pred_answer)
            all_is_valid_json.append(is_valid)
            all_error_type.append("")
        except Exception as exc:
            err = classify_error(exc)
            print(f"ERROR: {err} ({exc})")
            all_generated_texts.append(f"ERROR: {err}: {exc}")
            all_pred_answers.append("")
            all_is_valid_json.append(False)
            all_error_type.append(err)

    elapsed = time.perf_counter() - start_time

    metrics_list = []
    for i in range(len(df_input)):
        gold = df_input.iloc[i]["gold_answer"]
        pred_clean = clean_pred_answer(all_pred_answers[i])
        metrics_list.append(compute_em_precision_recall_f1(pred_clean, gold))

    predictions_df = df_input[
        ["question", "gold_answer", "gold_in_context_short"]
    ].copy()
    predictions_df["model_alias"] = model_alias
    predictions_df["pred_answer"] = all_pred_answers
    predictions_df["pred_answer_clean"] = [
        clean_pred_answer(ans) for ans in all_pred_answers
    ]
    predictions_df["is_valid_json"] = all_is_valid_json
    predictions_df["error_type"] = all_error_type
    predictions_df["generated_text"] = all_generated_texts
    for key in ["exact_match", "precision", "recall", "f1"]:
        predictions_df[key] = [m[key] for m in metrics_list]

    n = len(predictions_df)
    technical_error_rate = (predictions_df["error_type"] != "").mean()
    invalid_json_rate = (
        (~predictions_df["is_valid_json"]) & (predictions_df["error_type"] == "")
    ).mean()

    summary = {
        "model_alias": model_alias,
        "ollama_model": ollama_model,
        "quantization": model_config["quantization"],
        "n_examples": n,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "json_mode": use_json_mode,
        "warmup_ok": warmup_info["warmup_ok"],
        "warmup_sec": warmup_info["warmup_sec"],
        "warmup_error": warmup_info["warmup_error"],
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
        "invalid_json_rate": round(invalid_json_rate, 4),
        "technical_error_rate": round(technical_error_rate, 4),
        "python_ram_after_gb": round(get_process_ram_gb(), 3),
        "system_ram_used_after_gb": round(get_system_ram_used_gb(), 3),
    }

    predictions_df.to_csv(pred_path, index=False)
    pd.DataFrame([summary]).to_csv(sum_path, index=False)

    print("\nModel finished")
    print(f"Predictions saved: {pred_path}")
    print(f"Summary saved: {sum_path}")
    print(
        f"Elapsed without warm-up: {elapsed:.1f}s | "
        f"Avg latency: {summary['avg_latency_sec']}s | "
        f"Tokens/sec: {summary['tokens_per_sec']} | "
        f"F1: {summary['f1']} | "
        f"Invalid JSON: {summary['invalid_json_rate']} | "
        f"Technical errors: {summary['technical_error_rate']}"
    )
    print_memory("After inference")

    return pred_path, sum_path


def analyze_results(n_examples, results_dir=RESULTS_DIR, model_aliases=None):
    """Собирает итоговый отчет из уже существующих summary/predictions файлов."""
    results_dir = ensure_results_dir(results_dir)
    if model_aliases is None:
        model_aliases = list(MODEL_CONFIGS.keys())

    summaries = []
    predictions = []
    included_models = []
    missing_models = []

    for model_alias in model_aliases:
        sum_path = summary_path(model_alias, n_examples, results_dir)
        pred_path = predictions_path(model_alias, n_examples, results_dir)

        if not sum_path.exists() or not pred_path.exists():
            missing_models.append(model_alias)
            continue

        summaries.append(pd.read_csv(sum_path))
        predictions.append(pd.read_csv(pred_path))
        included_models.append(model_alias)

    if not summaries:
        raise FileNotFoundError(
            "No model results found. Run at least one command like:\n"
            f"python cuad_eval_ollama_steps.py run-model --model-alias llama3.2_3b --n-examples {n_examples}"
        )

    summary_df = pd.concat(summaries, ignore_index=True)
    predictions_df = pd.concat(predictions, ignore_index=True)

    # Для удобства сортируем: сначала лучшее качество, потом скорость.
    summary_df = summary_df.sort_values(
        ["f1", "tokens_per_sec"], ascending=[False, False]
    ).reset_index(drop=True)

    out_summary = summary_all_path(n_examples, results_dir)
    out_predictions = predictions_all_path(n_examples, results_dir)
    out_report = report_path(n_examples, results_dir)

    summary_df.to_csv(out_summary, index=False)
    predictions_df.to_csv(out_predictions, index=False)

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
        "technical_error_rate",
        "python_ram_after_gb",
        "system_ram_used_after_gb",
    ]
    existing = [col for col in display_cols if col in summary_df.columns]

    print("=" * 80)
    print("[analyze] Final summary")
    print("=" * 80)
    print(summary_df[existing].to_string(index=False))
    print()
    print(f"Included models: {included_models}")
    if missing_models:
        print(f"Missing models: {missing_models}")
    print(f"Summary saved: {out_summary}")
    print(f"Predictions saved: {out_predictions}")
    print(f"Report notes saved: {out_report}")

    return out_summary, out_predictions, out_report


# ====================================================================
# CLI
# ====================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="CUAD Legal IE benchmark through Ollama, split into independent steps."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    common_parent = argparse.ArgumentParser(add_help=False)
    common_parent.add_argument("--n-examples", type=int, default=DEFAULT_N_EXAMPLES)
    common_parent.add_argument("--seed", type=int, default=DEFAULT_SEED)
    common_parent.add_argument("--results-dir", type=str, default=str(RESULTS_DIR))

    p_sample = subparsers.add_parser("prepare-sample", parents=[common_parent])
    p_sample.add_argument(
        "--max-context-chars", type=int, default=DEFAULT_MAX_CONTEXT_CHARS
    )
    p_sample.add_argument("--overwrite-sample", action="store_true")

    p_run = subparsers.add_parser("run-model", parents=[common_parent])
    p_run.add_argument(
        "--model-alias", required=True, choices=list(MODEL_CONFIGS.keys())
    )
    p_run.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    p_run.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    p_run.add_argument("--timeout-sec", type=int, default=DEFAULT_TIMEOUT_SEC)
    p_run.add_argument(
        "--warmup-max-tokens", type=int, default=DEFAULT_WARMUP_MAX_TOKENS
    )
    p_run.add_argument("--no-warmup", action="store_true")
    p_run.add_argument("--no-json-mode", action="store_true")
    p_run.add_argument("--ollama-url", type=str, default=OLLAMA_URL)
    p_run.add_argument("--overwrite-model", action="store_true")

    p_analyze = subparsers.add_parser("analyze", parents=[common_parent])
    p_analyze.add_argument(
        "--model-aliases",
        nargs="*",
        default=None,
        choices=list(MODEL_CONFIGS.keys()),
        help="Optional list of models to include. By default all known models are checked.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    results_dir = ensure_results_dir(args.results_dir)

    if args.command == "prepare-sample":
        prepare_sample_file(
            n_examples=args.n_examples,
            seed=args.seed,
            max_context_chars=args.max_context_chars,
            results_dir=results_dir,
            overwrite=args.overwrite_sample,
        )

    elif args.command == "run-model":
        # Если sample еще не создан, не запускаем датасет неявно: так проще контролировать этапы.
        run_model_experiment(
            model_alias=args.model_alias,
            n_examples=args.n_examples,
            seed=args.seed,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            use_json_mode=not args.no_json_mode,
            do_warmup=not args.no_warmup,
            warmup_max_tokens=args.warmup_max_tokens,
            timeout_sec=args.timeout_sec,
            results_dir=results_dir,
            ollama_url=args.ollama_url,
            overwrite_model=args.overwrite_model,
        )

    elif args.command == "analyze":
        analyze_results(
            n_examples=args.n_examples,
            results_dir=results_dir,
            model_aliases=args.model_aliases,
        )


if __name__ == "__main__":
    main()
