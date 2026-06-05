# Legal Information Extraction на CUAD с локальными LLM

## Цель

Извлечение сущностей и событий из набора текстов из юридических контрактов на датасете **CUAD** через локальный Ollama.

Датасет [`theatticusproject/cuad-qa`](https://huggingface.co/datasets/theatticusproject/cuad-qa) в формате Question Answering:
- `context` — фрагмент юридического контракта
- `question` — вопрос о юридически значимом поле
- `answers` — эталонный ответ / span из контракта

## Модели

Локальные LLM через Ollama (CPU, Windows):

| Модель | Размер |
|--------|--------|
| `qwen2.5:7b` | 4.7 GB |
| `llama3.1:8b` | 4.9 GB |
| `llama3.2:3b` | 2.0 GB |

## Метрики

- Exact Match
- Precision, Recall, F1
- Скорость: tokens/sec, examples/sec, avg latency
- RAM после инференса
- Invalid JSON rate

## Запуск

Актуальный скрипт проекта:
- `cuad_eval_ollama_steps.py`

Пошаговый запуск в Git Bash:

1. Перейти в проект
```bash
cd /c/Users/Galina/projects/cuad-llm-eval
```

2. Создать виртуальное окружение
```bash
uv venv .venv
```

3. Установить зависимости
```bash
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
```

4. Подготовить новую выборку на 20 примерах
```bash
.venv/Scripts/python.exe cuad_eval_ollama_steps.py prepare-sample --n-examples 20 --results-dir results_restart20
```

5. Запустить модели по одной
```bash
.venv/Scripts/python.exe cuad_eval_ollama_steps.py run-model --model-alias llama3.2_3b --n-examples 20 --results-dir results_restart20
.venv/Scripts/python.exe cuad_eval_ollama_steps.py run-model --model-alias qwen2.5_7b --n-examples 20 --results-dir results_restart20
.venv/Scripts/python.exe cuad_eval_ollama_steps.py run-model --model-alias llama3.1_8b --n-examples 20 --results-dir results_restart20
```

6. Собрать итоговый анализ
```bash
.venv/Scripts/python.exe cuad_eval_ollama_steps.py analyze --n-examples 20 --results-dir results_restart20
```

Быстрая проверка, что скрипт запускается:
```bash
cd /c/Users/Galina/projects/cuad-llm-eval && .venv/Scripts/python.exe cuad_eval_ollama_steps.py --help
```

Требуется:
- Ollama запущен на `127.0.0.1:11434`
- Модели загружены: `qwen2.5:7b`, `llama3.1:8b`, `llama3.2:3b`
- `uv` установлен
- Python-зависимости ставятся из `requirements.txt`

## Дополнительный скрипт : full precision и batching

Что он показывает:
- сравнение `fp32` vs `dynamic int8 quantized`
- сравнение `batch_size=1` vs `batch_size=4`
- те же базовые метрики качества и скорости: Precision / Recall / F1 / tokens/sec / examples/sec / RAM
- по умолчанию используется маленькая CPU-friendly модель `google/flan-t5-small`

Как запустить:

1. Установить дополнительные зависимости
```bash
uv pip install --python .venv/Scripts/python.exe torch transformers sentencepiece
```

2. Запустить эксперимент на полном подготовленном наборе
```bash
.venv/Scripts/python.exe cuad_eval_smallmodel_fp_batch.py --sample-csv results_restart20/sample_20ex_seed42.csv --out-dir results_smallmodel_fp_batch --batch-sizes 1 4
```

3. Посмотреть итоговую таблицу
```bash
cat results_smallmodel_fp_batch/summary_all.csv
```

4. При необходимости открыть predictions по режимам
```bash
cat results_smallmodel_fp_batch/predictions_fp32_bs1.csv
cat results_smallmodel_fp_batch/predictions_fp32_bs4.csv
cat results_smallmodel_fp_batch/predictions_dynamic_int8_bs1.csv
cat results_smallmodel_fp_batch/predictions_dynamic_int8_bs4.csv
```

## Структура

```
cuad-llm-eval/
├── cuad_eval_ollama_steps.py  # актуальный пошаговый скрипт
├── cuad_eval_ollama_v1.py     # ранняя версия
├── requirements.txt           # зависимости Python
└── README.md
```
