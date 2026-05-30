# Legal Information Extraction на CUAD с локальными LLM

## Цель

Information Extraction из юридических контрактов на датасете **CUAD** через локальный Ollama.

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

```bash
cd cuad-llm-eval
python cuad_eval_ollama.py
```

Требуется:
- Ollama запущен на `127.0.0.1:11434`
- Модели загружены: `ollama pull qwen2.5:7b llama3.1:8b llama3.2:3b`
- Python 3.11+ с пакетами: `datasets`, `pandas`, `numpy`, `psutil`, `tqdm`

## Структура

```
cuad-llm-eval/
├── cuad_eval_ollama.py   # основной скрипт
├── results/              # результаты прогонов (CSV)
└── README.md
```

## Источник

Адаптировано из Colab-ноутбука OTUS_HW18 (Legal IE на CUAD с Hugging Face transformers).
