from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
    answer_similarity,
)
from ragas import evaluate
from datasets import Dataset
import json
import os
from dotenv import load_dotenv

load_dotenv()
LOG_PATH = "logs/ragas_eval_log.jsonl"

def load_ragas_samples(path):
    with open(path, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f]
    return Dataset.from_list(samples)

def main():
    dataset = load_ragas_samples(LOG_PATH)
    result = evaluate(
        dataset,
        metrics=[
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
            answer_similarity,
        ],
    )
    print("=== 📊 RAGAS評価スコア ===")
    print(result.to_pandas())

if __name__ == "__main__":
    main()
