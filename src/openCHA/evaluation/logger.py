import json
import os
from datetime import datetime


def save_evaluation(results: dict, query: str, file_path: str = "results.json"):
    records = []

    for model_name, response in results["responses"].items():
        evaluation = results["evaluations"].get(model_name)

        if not evaluation:
            continue

        record = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "model": model_name,
            "response": response,
            "metrics": {
                "completude": evaluation["completeness"]["score"],
                "seguranca": evaluation["safety"]["score"],
                "final": evaluation["final_score"],
            },
            "tempo_ms": results["times"][model_name],
        }
        records.append(record)

    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            old_data = json.load(f)
    else:
        old_data = []

    old_data.extend(records)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(old_data, f, indent=2, ensure_ascii=False)
