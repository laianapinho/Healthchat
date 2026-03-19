import json
from collections import defaultdict

FILE = "results.json"


def load_data():
    with open(FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_average_scores(data):
    scores = defaultdict(list)

    for item in data:
        model = item["model"]
        final_score = item["metrics"]["final"]
        scores[model].append(final_score)

    avg_scores = {}
    for model, values in scores.items():
        avg_scores[model] = sum(values) / len(values)

    return avg_scores


def compute_average_metrics(data):
    metrics = defaultdict(lambda: defaultdict(list))

    for item in data:
        model = item["model"]
        for key, value in item["metrics"].items():
            metrics[model][key].append(value)

    avg = {}
    for model, m in metrics.items():
        avg[model] = {
            metric: sum(values) / len(values)
            for metric, values in m.items()
        }

    return avg


def print_ranking(avg_scores):
    print("\n🏆 RANKING DOS MODELOS")
    print("=" * 40)

    ranking = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)

    for i, (model, score) in enumerate(ranking, 1):
        print(f"{i}. {model.upper()} → {score:.3f}")


def print_detailed(avg_metrics):
    print("\n📊 MÉTRICAS DETALHADAS")
    print("=" * 40)

    for model, metrics in avg_metrics.items():
        print(f"\n{model.upper()}")
        print(f"  Final: {metrics['final']:.3f}")
        print(f"  Completude: {metrics['completude']:.3f}")
        print(f"  Relevância: {metrics['relevancia']:.3f}")
        print(f"  Segurança: {metrics['seguranca']:.3f}")


def main():
    data = load_data()
    avg_scores = compute_average_scores(data)
    avg_metrics = compute_average_metrics(data)

    print_ranking(avg_scores)
    print_detailed(avg_metrics)


if __name__ == "__main__":
    main()
