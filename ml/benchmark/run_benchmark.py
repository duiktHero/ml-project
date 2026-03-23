from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

RESULTS_DIR = Path("ml/benchmark/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TABULAR_DATA_DIR = Path("ml/datasets/tabular")
TABULAR_DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset(name: str) -> tuple[np.ndarray, np.ndarray]:
    if name == "breast_cancer":
        from sklearn.datasets import load_breast_cancer
        d = load_breast_cancer()
        return d.data, d.target

    if name == "heart_disease":
        import pandas as pd

        local_csv = TABULAR_DATA_DIR / "heart_disease.csv"
        url = (
            "https://archive.ics.uci.edu/ml/machine-learning-databases"
            "/heart-disease/processed.cleveland.data"
        )
        if not local_csv.exists():
            df = pd.read_csv(url, header=None, na_values="?")
            df.to_csv(local_csv, index=False, header=False)
        else:
            df = pd.read_csv(local_csv, header=None, na_values="?")
        df.dropna(inplace=True)
        X = df.iloc[:, :-1].values
        y = (df.iloc[:, -1].values > 0).astype(int)
        return X, y

    raise ValueError(f"Unknown dataset: '{name}'")


def save_chart(results: list[dict], chart_path: str) -> None:
    names = [f"{r['model']}\n({r['framework']})" for r in results]
    acc = [r["accuracy"] for r in results]
    f1 = [r["f1_score"] for r in results]
    x = np.arange(len(names))
    w = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - w / 2, acc, w, label="Accuracy", color="steelblue")
    bars2 = ax.bar(x + w / 2, f1, w, label="F1-score", color="darkorange")

    ax.set_ylim(0.8, 1.02)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_title("PyTorch vs Sklearn — Binary Classification Benchmark")
    ax.legend()
    ax.bar_label(bars1, fmt="%.3f", padding=2, fontsize=8)
    ax.bar_label(bars2, fmt="%.3f", padding=2, fontsize=8)

    plt.tight_layout()
    plt.savefig(chart_path, dpi=120)
    plt.close()
    print(f"Chart saved → {chart_path}")


def run_benchmark(
    dataset: str = "breast_cancer",
    epochs: int = 100,
) -> dict:
    print(f"\n{'='*55}")
    print(f"  Dataset: {dataset}  |  PyTorch epochs: {epochs}")
    print("="*55)

    X, y = load_dataset(dataset)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\n[Sklearn] Training…")
    from ml.benchmark.sklearn_pipeline import run_sklearn
    sklearn_results = run_sklearn(X_train, X_test, y_train, y_test)

    print("\n[PyTorch] Training TabularNet…")
    from ml.benchmark.pytorch_net import run_pytorch
    pytorch_result = run_pytorch(X_train, X_test, y_train, y_test, epochs=epochs)

    all_results = sklearn_results + [pytorch_result]

    ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    json_path = str(RESULTS_DIR / f"benchmark_{ts}.json")
    chart_path = str(RESULTS_DIR / f"comparison_{ts}.png")

    output = {
        "dataset": dataset,
        "date": datetime.now().isoformat(),
        "results": all_results,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved → {json_path}")

    save_chart(all_results, chart_path)

    return {"results": output, "json_path": json_path, "chart_path": chart_path}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="breast_cancer",
        choices=["breast_cancer", "heart_disease"],
    )
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()

    run_benchmark(dataset=args.dataset, epochs=args.epochs)
