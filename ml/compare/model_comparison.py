"""Compare classical ML models vs CNN on CIFAR-10.

Trains Logistic Regression, SVM (linear), Random Forest, and a small CNN,
then produces a comparison table, bar chart, and k-fold cross-validation
results.  All reported on the same 70/15/15 split used by train.py.

Usage:
    python -m ml.compare.model_comparison
    python -m ml.compare.model_comparison --n-samples 5000 --cv-folds 5
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path("ml/image_model/models")
OUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_PATH = OUT_DIR / "model_comparison.json"
CHART_PATH  = OUT_DIR / "model_comparison.png"
CV_PATH     = OUT_DIR / "model_comparison_cv.png"


# ── Data ──────────────────────────────────────────────────────────────────────

def load_cifar10_flat(n_samples: int | None = None):
    """Return flattened (N, 3072) uint8 arrays with 70/15/15 split."""
    import tensorflow as tf

    (x_tr, y_tr), (x_te, y_te) = tf.keras.datasets.cifar10.load_data()
    x_all = np.concatenate([x_tr, x_te], axis=0)
    y_all = np.concatenate([y_tr, y_te], axis=0).ravel()

    rng = np.random.default_rng(42)
    idx = rng.permutation(len(x_all))
    x_all, y_all = x_all[idx], y_all[idx]

    if n_samples:
        x_all, y_all = x_all[:n_samples], y_all[:n_samples]

    n = len(x_all)
    n_train = int(n * 0.70)
    n_val   = int(n * 0.15)

    x_train = x_all[:n_train].reshape(n_train, -1).astype(np.float32) / 255.0
    y_train = y_all[:n_train]
    x_val   = x_all[n_train:n_train + n_val].reshape(n_val, -1).astype(np.float32) / 255.0
    y_val   = y_all[n_train:n_train + n_val]
    x_test  = x_all[n_train + n_val:].reshape(-1, 3072).astype(np.float32) / 255.0
    y_test  = y_all[n_train + n_val:]

    return x_train, y_train, x_val, y_val, x_test, y_test


def load_cifar10_images(n_samples: int | None = None):
    """Return (H,W,C) images with 70/15/15 split for CNN."""
    import tensorflow as tf

    (x_tr, y_tr), (x_te, y_te) = tf.keras.datasets.cifar10.load_data()
    x_all = np.concatenate([x_tr, x_te], axis=0).astype(np.float32)
    y_all = np.concatenate([y_tr, y_te], axis=0)

    rng = np.random.default_rng(42)
    idx = rng.permutation(len(x_all))
    x_all, y_all = x_all[idx], y_all[idx]

    if n_samples:
        x_all, y_all = x_all[:n_samples], y_all[:n_samples]

    n = len(x_all)
    n_train = int(n * 0.70)
    n_val   = int(n * 0.15)

    return (
        x_all[:n_train],            y_all[:n_train],
        x_all[n_train:n_train + n_val], y_all[n_train:n_train + n_val],
        x_all[n_train + n_val:],    y_all[n_train + n_val:],
    )


# ── Classical models ───────────────────────────────────────────────────────────

def train_logistic_regression(x_train, y_train, x_test, y_test):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score

    print("  Training Logistic Regression… (saga solver, up to 1000 iter)")
    t0 = time.perf_counter()
    clf = LogisticRegression(max_iter=1000, C=0.1, solver="saga", n_jobs=-1, random_state=42, verbose=1)
    clf.fit(x_train, y_train)
    train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred = clf.predict(x_test)
    infer_time_per_sample = (time.perf_counter() - t0) / len(x_test)

    return {
        "model": "Logistic Regression",
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "f1_macro": round(float(f1_score(y_test, y_pred, average="macro")), 4),
        "train_time_s": round(train_time, 2),
        "infer_ms_per_sample": round(infer_time_per_sample * 1000, 3),
        "clf": clf,
    }


def train_svm(x_train, y_train, x_test, y_test):
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV

    print("  Training SVM (Linear)… (LinearSVC + calibration, 3 folds — може тривати 5-10 хв)")
    t0 = time.perf_counter()
    base = LinearSVC(C=0.1, max_iter=2000, dual=False, random_state=42, verbose=1)
    clf = CalibratedClassifierCV(base, cv=3)
    clf.fit(x_train, y_train)
    train_time = time.perf_counter() - t0
    print(f"  SVM done in {train_time:.1f}s")

    t0 = time.perf_counter()
    y_pred = clf.predict(x_test)
    infer_time_per_sample = (time.perf_counter() - t0) / len(x_test)

    return {
        "model": "SVM (Linear)",
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "f1_macro": round(float(f1_score(y_test, y_pred, average="macro")), 4),
        "train_time_s": round(train_time, 2),
        "infer_ms_per_sample": round(infer_time_per_sample * 1000, 3),
        "clf": clf,
    }


def train_random_forest(x_train, y_train, x_test, y_test):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, f1_score

    print("  Training Random Forest… (200 trees, verbose кожні 10)")
    t0 = time.perf_counter()
    clf = RandomForestClassifier(n_estimators=200, max_depth=20, n_jobs=-1, random_state=42, verbose=2)
    clf.fit(x_train, y_train)
    train_time = time.perf_counter() - t0
    print(f"  Random Forest done in {train_time:.1f}s")

    t0 = time.perf_counter()
    y_pred = clf.predict(x_test)
    infer_time_per_sample = (time.perf_counter() - t0) / len(x_test)

    return {
        "model": "Random Forest",
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "f1_macro": round(float(f1_score(y_test, y_pred, average="macro")), 4),
        "train_time_s": round(train_time, 2),
        "infer_ms_per_sample": round(infer_time_per_sample * 1000, 3),
        "clf": clf,
    }


# ── Small CNN for comparison ───────────────────────────────────────────────────

def train_small_cnn(x_train, y_train, x_val, y_val, x_test, y_test, epochs: int = 15):
    import tensorflow as tf
    from sklearn.metrics import accuracy_score, f1_score

    print("  Training Small CNN (baseline)…")

    def build():
        m = tf.keras.Sequential([
            tf.keras.layers.Rescaling(1.0 / 255, input_shape=(32, 32, 3)),
            tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(10, activation="softmax"),
        ])
        m.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        return m

    t0 = time.perf_counter()
    model = build()
    model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=64,
        validation_data=(x_val, y_val),
        verbose=0,
    )
    train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    preds = model.predict(x_test, verbose=0)
    infer_time_per_sample = (time.perf_counter() - t0) / len(x_test)

    y_pred = np.argmax(preds, axis=1)
    y_true = y_test.ravel()

    return {
        "model": "CNN (3-conv baseline)",
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro")), 4),
        "train_time_s": round(train_time, 2),
        "infer_ms_per_sample": round(infer_time_per_sample * 1000, 4),
    }


# ── Cross-validation ───────────────────────────────────────────────────────────

def cross_validate_models(x_cv, y_cv, folds: int = 5):
    """K-fold CV on classical models (subset for speed)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score

    models = {
        "Logistic Regression": LogisticRegression(max_iter=200, C=0.1, solver="saga", n_jobs=-1, random_state=42),
        "SVM (Linear)":        LinearSVC(C=0.1, max_iter=1000, dual=False, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=100, max_depth=15, n_jobs=-1, random_state=42),
    }

    cv_results = {}
    for name, clf in models.items():
        print(f"  Cross-validating {name} ({folds}-fold)…")
        t0 = time.perf_counter()
        scores = cross_val_score(clf, x_cv, y_cv, cv=folds, scoring="accuracy", n_jobs=-1, verbose=2)
        elapsed = time.perf_counter() - t0
        cv_results[name] = {
            "mean": round(float(scores.mean()), 4),
            "std": round(float(scores.std()), 4),
            "scores": [round(float(s), 4) for s in scores],
        }
        print(f"    {name}: {scores.mean():.4f} ± {scores.std():.4f}  ({elapsed:.1f}s)")

    return cv_results


# ── Plots ──────────────────────────────────────────────────────────────────────

def save_comparison_chart(results: list[dict]) -> None:
    names   = [r["model"] for r in results]
    accs    = [r["accuracy"] * 100 for r in results]
    f1s     = [r["f1_macro"] * 100 for r in results]

    x = np.arange(len(names))
    w = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - w / 2, accs, w, label="Accuracy %", color="#4C72B0")
    bars2 = ax.bar(x + w / 2, f1s,  w, label="F1-macro %", color="#DD8452")

    ax.set_ylabel("Score (%)")
    ax.set_title("Model Comparison — CIFAR-10")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylim(0, 100)
    ax.axhline(90, color="red", linestyle="--", linewidth=0.8, label="90% target")
    ax.legend()

    for bar in (*bars1, *bars2):
        ax.annotate(
            f"{bar.get_height():.1f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3), textcoords="offset points",
            ha="center", va="bottom", fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(CHART_PATH, dpi=120)
    plt.close()
    print(f"Saved → {CHART_PATH}")


def save_cv_chart(cv_results: dict) -> None:
    names  = list(cv_results.keys())
    means  = [cv_results[n]["mean"] * 100 for n in names]
    stds   = [cv_results[n]["std"] * 100 for n in names]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(names, means, xerr=stds, color="#4C72B0", capsize=5)
    ax.set_xlabel("Accuracy % (mean ± std)")
    ax.set_title(f"{len(next(iter(cv_results.values()))['scores'])}-Fold Cross-Validation — CIFAR-10")
    ax.axvline(90, color="red", linestyle="--", linewidth=0.8, label="90% target")
    ax.legend()
    plt.tight_layout()
    plt.savefig(CV_PATH, dpi=120)
    plt.close()
    print(f"Saved → {CV_PATH}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=None,
                        help="Limit total samples (default: all 60 000). Use 10000 for a quick run.")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--cv-samples", type=int, default=10000,
                        help="Samples used for k-fold CV (subset for speed).")
    parser.add_argument("--cnn-epochs", type=int, default=15)
    args = parser.parse_args()

    print("Loading data…")
    x_train_f, y_train_f, x_val_f, y_val_f, x_test_f, y_test_f = load_cifar10_flat(args.n_samples)
    x_train_i, y_train_i, x_val_i, y_val_i, x_test_i, y_test_i = load_cifar10_images(args.n_samples)
    print(f"Train: {len(x_train_f)} | Val: {len(x_val_f)} | Test: {len(x_test_f)}")

    print("\n── Training classical models ─────────────────────────────────────")
    results = []
    results.append(train_logistic_regression(x_train_f, y_train_f, x_test_f, y_test_f))
    results.append(train_svm(x_train_f, y_train_f, x_test_f, y_test_f))
    results.append(train_random_forest(x_train_f, y_train_f, x_test_f, y_test_f))
    results.append(train_small_cnn(x_train_i, y_train_i, x_val_i, y_val_i, x_test_i, y_test_i, args.cnn_epochs))

    print("\n── Results ───────────────────────────────────────────────────────")
    print(f"{'Model':<28} {'Accuracy':>10} {'F1-macro':>10} {'Train (s)':>10} {'Infer (ms)':>12}")
    print("-" * 75)
    for r in results:
        print(
            f"{r['model']:<28} {r['accuracy']:>10.4f} {r['f1_macro']:>10.4f} "
            f"{r['train_time_s']:>10.1f} {r['infer_ms_per_sample']:>12.3f}"
        )

    print("\n── Cross-validation ──────────────────────────────────────────────")
    cv_n = min(args.cv_samples, len(x_train_f))
    cv_results = cross_validate_models(x_train_f[:cv_n], y_train_f[:cv_n], args.cv_folds)

    # Save report
    report = {
        "results": [{k: v for k, v in r.items() if k != "clf"} for r in results],
        "cross_validation": cv_results,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved → {REPORT_PATH}")

    save_comparison_chart(results)
    save_cv_chart(cv_results)


if __name__ == "__main__":
    main()
