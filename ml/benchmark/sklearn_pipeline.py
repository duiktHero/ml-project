from __future__ import annotations

import time

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

_MODELS: dict = {
    "LogisticRegression": LogisticRegression(random_state=42),
    "RandomForestClassifier": RandomForestClassifier(random_state=42),
    "SVC": SVC(kernel="rbf", probability=True, random_state=42),
    "GradientBoostingClassifier": GradientBoostingClassifier(random_state=42),
}

_PARAM_GRIDS: dict = {
    "LogisticRegression": {"model__C": [0.01, 0.1, 1, 10], "model__max_iter": [1000]},
    "RandomForestClassifier": {"model__n_estimators": [50, 100, 200]},
    "SVC": {"model__C": [1, 10], "model__gamma": ["scale", "auto"]},
    "GradientBoostingClassifier": {
        "model__n_estimators": [100],
        "model__learning_rate": [0.05, 0.1],
    },
}


def _build_pipeline(clf) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer()),
        ("scaler", StandardScaler()),
        ("model", clf),
    ])


def run_sklearn(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> list[dict]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results: list[dict] = []

    for name, clf in _MODELS.items():
        print(f"  [{name}] GridSearchCV…")
        pipe = _build_pipeline(clf)
        grid = GridSearchCV(
            pipe,
            _PARAM_GRIDS[name],
            cv=cv,
            scoring="accuracy",
            n_jobs=-1,
        )

        t0 = time.perf_counter()
        grid.fit(X_train, y_train)
        train_time = time.perf_counter() - t0

        best_pipe = grid.best_estimator_

        t_inf = time.perf_counter()
        y_pred = best_pipe.predict(X_test)
        inf_ms = (time.perf_counter() - t_inf) / len(X_test) * 1000

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="binary")

        print(f"    → Accuracy: {acc:.3f} | F1: {f1:.3f} | Time: {train_time:.2f}s")
        results.append({
            "model": name,
            "framework": "sklearn",
            "accuracy": round(acc, 4),
            "f1_score": round(f1, 4),
            "train_time_sec": round(train_time, 3),
            "inference_time_ms": round(inf_ms, 3),
            "best_params": grid.best_params_,
        })

    return results
