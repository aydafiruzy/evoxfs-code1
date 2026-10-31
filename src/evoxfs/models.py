"""Classifiers and test-set metrics.

- Evaluation classifiers: XGBoost, LightGBM, CatBoost.
- Metrics on the held-out TEST set: Accuracy, weighted F1,
  macro one-vs-rest AUC, macro AUPRC.
- "Fitness XGBoost" is the XGBoost used INSIDE the fitness function.
- Each model uses one thread; parallelism is across runs (joblib).
"""
from __future__ import annotations

import numpy as np
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from xgboost import XGBClassifier

from .data import Data


def make_model(name: str, params: dict, seed: int):
    base = "XGBoost" if name == "Fitness XGBoost" else name
    if base == "XGBoost":
        return XGBClassifier(**params, random_state=seed, n_jobs=1, verbosity=0)
    if base == "LightGBM":
        return LGBMClassifier(**params, random_state=seed, n_jobs=1, verbosity=-1)
    if base == "CatBoost":
        return CatBoostClassifier(**params, random_seed=seed, thread_count=1, verbose=False)
    raise ValueError(f"Unknown model: {name}")


def test_scores(model, data: Data, X_tr: np.ndarray, X_te: np.ndarray) -> dict:
    """Fit on the training part, report the 4 metrics on the test part."""
    model.fit(X_tr, data.y_train)
    pred = np.asarray(model.predict(X_te)).reshape(-1)
    proba = model.predict_proba(X_te)
    if data.binary:
        auc = roc_auc_score(data.y_test, proba[:, 1])
        auprc = average_precision_score(data.y_test, proba[:, 1])
    else:
        y_bin = label_binarize(data.y_test, classes=data.classes)
        auc = roc_auc_score(y_bin, proba, average="macro", multi_class="ovr")
        auprc = average_precision_score(y_bin, proba, average="macro")
    return {"Accuracy": accuracy_score(data.y_test, pred),
            "F1": f1_score(data.y_test, pred, average="weighted"),
            "AUC": auc,
            "AUPRC": auprc}


def evaluate_subset(mask, model_params: dict, data: Data, eval_models: list, seed: int) -> list[dict]:
    """Evaluate a feature subset (boolean mask) with every evaluation classifier."""
    idx = np.flatnonzero(np.asarray(mask, dtype=bool))
    X_tr, X_te = data.X_train, data.X_test
    rows = []
    for name in eval_models:
        scores = test_scores(make_model(name, model_params[name], seed), data, X_tr[:, idx], X_te[:, idx])
        rows.append({"Classifier": name, "Features": int(idx.size), **scores})
    return rows
