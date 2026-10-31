"""SHAP importance: RandomForest proxy + TreeExplainer, on the TRAIN set only.

Global importance = mean |SHAP| over samples and classes, normalised to
sum to 1. This vector (`shap_norm`) is used in (1) the fitness SHAP term
and (2) the SHAP-guided start population.
"""
from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier

from .data import Data


@dataclass
class ShapResult:
    model: RandomForestClassifier
    explainer: object
    values_train: object
    norm: np.ndarray            # normalised global importance, sums to 1


def signed_shap_for_class(values, class_index: int, n_features: int) -> np.ndarray:
    """(samples x features) signed SHAP matrix of one class, for every SHAP output format."""
    if isinstance(values, list):
        return np.asarray(values[class_index], dtype=float)
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 2:
        return arr
    return arr[:, :, class_index] if arr.shape[1] == n_features else arr[:, class_index, :]


def compute_shap(data: Data, cfg: dict) -> ShapResult:
    rf = RandomForestClassifier(n_estimators=cfg["shap_rf_trees"], random_state=cfg["seed"],
                                n_jobs=cfg["n_jobs"], class_weight="balanced")
    rf.fit(data.X_train_df, data.y_train)
    explainer = shap.TreeExplainer(rf)
    values = explainer.shap_values(data.X_train_df)
    abs_mean = np.mean([np.abs(signed_shap_for_class(values, c, data.n_features))
                        for c in range(data.n_classes)], axis=0)
    importance = abs_mean.mean(axis=0)
    return ShapResult(rf, explainer, values, importance / importance.sum())


def top_k(cfg: dict, n_features: int) -> int:
    return max(1, int(np.ceil(cfg["shap_top_ratio"] * n_features)))


def shap_table(res: ShapResult, data: Data, cfg: dict) -> pd.DataFrame:
    table = (pd.DataFrame({"Feature": data.feature_names, "SHAP normalized": res.norm})
             .sort_values("SHAP normalized", ascending=False)
             .head(cfg["top_features_shown"]).reset_index(drop=True))
    table["Top p_h (favoured in SHAP init)"] = table.index < top_k(cfg, data.n_features)
    return table


def plot_shap(res: ShapResult, data: Data, cfg: dict) -> None:
    """Global bar plot, signed beeswarm per class, force plots of a few test samples."""
    abs_train = np.mean([np.abs(signed_shap_for_class(res.values_train, c, data.n_features))
                         for c in range(data.n_classes)], axis=0)
    plt.figure()
    shap.summary_plot(abs_train, data.X_train_df, feature_names=data.feature_names, plot_type="bar",
                      show=False, max_display=cfg["top_features_shown"])
    plt.title("Global SHAP importance (training set)")
    plt.tight_layout(); plt.show()

    classes = [1] if data.binary else range(min(data.n_classes, cfg["beeswarm_max_classes"]))
    for c in classes:
        plt.figure()
        shap.summary_plot(signed_shap_for_class(res.values_train, c, data.n_features), data.X_train_df,
                          feature_names=data.feature_names, show=False,
                          max_display=cfg["top_features_shown"])
        plt.title(f"SHAP beeswarm (signed values), class {c}")
        plt.tight_layout(); plt.show()

    explanation = res.explainer(data.X_test_df)
    values, base = np.asarray(explanation.values), np.asarray(explanation.base_values)
    for s in range(cfg["force_plot_samples"]):
        predicted = int(res.model.predict(data.X_test_df.iloc[[s]])[0])
        sample_values = values[s, :, predicted] if values.ndim == 3 else values[s]
        base_value = base[s, predicted] if base.ndim == 2 else base[s]
        shap.force_plot(base_value, sample_values, data.X_test_df.iloc[s].round(3), matplotlib=True,
                        show=False, figsize=(20, 3.5), text_rotation=15)
        plt.title(f"Test sample {s}: predicted class {predicted} (true {int(data.y_test[s])})", pad=40)
        plt.show()
