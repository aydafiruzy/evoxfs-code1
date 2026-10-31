"""Fitness functions (lower is better).

  fitness = w_error      * (1 - F1_cv)
          + w_size       * (N_selected / N_total)
          + w_shap_loss  * (1 - SHAP_norm(selected))
          + w_redundancy * mean |Pearson corr| among selected features

- F1_cv: weighted F1 of `fitness_cv_folds`-fold stratified CV of the
  fitness XGBoost on the TRAIN set only.
- SHAP_norm(selected): SUM of the normalised SHAP importance of the
  selected features (0..1).
  NOTE: because this is a sum, every extra feature lowers the SHAP loss,
  so this term rewards LARGER subsets (see docs/REVIEW_fa.md).
- AvgAbsCorr: mean |Pearson| over all pairs of selected features, 0 if
  fewer than 2 features.
- Empty subset -> `empty_subset_penalty`.
- Identical subsets are cached within a run, but every call still counts
  against the evaluation budget.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score

from .data import Data
from .models import make_model


@dataclass
class Context:
    """Everything a single DE run needs (picklable, sent to each worker)."""
    cfg: dict
    data: Data
    shap_norm: np.ndarray
    eval_params: dict
    fitness_params: dict
    corr_train: np.ndarray = field(init=False)

    def __post_init__(self):
        corr = np.abs(np.nan_to_num(np.corrcoef(self.data.X_train, rowvar=False)))
        np.fill_diagonal(corr, 0.0)
        self.corr_train = corr
        # cache the numpy arrays once (Data builds them on every access)
        self.X_train = self.data.X_train
        self.n_features = self.data.n_features


def make_fitness_functions(ctx: Context, seed: int, weights_by_kind: dict | None = None) -> dict:
    """Return {fitness_name: f(vector) -> (value, details)} for one run (own cache, own CV seed)."""
    cfg = ctx.cfg
    weights_by_kind = weights_by_kind or cfg["fitness_weights"]
    cache = {}
    cv = StratifiedKFold(n_splits=cfg["fitness_cv_folds"], shuffle=True, random_state=seed)
    X, y = ctx.X_train, ctx.data.y_train

    def cv_error(idx):
        key = idx.tobytes()
        if key not in cache:
            scores = cross_val_score(make_model("Fitness XGBoost", ctx.fitness_params, seed),
                                     X[:, idx], y, cv=cv, scoring="f1_weighted", n_jobs=1)
            cache[key] = float(1.0 - scores.mean())
        return cache[key]

    def score(vec, kind):
        idx = np.flatnonzero(np.asarray(vec) >= cfg["threshold"])
        if idx.size == 0:
            return cfg["empty_subset_penalty"], {"f1": 0.0, "shap_norm": 0.0, "avg_corr": 0.0}
        error = cv_error(idx)
        shap_norm = float(ctx.shap_norm[idx].sum())
        avg_corr = (float(ctx.corr_train[np.ix_(idx, idx)][np.triu_indices(idx.size, 1)].mean())
                    if idx.size > 1 else 0.0)
        terms = {"error": error, "size": idx.size / ctx.n_features,
                 "shap_loss": 1.0 - shap_norm, "redundancy": avg_corr}
        w = weights_by_kind[kind]
        value = float(sum(w[k] * terms[k] for k in terms))
        return value, {"f1": 1.0 - error, "shap_norm": shap_norm, "avg_corr": avg_corr}

    return {kind: (lambda vec, k=kind: score(vec, k)) for kind in weights_by_kind}


def fitness_with_weights(ctx: Context, seed: int, weights: dict):
    """Same fitness with an arbitrary weight set (used by the extra experiments)."""
    return make_fitness_functions(ctx, seed, {"custom": weights})["custom"]


class CountingFitness:
    """Wraps a fitness function and counts every evaluation (budget control)."""
    def __init__(self, fn):
        self.fn, self.n_evals = fn, 0

    def __call__(self, vec):
        self.n_evals += 1
        return self.fn(vec)
