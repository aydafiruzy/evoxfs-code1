"""Hyperparameter tuning: library Default vs Greedy coordinate search.

- Greedy: starting from the default, ONE parameter at a time is set to the
  best of `grid_points` candidate values while all others stay fixed;
  up to `max_passes` passes, stops early if a pass brings no gain.
- Score: weighted F1 of repeated stratified K-fold CV on the TRAIN set,
  all features. The test set is NOT used for the decision.
- Rule "cv_rule": Greedy is used only if its CV gain over the default is
  larger than the default's CV standard deviation.
"""
from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import f1_score
from sklearn.model_selection import RepeatedStratifiedKFold

from .data import Data
from .models import make_model, test_scores


def candidate_values(kind, low, high, log, n_points):
    values = np.geomspace(low, high, n_points) if log else np.linspace(low, high, n_points)
    return sorted({int(round(v)) for v in values}) if kind == "int" else [float(v) for v in values]


class Tuner:
    def __init__(self, data: Data, cfg: dict):
        self.data, self.cfg = data, cfg
        t = cfg["tuning"]
        cv = RepeatedStratifiedKFold(n_splits=t["cv_folds"], n_repeats=t["cv_repeats"],
                                     random_state=cfg["seed"])
        self.folds = list(cv.split(data.X_train, data.y_train))

    def _fold_score(self, target, params, fold):
        tr, va = self.folds[fold]
        X, y = self.data.X_train, self.data.y_train
        model = make_model(target, params, self.cfg["seed"]).fit(X[tr], y[tr])
        pred = np.asarray(model.predict(X[va])).reshape(-1)
        return f1_score(y[va], pred, average="weighted")

    def cv_scores_batch(self, target, param_list):
        """CV mean/std of several parameter sets; all (set x fold) fits run in parallel."""
        n = len(self.folds)
        scores = Parallel(n_jobs=self.cfg["n_jobs"])(
            delayed(self._fold_score)(target, p, f) for p in param_list for f in range(n))
        scores = np.asarray(scores).reshape(len(param_list), n)
        return [(float(s.mean()), float(s.std())) for s in scores]

    def greedy_search(self, target):
        cfg, g = self.cfg, self.cfg["greedy_tuning"]
        current = {**cfg["default_params"][target], **cfg["fixed_params"][target]}
        start, rows = time.perf_counter(), []

        def record(param_list, results):
            for p, (mean, std) in zip(param_list, results):
                rows.append({"trial": len(rows) + 1, "cv_f1": mean, "cv_std": std,
                             "elapsed": time.perf_counter() - start, "params": dict(p)})

        first = self.cv_scores_batch(target, [current])
        record([current], first)
        best_score = first[0][0]
        for pass_id in range(1, g["max_passes"] + 1):
            improved = False
            for name, (kind, low, high, log) in cfg["search_spaces"][target].items():
                candidates = [v for v in candidate_values(kind, low, high, log, g["grid_points"])
                              if v != current.get(name)]
                param_list = [{**current, name: v} for v in candidates]
                results = self.cv_scores_batch(target, param_list)
                record(param_list, results)
                means = [m for m, _ in results]
                if means and max(means) > best_score:
                    i = int(np.argmax(means))
                    best_score, current[name], improved = means[i], candidates[i], True
                print(f"  {target:15s} | pass {pass_id} | {name:17s} -> {current.get(name, 'default')!s:>10} "
                      f"| best CV F1 {best_score:.4f} | {(time.perf_counter() - start)/60:.1f} min", flush=True)
            if not improved:
                break
        return rows


def run_tuning(data: Data, cfg: dict, plot: bool = True):
    """Return (tuning_df, tuned_params); tuned_params[model] is the chosen parameter dict."""
    tuner = Tuner(data, cfg)
    targets = list(cfg["search_spaces"])
    print("Tuning options: Default, Greedy | selection rule:", cfg["tuning_selection"],
          f"| {len(tuner.folds)} CV folds | {cfg['n_jobs']} cores")
    start = time.perf_counter()
    trials, defaults = {}, {}
    for t in targets:
        t0 = time.perf_counter()
        defaults[t] = tuner.cv_scores_batch(t, [cfg["default_params"][t]])[0]
        print(f"\n{t}: default CV F1 = {defaults[t][0]:.4f} ± {defaults[t][1]:.4f}")
        trials[t] = pd.DataFrame(tuner.greedy_search(t))
        print(f"{t}: {len(trials[t])} trials in {(time.perf_counter() - t0)/60:.1f} min")
    print(f"\nTuning finished in {(time.perf_counter() - start) / 60:.1f} min")

    rows = []
    for t in targets:
        d_mean, d_std = defaults[t]
        rows.append({"Model": t, "Option": "Default", "Trials": 1, "CV F1": d_mean, "CV std": d_std,
                     "params": cfg["default_params"][t]})
        best = trials[t].loc[trials[t]["cv_f1"].idxmax()]
        rows.append({"Model": t, "Option": "Greedy", "Trials": len(trials[t]), "CV F1": best["cv_f1"],
                     "CV std": best["cv_std"], "params": best["params"]})
    tuning_df = pd.DataFrame(rows)
    # Test F1 (all features) — information only, NOT used for the decision
    tuning_df["Test F1 (info only)"] = [
        test_scores(make_model(r["Model"], r["params"], cfg["seed"]), data, data.X_train, data.X_test)["F1"]
        for _, r in tuning_df.iterrows()]

    chosen = {}
    for target, group in tuning_df.groupby("Model", sort=False):
        default = group[group["Option"] == "Default"].iloc[0]
        greedy = group[group["Option"] == "Greedy"].iloc[0]
        if cfg["tuning_selection"] == "cv_rule":
            gain = greedy["CV F1"] - default["CV F1"]
            keep_default = gain <= 0 or (cfg["tuning"]["keep_default_if_within_std"] and gain < default["CV std"])
            chosen[target] = default if keep_default else greedy
        else:                                   # forced: "Default" or "Greedy"
            chosen[target] = group[group["Option"] == cfg["tuning_selection"]].iloc[0]
    tuning_df["Chosen"] = [chosen[r["Model"]]["Option"] == r["Option"] for _, r in tuning_df.iterrows()]
    tuned = {t: dict(chosen[t]["params"]) for t in targets}

    print("\nParameters used in the experiment:")
    for t in targets:
        shown = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in tuned[t].items()}
        print(f"  {t:16s} -> {chosen[t]['Option']:7s} {shown if shown else '(library defaults)'}")

    if plot:
        fig, axes = plt.subplots(1, len(targets), figsize=(5 * len(targets), 4))
        for ax, t in zip(np.atleast_1d(axes), targets):
            ax.plot(trials[t]["trial"], trials[t]["cv_f1"].cummax(), label="Greedy (best so far)")
            ax.axhline(defaults[t][0], linestyle="--", color="gray", label="Default")
            ax.set(title=t, xlabel="Trial", ylabel="CV F1"); ax.grid(alpha=0.3); ax.legend()
        plt.tight_layout(); plt.show()
    return tuning_df, tuned
