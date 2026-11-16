"""Comparison with MI-MPODE (Yu et al., ESWA 2025) on its 11 UCI datasets.

Two experiments, both with 30 independent 70/30 train/test splits
(run r uses seed r for the split and for the optimiser):

A) Full-feature baselines (no feature selection):
   KNN (k=5, the paper's classifier), XGBoost, LightGBM, CatBoost
   with library defaults. Test Accuracy, Macro-F1, Balanced accuracy.

B) Feature selection under the paper's protocol:
   KNN (k=5) wrapper, population 10, 1000 fitness evaluations
   (= NP 10 x 100 iterations), fitness = 0.95*(1 - acc) + 0.05*Ns/Nf
   (alpha = 0.95, Table 2 of the paper). acc = mean 5-fold CV accuracy
   on the TRAIN part only; the 30% test part is used only for the final
   report. Methods (all use the repo's switchable DE, src/evoxfs/de.py):
     DE                    random init, F=0.5, CR=0.9
     DE + hybrid init      SHAP-guided + opposite start population
     EvoX-FS (KNN)         hybrid init + ODE jumping + self-adaptive F/CR

Usage:
  python scripts/paper_mimpode/run_uci.py --raw-dir <uci files> --out results/paper_mimpode
  (add --download the first time; needs access to archive.ics.uci.edu)
"""
from __future__ import annotations

import argparse
import os
import time
import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from evoxfs.config import load_config
from evoxfs.de import switchable_de
from evoxfs.models import make_model
from evoxfs.uci_paper import UCI_IDS, download_all, load_uci

ALPHA = 0.95
POP, MAX_EVALS, TEST_SIZE, INNER_FOLDS = 10, 1000, 0.30, 5
METHODS = {
    "DE": dict(init="random", obl=False, adaptive=False),
    "DE + hybrid init": dict(init="hybrid", obl=False, adaptive=False),
    "EvoX-FS (KNN)": dict(init="hybrid", obl=True, adaptive=True),
}


def split(y, seed):
    """Stratified 70/30 split; classes with < 2 samples share one stratum."""
    counts = np.bincount(y)
    rare = counts[y] < 2
    # pool rare classes; if the pool itself has one sample, put it with the largest class
    strata = np.where(rare, -1 if rare.sum() >= 2 else counts.argmax(), y)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=seed)
    return next(sss.split(np.zeros(len(y)), strata))


def scores(y_true, y_pred):
    return {"Accuracy": accuracy_score(y_true, y_pred),
            "Macro-F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "Balanced acc": balanced_accuracy_score(y_true, y_pred)}


def knn():
    return KNeighborsClassifier(n_neighbors=5)


def prepare(X, y, seed):
    tr, te = split(y, seed)
    scaler = MinMaxScaler().fit(X[tr])
    return scaler.transform(X[tr]), scaler.transform(X[te]), y[tr], y[te]


# ------------------------------------------------------------- A) baselines
def baseline_run(name, X, y, run):
    Xtr, Xte, ytr, yte = prepare(X, y, run)
    rows = []
    for model_name in ["KNN", "XGBoost", "LightGBM", "CatBoost"]:
        t = time.perf_counter()
        if model_name == "KNN":
            pred = knn().fit(Xtr, ytr).predict(Xte)
        else:  # tree models need labels 0..K-1 present in the training part
            enc = LabelEncoder().fit(ytr)
            model = make_model(model_name, {}, run).fit(Xtr, enc.transform(ytr))
            pred = enc.inverse_transform(np.asarray(model.predict(Xte)).reshape(-1).astype(int))
        rows.append({"Dataset": name, "Run": run, "Model": model_name, "Features": X.shape[1],
                     **scores(yte, pred), "Time (s)": time.perf_counter() - t})
    return rows


# ------------------------------------------------------------- B) feature selection
def shap_importance(Xtr, ytr, seed):
    import shap
    from sklearn.ensemble import RandomForestClassifier

    from evoxfs.shap_tools import signed_shap_for_class
    rf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=1,
                                class_weight="balanced").fit(Xtr, ytr)
    values = shap.TreeExplainer(rf).shap_values(Xtr)
    n_cls = len(rf.classes_)
    imp = np.mean([np.abs(signed_shap_for_class(values, c, Xtr.shape[1])) for c in range(n_cls)],
                  axis=0).mean(axis=0)
    return imp / imp.sum() if imp.sum() > 0 else np.full(Xtr.shape[1], 1 / Xtr.shape[1])


def make_fitness(Xtr, ytr, seed):
    folds = list(StratifiedKFold(INNER_FOLDS, shuffle=True, random_state=seed).split(Xtr, ytr))
    cache, n = {}, Xtr.shape[1]

    def fitness(vec):
        idx = np.flatnonzero(np.asarray(vec) >= 0.5)
        if idx.size == 0:
            return 1.0e6, {"acc": 0.0}
        key = idx.tobytes()
        if key not in cache:
            Xs = Xtr[:, idx]
            cache[key] = float(np.mean([knn().fit(Xs[a], ytr[a]).score(Xs[b], ytr[b])
                                        for a, b in folds]))
        acc = cache[key]
        return ALPHA * (1 - acc) + (1 - ALPHA) * idx.size / n, {"acc": acc}
    return fitness


def fs_run(name, X, y, run, cfg):
    warnings.filterwarnings("ignore")
    Xtr, Xte, ytr, yte = prepare(X, y, run)
    t = time.perf_counter()
    shap_norm = shap_importance(Xtr, ytr, run)
    shap_time = time.perf_counter() - t
    ctx = SimpleNamespace(cfg=cfg, n_features=X.shape[1], shap_norm=shap_norm)
    rows = []
    for method, spec in METHODS.items():
        res = switchable_de(ctx, make_fitness(Xtr, ytr, run), spec["init"], spec["obl"],
                            spec["adaptive"], POP, run)
        mask = res["mask"] if res["mask"].any() else np.ones(X.shape[1], bool)
        pred = knn().fit(Xtr[:, mask], ytr).predict(Xte[:, mask])
        rows.append({"Dataset": name, "Run": run, "Method": method, "Features": X.shape[1],
                     "Selected": int(mask.sum()), **scores(yte, pred),
                     "CV acc (train)": res["details"]["acc"], "Fitness": res["fitness"],
                     "Evaluations": res["n_evals"], "Generations": res["generations"],
                     "Search time (s)": res["search_time"] + (shap_time if spec["init"] != "random" else 0),
                     "history": res["history"]})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--out", default="results/paper_mimpode")
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--datasets", nargs="*", default=list(UCI_IDS))
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--skip-baselines", action="store_true")
    ap.add_argument("--skip-fs", action="store_true")
    ap.add_argument("--n-jobs", type=int, default=-1)
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    if args.download:
        download_all(args.raw_dir)
    os.makedirs(args.out, exist_ok=True)
    data = {n: load_uci(n, args.raw_dir) for n in args.datasets}

    cfg = load_config("ionosphere")  # only DE / init settings are used (dataset-independent)
    cfg.update({"max_evals": MAX_EVALS})

    runs = range(1, args.runs + 1)
    if not args.skip_baselines:
        out = Parallel(n_jobs=args.n_jobs, verbose=5)(
            delayed(baseline_run)(n, *data[n], r) for n in data for r in runs)
        pd.DataFrame([row for rows in out for row in rows]).to_csv(
            os.path.join(args.out, "baselines_runs.csv"), index=False)
    if not args.skip_fs:
        out = Parallel(n_jobs=args.n_jobs, verbose=5)(
            delayed(fs_run)(n, *data[n], r, cfg) for n in data for r in runs)
        df = pd.DataFrame([row for rows in out for row in rows])
        df.drop(columns="history").to_csv(os.path.join(args.out, "fs_runs.csv"), index=False)
        df[["Dataset", "Run", "Method", "history"]].to_pickle(os.path.join(args.out, "fs_histories.pkl"))


if __name__ == "__main__":
    main()
