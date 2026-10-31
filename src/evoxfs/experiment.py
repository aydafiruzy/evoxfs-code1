"""Methods, single runs, and the long checkpointed experiments.

Each method is the same DE with different switches, so every comparison
isolates ONE change (same budget, population, seeds and split):

 Main ladder:
  1 Classic DE                          : classic2 fitness, random init
  2 DE + Scenario B fitness             : + SHAP/redundancy fitness
  3 DE + Scenario B + SHAP-guided init  : + SHAP-guided start population
  4 ODE + Scenario B + hybrid SHAP init : + opposition (init + jumping)
  5 EvoX-FS                             : + self-adaptive F/CR (full method)
 Side comparisons:
  6 Classic DE + SHAP-guided init       : SHAP in the start population only
  7 EvoX-FS (Scenario A fitness)        : full method, fitness A
  8 EvoX-FS (classic2 fitness)          : full method, no SHAP term

Run r uses seed = cfg["seed"] + r - 1 for ALL methods (paired design).

Checkpointing: finished runs are saved every `checkpoint_rounds` parallel
rounds into a folder named after an MD5 fingerprint of the settings that
change results. Re-running continues from the saved runs; changing any such
setting starts a new folder automatically.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .config import EXECUTION_ONLY_KEYS, out_prefix
from .de import switchable_de
from .fitness import Context, fitness_with_weights, make_fitness_functions
from .models import evaluate_subset

METHODS = {
    "Classic DE":                          {"fitness": "classic2",   "init": "random", "obl": False, "adaptive": False, "previous": None},
    "DE + Scenario B fitness":             {"fitness": "scenario_B", "init": "random", "obl": False, "adaptive": False, "previous": "Classic DE"},
    "DE + Scenario B + SHAP-guided init":  {"fitness": "scenario_B", "init": "shap",   "obl": False, "adaptive": False, "previous": "DE + Scenario B fitness"},
    "ODE + Scenario B + hybrid SHAP init": {"fitness": "scenario_B", "init": "hybrid", "obl": True,  "adaptive": False, "previous": "DE + Scenario B + SHAP-guided init"},
    "EvoX-FS":                             {"fitness": "scenario_B", "init": "hybrid", "obl": True,  "adaptive": True,  "previous": "ODE + Scenario B + hybrid SHAP init"},
    "Classic DE + SHAP-guided init":       {"fitness": "classic2",   "init": "shap",   "obl": False, "adaptive": False, "previous": "Classic DE"},
    "EvoX-FS (Scenario A fitness)":        {"fitness": "scenario_A", "init": "hybrid", "obl": True,  "adaptive": True,  "previous": "EvoX-FS"},
    "EvoX-FS (classic2 fitness)":          {"fitness": "classic2",   "init": "hybrid", "obl": True,  "adaptive": True,  "previous": "EvoX-FS"},
}
EVOX = "EvoX-FS"
BASELINE = "Classic DE"


# ---------------------------------------------------------------- single runs
def run_job(ctx: Context, method: str, pop_size: int, run_id: int):
    seed = ctx.cfg["seed"] + run_id - 1
    spec = METHODS[method]
    result = switchable_de(ctx, make_fitness_functions(ctx, seed)[spec["fitness"]],
                           spec["init"], spec["obl"], spec["adaptive"], pop_size, seed)
    # evaluation models always use the base seed (as in the original notebooks)
    rows = evaluate_subset(result["mask"], ctx.eval_params, ctx.data, ctx.cfg["eval_models"], ctx.cfg["seed"])
    for row in rows:
        row.update({"Method": method, "Population": pop_size, "Run": run_id, "Seed": seed,
                    "Fitness function": spec["fitness"], "Fitness": result["fitness"],
                    "Evaluations": result["n_evals"], "Generations": result["generations"],
                    "OBL kept": result["obl_kept"], "CV F1 (train)": result["details"]["f1"],
                    "SHAP_norm": result["details"]["shap_norm"], "Avg_Corr": result["details"]["avg_corr"],
                    "Search time (s)": result["search_time"]})
    return rows, {"method": method, "population": pop_size, "run": run_id,
                  "mask": result["mask"], "history": result["history"]}


def sensitivity_weights(cfg: dict, value: float) -> dict:
    """Set one weight to `value`, rescale the others proportionally (sum stays 1)."""
    extra = cfg["extra"]
    base = cfg["fitness_weights"][METHODS[extra["sensitivity_method"]]["fitness"]]
    term = extra["sensitivity_term"]
    rest = {k: v for k, v in base.items() if k != term}
    scale = (1 - value) / sum(rest.values())
    weights = {k: v * scale for k, v in rest.items()}
    weights[term] = value
    return weights


def run_custom_job(ctx: Context, group, name, spec, weights, pop_size, run_id):
    seed = ctx.cfg["seed"] + run_id - 1
    result = switchable_de(ctx, fitness_with_weights(ctx, seed, weights), spec["init"], spec["obl"],
                           spec["adaptive"], pop_size, seed)
    # evaluation models always use the base seed (as in the original notebooks)
    rows = evaluate_subset(result["mask"], ctx.eval_params, ctx.data, ctx.cfg["eval_models"], ctx.cfg["seed"])
    for row in rows:
        row.update({"Group": group, "Method": name, "Population": pop_size, "Run": run_id, "Seed": seed,
                    "Weights": json.dumps({k: round(v, 4) for k, v in weights.items()}),
                    "Fitness": result["fitness"], "CV F1 (train)": result["details"]["f1"],
                    "SHAP_norm": result["details"]["shap_norm"], "Avg_Corr": result["details"]["avg_corr"],
                    "Generations": result["generations"], "Search time (s)": result["search_time"]})
    return rows, {"mask": result["mask"], "history": result["history"]}


# ---------------------------------------------------------------- fingerprints
def fingerprint(ctx: Context) -> str:
    """Same formula as the original notebooks, so old Drive checkpoints are reused."""
    cfg = {k: v for k, v in ctx.cfg.items() if k not in EXECUTION_ONLY_KEYS and k != "extra"}
    payload = {"config": cfg, "eval": ctx.eval_params, "fitness": ctx.fitness_params, "methods": METHODS}
    return hashlib.md5(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:10]


def extra_fingerprint(ctx: Context) -> str:
    return hashlib.md5(json.dumps({"main": fingerprint(ctx), "extra": ctx.cfg["extra"]},
                                  sort_keys=True, default=str).encode()).hexdigest()[:10]


# ---------------------------------------------------------------- checkpointed loop
def _run_checkpointed(ctx: Context, jobs: dict, ckpt_dir: str, call):
    """jobs: {key: args}; call(ctx, key, args) -> result. Saves part_*.pkl files, resumes from them."""
    os.makedirs(ckpt_dir, exist_ok=True)
    done = {}
    for f in sorted(glob.glob(os.path.join(ckpt_dir, "part_*.pkl"))):
        done.update(pd.read_pickle(f))
    todo = [k for k in jobs if k not in done]
    print(f"Checkpoint folder: {ckpt_dir}")
    print(f"Runs finished: {len(done)} / {len(jobs)} | remaining: {len(todo)}")

    chunk_size = ctx.cfg["n_jobs"] * ctx.cfg["checkpoint_rounds"]
    t_start = time.perf_counter()
    for c, start in enumerate(range(0, len(todo), chunk_size)):
        chunk = todo[start:start + chunk_size]
        out = Parallel(n_jobs=ctx.cfg["n_jobs"])(delayed(call)(ctx, k, jobs[k]) for k in chunk)
        part = dict(zip(chunk, out))
        pd.to_pickle(part, os.path.join(ckpt_dir, f"part_{time.strftime('%Y%m%d_%H%M%S')}_{c:03d}.pkl"))
        done.update(part)
        finished = start + len(chunk)
        elapsed = time.perf_counter() - t_start
        print(f"[{time.strftime('%H:%M:%S')}] saved {len(done)}/{len(jobs)} runs | elapsed {elapsed/3600:.2f} h | "
              f"remaining ≈ {elapsed/finished*(len(todo)-finished)/3600:.2f} h", flush=True)
    return done


def _call_main(ctx, key, _):
    return run_job(ctx, *key)


def _call_extra(ctx, key, args):
    group, name, pop, run = key
    spec, weights = args
    return run_custom_job(ctx, group, name, spec, weights, pop, run)


def run_main(ctx: Context):
    """8 methods x population sizes x n_runs. Returns (experiment_df, masks, histories)."""
    cfg = ctx.cfg
    fp = fingerprint(ctx)
    print("Experiment fingerprint:", fp)
    jobs = {(m, p, r): None for p in cfg["population_sizes"] for m in METHODS
            for r in range(1, cfg["n_runs"] + 1)}
    done = _run_checkpointed(ctx, jobs, os.path.join(cfg["results_dir"], f"checkpoints_{cfg['dataset']}_{fp}"),
                             _call_main)
    rows, masks, histories = [], {}, {}
    for job in jobs:
        r, extra = done[job]
        rows.extend(r)
        key = (extra["method"], extra["population"])
        masks.setdefault(key, []).append(extra["mask"])
        histories.setdefault(key, []).append(extra["history"])
    experiment_df = pd.DataFrame(rows)
    path = out_prefix(cfg) + "all_runs.csv"
    experiment_df.to_csv(path, index=False)
    print(f"\nAll {len(jobs)} runs available | rows: {len(experiment_df)} | saved {path}")
    return experiment_df, masks, histories


def sensitivity_names(cfg: dict) -> dict:
    e = cfg["extra"]
    return {v: f"{e['sensitivity_method']} | w_{e['sensitivity_term']} = {v:.2f}" for v in e["sensitivity_values"]}


def run_extra(ctx: Context) -> pd.DataFrame:
    """(a) control fitness without the SHAP term, (b) sensitivity of one weight."""
    cfg, e = ctx.cfg, ctx.cfg["extra"]
    jobs = {}
    for name, spec in e["control_methods"].items():
        w = e["control_fitness"][spec["fitness"]]
        for p in cfg["population_sizes"]:
            for r in range(1, cfg["n_runs"] + 1):
                jobs[("control", name, p, r)] = (spec, w)
    sens_spec = METHODS[e["sensitivity_method"]]
    for v, name in sensitivity_names(cfg).items():
        for r in range(1, cfg["n_runs"] + 1):
            jobs[("sensitivity", name, e["sensitivity_population"], r)] = (sens_spec, sensitivity_weights(cfg, v))
    ckpt = os.path.join(cfg["results_dir"], f"checkpoints_{cfg['dataset']}_extra_{extra_fingerprint(ctx)}")
    done = _run_checkpointed(ctx, jobs, ckpt, _call_extra)
    extra_df = pd.DataFrame([row for k in jobs for row in done[k][0]])
    path = out_prefix(cfg) + "extra_runs.csv"
    extra_df.to_csv(path, index=False)
    print(f"\nAll {len(jobs)} additional runs available | saved {path}")
    return extra_df
