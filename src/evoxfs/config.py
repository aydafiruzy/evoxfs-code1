"""Experiment configuration.

All numerical settings live in ONE place: a YAML file in `configs/`.
`load_config()` reads `configs/base.yaml`, then applies the dataset file
(e.g. `configs/wine.yaml`) on top of it, then checks that the values make
sense (weights sum to 1, etc.).

Sources of the default values (same as the original Colab notebooks):
- 30 runs, 80/20 stratified hold-out, 1000 fitness evaluations per run
  (DESFO protocol). Populations {10, 20, 50, 100}.
- DE/rand/1/bin with F = 0.5, CR = 0.9 (Storn & Price); self-adaptive
  F in [F_low, F_low + F_up] (jDE, Brest et al. 2006); opposition-based
  learning with jump rate 0.3 (ODE, Rahnamayan et al. 2008).
- Threshold 0.5 maps a gene in [0, 1] to selected / not selected.
- Hybrid start population: 50% random, 30% SHAP-guided, 10% opposite of
  SHAP-guided, 10% opposite of random.
- fitness = w_err*(1-F1_cv) + w_size*(N_sel/N) + w_shap*(1-SHAP_norm)
            + w_red*AvgAbsCorr
"""
from __future__ import annotations

import copy
import os
from pathlib import Path

import yaml

TOL = 1e-9

# Keys that only change HOW the experiment runs, not its results. They are
# left out of the experiment fingerprint (see experiment.fingerprint).
EXECUTION_ONLY_KEYS = ("n_jobs", "results_dir", "checkpoint_rounds", "tuning", "greedy_tuning",
                       "tuning_selection", "search_spaces", "default_params", "fixed_params")

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


def _deep_update(base: dict, new: dict) -> dict:
    for key, value in new.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _restore_tuples(cfg: dict) -> dict:
    """YAML has no tuples; turn the (low, high) ranges and search-space entries back into tuples."""
    for key in ("shap_high_range", "shap_low_range"):
        cfg[key] = tuple(cfg[key])
    for model, space in cfg["search_spaces"].items():
        cfg["search_spaces"][model] = {p: tuple(v) for p, v in space.items()}
    return cfg


def validate(cfg: dict) -> None:
    for name, w in cfg["fitness_weights"].items():
        assert abs(sum(w.values()) - 1) < TOL, f"Fitness weights of {name} do not sum to 1"
    for key in ("init_ratios_hybrid", "init_ratios_shap"):
        assert abs(sum(cfg[key].values()) - 1) < TOL, f"{key} does not sum to 1"
    assert 0 < cfg["test_size"] < 1
    assert cfg["F_low"] < cfg["F_up"]
    assert cfg["checkpoint_rounds"] >= 1
    extra = cfg.get("extra")
    if extra:
        for name, w in extra["control_fitness"].items():
            assert abs(sum(w.values()) - 1) < TOL, f"{name} weights do not sum to 1"
        assert all(0 < v < 1 for v in extra["sensitivity_values"])
        assert extra["sensitivity_population"] in cfg["population_sizes"]


def load_config(dataset_file: str | os.PathLike, results_dir: str | None = None,
                overrides: dict | None = None) -> dict:
    """Read configs/base.yaml + one dataset file (+ optional overrides) and validate.

    dataset_file : "wine", "configs/wine.yaml" or any path to a YAML file.
    results_dir  : where CSVs, figures and checkpoints are written.
                   Default: Google Drive on Colab, ./results elsewhere.
    overrides    : dict merged last, handy for quick tests, e.g. {"n_runs": 2}.
    """
    path = Path(dataset_file)
    if not path.suffix:
        path = CONFIG_DIR / f"{dataset_file}.yaml"
    with open(CONFIG_DIR / "base.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with open(path, encoding="utf-8") as f:
        _deep_update(cfg, yaml.safe_load(f) or {})
    if overrides:
        _deep_update(cfg, copy.deepcopy(overrides))

    cfg["n_jobs"] = cfg.get("n_jobs") or os.cpu_count()
    cfg["results_dir"] = results_dir or cfg.get("results_dir") or default_results_dir()
    os.makedirs(cfg["results_dir"], exist_ok=True)
    _restore_tuples(cfg)
    validate(cfg)
    return cfg


def default_results_dir() -> str:
    if os.path.isdir("/content/drive/MyDrive"):          # Colab with Drive mounted
        return "/content/drive/MyDrive/EvoXFS_results"
    return str(REPO_ROOT / "results")


def out_prefix(cfg: dict) -> str:
    """File-name prefix of every result CSV, e.g. .../EvoXFS_colab_wine_summary.csv."""
    return os.path.join(cfg["results_dir"], f"EvoXFS_colab_{cfg['dataset']}_")


def print_summary(cfg: dict) -> None:
    print("Results folder :", cfg["results_dir"])
    print("CPU cores used :", cfg["n_jobs"], "| checkpoint every", cfg["checkpoint_rounds"], "rounds")
    print("Dataset        :", cfg["dataset"], "| split", f"{1-cfg['test_size']:.0%}/{cfg['test_size']:.0%}")
    print("Runs           :", cfg["n_runs"], "| budget", cfg["max_evals"], "evaluations")
    print("Population     :", cfg["population_sizes"])
    print("Hybrid init    :", cfg["init_ratios_hybrid"])
    print("Fitness weights:")
    for name, w in cfg["fitness_weights"].items():
        print(f"  {name:11s}", {k: round(v, 3) for k, v in w.items()}, "| sum =", round(sum(w.values()), 6))
    print("All checks passed ✓")
