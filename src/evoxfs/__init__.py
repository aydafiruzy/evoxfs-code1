"""EvoX-FS: SHAP-guided, self-adaptive, opposition-based DE for feature selection.

Typical use (see notebooks/run_experiment.ipynb):

    from evoxfs import prepare
    ctx, tuning_df, shap_res = prepare("wine")          # config + data + SHAP + tuning
    from evoxfs.experiment import run_main
    df, masks, histories = run_main(ctx)
"""
from .config import load_config
from .data import load_data
from .fitness import Context
from .shap_tools import compute_shap
from .tuning import run_tuning

__version__ = "1.0.0"


def prepare(dataset: str, results_dir: str | None = None, overrides: dict | None = None,
            tune: bool = True, plot: bool = True):
    """Load config + data, compute SHAP, tune the models. Returns (ctx, tuning_df, shap_res).

    tune=False skips tuning and uses the library defaults (what the tuning
    chose for wine, breast_cancer and sonar anyway) — handy for quick tests.
    """
    cfg = load_config(dataset, results_dir, overrides)
    data = load_data(cfg)
    shap_res = compute_shap(data, cfg)
    if tune:
        tuning_df, tuned = run_tuning(data, cfg, plot=plot)
    else:
        tuning_df, tuned = None, {m: dict(p) for m, p in cfg["default_params"].items()}
    ctx = Context(cfg, data, shap_res.norm,
                  eval_params={m: tuned[m] for m in cfg["eval_models"]},
                  fitness_params=tuned["Fitness XGBoost"])
    return ctx, tuning_df, shap_res
