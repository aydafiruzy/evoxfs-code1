"""The refactored code must give EXACTLY the same runs as the original
Colab notebooks (notebooks/archive/). Slow (~2 min per dataset)."""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from evoxfs import prepare
from evoxfs.experiment import run_job

ARCHIVE = Path(__file__).resolve().parents[1] / "notebooks" / "archive"


def load_original(notebook: Path, workdir: Path, max_evals: int) -> dict:
    """Execute the original notebook cells (config, data, models, SHAP, fitness, DE, methods)."""
    cells = [''.join(c["source"]) for c in json.load(open(notebook))["cells"] if c["cell_type"] == "code"]
    g = {"display": lambda *a, **k: None, "__name__": "original"}
    old_show, plt.show = plt.show, (lambda *a, **k: plt.close("all"))
    cwd = os.getcwd(); os.chdir(workdir)
    try:
        exec(cells[0].replace("!pip install -q catboost shap", ""), g)
        for i in (1, 2, 3, 4):
            exec(cells[i], g)
        g["CONFIG"]["max_evals"] = max_evals
        g["FITNESS_PARAMS"] = {}                     # tuning chose library defaults
        g["EVAL_PARAMS"] = {m: {} for m in g["CONFIG"]["eval_models"]}
        for i in (6, 7, 8):
            exec(cells[i], g)
    finally:
        os.chdir(cwd); plt.show = old_show
    return g


@pytest.mark.slow
@pytest.mark.parametrize("dataset, notebook", [("wine", "wine_original.ipynb"),
                                               ("breast_cancer", "breast_cancer_original.ipynb")])
def test_same_results_as_original(dataset, notebook, tmp_path):
    orig = load_original(ARCHIVE / notebook, tmp_path, max_evals=80)
    ctx, _, _ = prepare(dataset, results_dir=str(tmp_path), overrides={"max_evals": 80}, tune=False, plot=False)
    assert np.allclose(ctx.shap_norm, orig["shap_norm_global"])
    for method in ["Classic DE", "DE + Scenario B + SHAP-guided init", "ODE + Scenario B + hybrid SHAP init", "EvoX-FS"]:
        a_rows, a_extra = orig["run_job"](method, 10, 3)
        b_rows, b_extra = run_job(ctx, method, 10, 3)
        for a, b in zip(a_rows, b_rows):
            assert {k: v for k, v in a.items() if k != "Search time (s)"} == \
                   {k: v for k, v in b.items() if k != "Search time (s)"}
        assert a_extra["history"] == b_extra["history"]
        assert (a_extra["mask"] == b_extra["mask"]).all()
