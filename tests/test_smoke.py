"""Fast end-to-end check: tiny budget, a few runs, every analysis step.

Run with:  python -m pytest -q tests/
"""
import matplotlib
matplotlib.use("Agg")

import pytest

from evoxfs import prepare
from evoxfs.analysis import Results
from evoxfs.experiment import run_extra, run_main

TINY = {"n_runs": 3, "population_sizes": [10, 15, 20], "max_evals": 60, "checkpoint_rounds": 1, "n_jobs": 2}


@pytest.mark.parametrize("dataset", ["wine"])
def test_pipeline_runs(dataset, tmp_path):
    ctx, _, _ = prepare(dataset, results_dir=str(tmp_path), overrides=TINY, tune=False, plot=False)
    df, masks, histories = run_main(ctx)
    assert len(df) == 8 * 3 * 3 * 3          # methods x populations x runs x classifiers
    res = Results(ctx, df, masks, histories)
    res.convergence(); res.summaries(); res.statistics()
    res.extra(run_extra(ctx))
    res.save()
    assert (tmp_path / f"EvoXFS_colab_{dataset}_summary.csv").exists()
    assert (tmp_path / f"EvoXFS_colab_{dataset}_sensitivity_w_size.csv").exists()

    # second call must reuse the checkpoints (nothing left to run)
    df2, _, _ = run_main(ctx)
    assert df2.drop(columns="Search time (s)").equals(df.drop(columns="Search time (s)"))
