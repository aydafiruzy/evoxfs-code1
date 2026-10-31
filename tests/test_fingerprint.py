"""The checkpoint fingerprints must match the folders of the October 2026
Colab runs on Google Drive, so the new code can reuse those runs."""
import numpy as np
import pytest

from evoxfs.config import load_config
from evoxfs.experiment import extra_fingerprint, fingerprint
from evoxfs.fitness import Context
from evoxfs.data import load_data


@pytest.mark.parametrize("dataset, main_fp, extra_fp", [
    ("wine", "8cbac0018b", "628ed52522"),
    ("breast_cancer", "48e3e8305e", "a889b1a812"),
    ("sonar", "83d5d3c3e1", "89949cfeda"),
])
def test_fingerprint_matches_drive(dataset, main_fp, extra_fp, tmp_path):
    cfg = load_config(dataset, results_dir=str(tmp_path))
    data = load_data(load_config("wine", results_dir=str(tmp_path)))   # data does not enter the fingerprint
    ctx = Context(cfg, data, np.full(data.n_features, 1 / data.n_features),
                  {m: {} for m in cfg["eval_models"]}, {})
    assert fingerprint(ctx) == main_fp
    assert extra_fingerprint(ctx) == extra_fp
