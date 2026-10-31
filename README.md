# EvoX-FS

**SHAP-guided, self-adaptive, opposition-based Differential Evolution for feature selection.**

EvoX-FS searches for a small feature subset that keeps classification accuracy high. Each candidate is a vector in [0, 1]^N, and a gene ≥ 0.5 means "feature selected". The search uses DE/rand/1/bin with these additions:

| Component | What it does |
|---|---|
| SHAP-guided start population | Features with high SHAP importance start with genes in [0.5, 1] |
| Opposition-based learning (ODE) | Opposite individuals at the start, plus opposition "jumps" during the search |
| Self-adaptive F and CR | Each individual keeps its own F / CR; worse individuals resample them more often |
| Fitness A / B | Error (1 − CV F1) + feature ratio + SHAP loss (+ mean \|Pearson correlation\| in B) |

The classifiers are XGBoost, LightGBM and CatBoost. Results are reported as Accuracy, F1, AUC and AUPRC on a held-out 20% test set. Each setting is run 30 times, the paired runs are compared with Wilcoxon + Holm and Friedman tests, and each component is checked in an ablation study.

## Repository layout

```
configs/            all settings (YAML)
  base.yaml         shared by every dataset: budget, DE parameters, fitness weights, ...
  wine.yaml, breast_cancer.yaml, sonar.yaml, ionosphere.yaml
src/evoxfs/         the method, written once
  config.py         read + check the YAML settings
  data.py           datasets and the fixed 80/20 split
  models.py         XGBoost / LightGBM / CatBoost + test metrics
  shap_tools.py     SHAP importance (RandomForest + TreeExplainer)
  tuning.py         Default vs Greedy hyperparameter search
  fitness.py        fitness functions A / B / classic
  de.py             the switchable DE (init, OBL, self-adaptation)
  experiment.py     the 8 methods, checkpointed runs, extra experiments
  analysis.py       tables, figures, statistical tests
notebooks/
  run_experiment.ipynb   the ONLY notebook to run (choose the dataset at the top)
  archive/               the original Colab notebooks of October 2026, with their outputs
tests/              automatic checks (see "Tests")
docs/               review notes and a Git guide (Persian)
EXPERIMENTS.md      log of every experiment: date, code version, settings, result
```

## How to run (Google Colab)

1. Open `notebooks/run_experiment.ipynb` in Colab (File → Open notebook → GitHub → `aydafiruzy/evoxfs-code1`).
2. Set `DATASET = "wine"` (or `breast_cancer`, `sonar`, `ionosphere`) in the first cell.
3. Runtime → Run all. Results go to `MyDrive/EvoXFS_results` on Google Drive.

The full experiment (8 methods × 4 populations × 30 runs) takes about 5 hours per dataset on 8 cores. It saves checkpoints, so a disconnected session continues where it stopped. The checkpoint folder names match the October 2026 runs, so those runs are reused and not recomputed.

To change a setting (for example a fitness weight), edit `configs/base.yaml`. Then commit the change and add a line to `EXPERIMENTS.md`.

## Results so far (October 2026, code version `v1.0`)

Test F1 averaged over the 3 classifiers and 30 runs, at population size 20. The number of selected features is in brackets.

| Dataset (features) | All features | Classic DE | EvoX-FS |
|---|---|---|---|
| Wine (13) | 1.000 | 0.916 (2.0) | 1.000 (8.7) |
| Breast Cancer (30) | 0.959 | 0.931 (2.5) | 0.956 (20.6) |
| Sonar (60) | 0.904 | 0.830 (12.8) | 0.869 (30.5) |

Known issues with this version, to be fixed in the next one (details in `docs/REVIEW_fa.md`):
- The SHAP term is `1 − sum(SHAP of selected features)`, so it rewards **larger** subsets. Without it, EvoX-FS keeps about 3–5 features instead of 9–21.
- The Classic DE baseline uses a size weight of 0.4, which is unusually high (0.1 or 0.01 is common), so it keeps very few features.
- EvoX-FS does not yet beat "all features" on any dataset.

All result CSVs and figures are on Google Drive (`EvoXFS_results`, one file per table, named `EvoXFS_colab_<dataset>_<table>.csv`).

## Tests

```bash
pip install -e . pytest
python -m pytest -q -m "not slow"   # fast checks (~10 min on 2 cores)
python -m pytest -q                 # + exact comparison with the original notebooks
```

- `test_matches_original.py`: the refactored code gives **exactly** the same runs as the original notebooks (Wine, Breast Cancer).
- `test_fingerprint.py`: the checkpoint folder names match the existing Drive folders.
- `test_smoke.py`: a tiny end-to-end run through every analysis step.
