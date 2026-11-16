# Comparison with MI-MPODE (Yu et al., ESWA 2025) on its 11 UCI datasets

Paper: F. Yu et al., "Multi-population differential evolution approach for feature selection
with mutual information ranking", Expert Systems with Applications 260 (2025) 125404.

| File | What |
|---|---|
| `run_uci.py` | 30 runs: full-feature baselines (KNN, XGBoost, LightGBM, CatBoost) and FS with DE / DE + hybrid init / EvoX-FS, KNN wrapper, paper protocol |
| `summarize.py` | builds `results/paper_mimpode/summary.md` (Turkish tables) and summary CSVs |
| `paper_tables_3_4_uci.csv` | Table 3 (accuracy mean/std) and Table 4 (selected features) of the paper, UCI rows, all 10 algorithms |
| `../../src/evoxfs/uci_paper.py` | download + loaders for the 11 UCI datasets |

```bash
pip install -e . openpyxl            # + `apt-get install unar` for the SCADI .rar
python scripts/paper_mimpode/run_uci.py --raw-dir data/uci --download --out results/paper_mimpode
python scripts/paper_mimpode/summarize.py --out results/paper_mimpode
```
About 45 minutes on 4 cores (CatBoost on Arrhythmia is the slowest part).

Assumptions (the paper does not state them clearly):
- "10-fold CV ... 70% / 30%": we use 30 random stratified 70/30 splits (one per run); the fitness
  accuracy is 5-fold CV on the 70% part only; the reported accuracy is on the untouched 30%.
- Fitness: the paper writes `alpha*accuracy + (1-alpha)*Ns/Nf`, but its Table 6 values match
  `alpha*(1-accuracy) + (1-alpha)*Ns/Nf` (minimised), alpha = 0.95. We use the latter.
- Budget: NP 10 x 100 iterations = 1000 fitness evaluations, counting opposite solutions too.
- KNN k = 5 on min-max scaled features (scaler fitted on the training part).
- Missing values (`?`) are filled with the column median.
