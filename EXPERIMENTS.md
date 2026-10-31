# Experiment log

One row per experiment. Newest at the top.
- **Code version**: the git tag or the first 7 characters of the commit (`git log -1 --format=%h`).
- **Results**: the folder on Google Drive (`MyDrive/EvoXFS_results/...`).

| Date | Dataset | Code version | What changed | Main result (NP = 20) | Results |
|---|---|---|---|---|---|
| 2026-10-18 | Sonar | `v1.0` (original notebook) | first full run + extra experiments | EvoX-FS F1 0.869 with 30.5/60 features; all features 0.904 | `checkpoints_sonar_83d5d3c3e1`, `checkpoints_sonar_extra_89949cfeda` |
| 2026-10-17 | Breast Cancer | `v1.0` (original notebook) | first full run + extra experiments | EvoX-FS F1 0.956 with 20.6/30 features; all features 0.959 | `checkpoints_breast_cancer_48e3e8305e`, `checkpoints_breast_cancer_extra_a889b1a812` |
| 2026-10-16 | Wine | `v1.0` (original notebook) | first full run + extra experiments | EvoX-FS F1 1.000 with 8.7/13 features; all features 1.000 | `checkpoints_wine_8cbac0018b`, `checkpoints_wine_extra_628ed52522` |

## Planned

- [ ] `v1.1`: SHAP term as the **mean** SHAP of the selected features (now: sum, rewards large subsets).
- [ ] `v1.1`: Classic DE baseline with common weights (error 0.9, size 0.1).
- [ ] Add a harder / high-dimensional dataset (e.g. Ionosphere, a microarray set).
