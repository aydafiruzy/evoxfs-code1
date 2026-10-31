"""Dataset loading and the fixed train/test split.

- The TEST set is never seen by feature selection, SHAP, tuning or the
  fitness; it is used only for the final reporting.
- The split is fixed by cfg["seed"], so every method in every run is
  compared on exactly the same partition; run-to-run variation comes only
  from the stochastic optimiser.
- No scaling: the classifiers are tree-based and Pearson correlation is
  scale-invariant.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml, load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split

SKLEARN_LOADERS = {
    "wine": lambda: load_wine(as_frame=True),                    # 178 x 13, 3 classes
    "breast_cancer": lambda: load_breast_cancer(as_frame=True),  # 569 x 30, 2 classes
}
OPENML_IDS = {"sonar": 40, "ionosphere": 59}                     # 208 x 60 / 351 x 34, 2 classes


@dataclass
class Data:
    name: str
    X_train_df: pd.DataFrame
    X_test_df: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray

    @property
    def X_train(self) -> np.ndarray:
        return self.X_train_df.to_numpy(dtype=float)

    @property
    def X_test(self) -> np.ndarray:
        return self.X_test_df.to_numpy(dtype=float)

    @property
    def feature_names(self) -> np.ndarray:
        return np.asarray(self.X_train_df.columns)

    @property
    def n_features(self) -> int:
        return self.X_train_df.shape[1]

    @property
    def classes(self) -> np.ndarray:
        return np.unique(self.y_train)

    @property
    def n_classes(self) -> int:
        return self.classes.size

    @property
    def binary(self) -> bool:
        return self.n_classes == 2


def load_raw(name: str) -> tuple[pd.DataFrame, np.ndarray]:
    if name in SKLEARN_LOADERS:
        d = SKLEARN_LOADERS[name]()
        return d.data.copy(), d.target.to_numpy()
    if name in OPENML_IDS or name.startswith("openml:"):
        data_id = OPENML_IDS.get(name) or int(name.split(":", 1)[1])
        d = fetch_openml(data_id=data_id, as_frame=True, parser="auto")
        y, _ = pd.factorize(d.target, sort=True)          # class labels -> 0..K-1
        return d.data.copy().astype(float), y
    raise ValueError(f"Unsupported dataset: {name}")


def load_data(cfg: dict) -> Data:
    X_df, y = load_raw(cfg["dataset"])
    non_numeric = X_df.columns[~X_df.dtypes.apply(pd.api.types.is_numeric_dtype)]
    assert len(non_numeric) == 0, f"Non-numeric columns must be encoded first: {list(non_numeric)}"
    assert not X_df.isna().any().any(), "Dataset contains missing values"
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_df, y, test_size=cfg["test_size"], stratify=y, random_state=cfg["seed"])
    return Data(cfg["dataset"], X_tr, X_te, y_tr, y_te)


def describe(data: Data, test_size: float) -> pd.DataFrame:
    """Print a short summary and return the class-distribution table."""
    n = len(data.y_train) + len(data.y_test)
    print(f"Dataset          : {data.name}")
    print(f"Samples/features : {n} / {data.n_features}")
    print(f"Classes          : {data.n_classes} ({'binary' if data.binary else 'multi-class'})")
    print(f"Train / Test     : {len(data.y_train)} / {len(data.y_test)}  "
          f"({1-test_size:.0%} / {test_size:.0%})")
    y_all = np.concatenate([data.y_train, data.y_test])
    dist = pd.DataFrame({"full": pd.Series(y_all).value_counts().sort_index(),
                         "train": pd.Series(data.y_train).value_counts().sort_index(),
                         "test": pd.Series(data.y_test).value_counts().sort_index()})
    dist["train_%"] = (dist["train"] / dist["train"].sum() * 100).round(1)
    dist["test_%"] = (dist["test"] / dist["test"].sum() * 100).round(1)
    return dist
