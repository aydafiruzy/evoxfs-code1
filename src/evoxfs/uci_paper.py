"""The 11 UCI datasets of Yu et al. (2025), "Multi-population differential
evolution approach for feature selection with mutual information ranking"
(MI-MPODE), Expert Systems with Applications 260, 125404, Table 1.

Download once with `download_all(raw_dir)` (needs access to
archive.ics.uci.edu), then load with `load_uci(name, raw_dir)`.

Every loader returns (X, y): X float array, y integer labels 0..K-1.
Preprocessing is the minimum needed to make the data numeric:
  - id / name columns are dropped,
  - '?' (missing) is replaced by the column median (training-free, so it is
    the same for every split; few values are affected, see MISSING below),
  - columns that are constant over the whole dataset are KEPT, so that the
    feature count matches the UCI description (they are harmless for FS).

Differences from the paper's Table 1 (we use the official UCI files):
  - Movement-libras: UCI has 90 features + class; the paper says 91
    (it most likely counted the class column).
  - LSVT: UCI has 310 features; the paper says 309.
  - SCADI: UCI has 205 features (Gender, Age, 203 binary ICF codes).
  - Arrhythmia: 16 classes are defined but only 13 occur in the data.
"""
from __future__ import annotations

import io
import os
import subprocess
import urllib.request
import zipfile

import numpy as np
import pandas as pd

UCI_IDS = {  # name -> (UCI id, slug)
    "Heart": (45, "heart+disease"),
    "Zoo": (111, "zoo"),
    "Parkinsons": (174, "parkinsons"),
    "Dermatology": (33, "dermatology"),
    "Ionosphere": (52, "ionosphere"),
    "LungCancer": (62, "lung+cancer"),
    "Movement-libras": (181, "libras+movement"),
    "Musk1": (74, "musk+version+1"),
    "Arrhythmia": (5, "arrhythmia"),
    "LSVT": (282, "lsvt+voice+rehabilitation"),
    "SCADI": (446, "scadi"),
}
PAPER_TABLE1 = {  # name -> (instances, features, classes) as printed in the paper
    "Heart": (303, 13, 5), "Zoo": (101, 16, 7), "Parkinsons": (195, 22, 2),
    "Dermatology": (366, 34, 6), "Ionosphere": (351, 34, 2), "LungCancer": (32, 56, 3),
    "Movement-libras": (360, 91, 15), "Musk1": (476, 166, 2), "Arrhythmia": (452, 279, 16),
    "LSVT": (126, 309, 2), "SCADI": (70, 205, 7),
}


def download_all(raw_dir: str) -> None:
    """Download and unpack every dataset zip from the UCI archive."""
    for name, (uid, slug) in UCI_IDS.items():
        target = os.path.join(raw_dir, str(uid))
        if os.path.isdir(target) and os.listdir(target):
            continue
        os.makedirs(target, exist_ok=True)
        url = f"https://archive.ics.uci.edu/static/public/{uid}/{slug}.zip"
        with urllib.request.urlopen(url) as r:
            zipfile.ZipFile(io.BytesIO(r.read())).extractall(target)
    musk = os.path.join(raw_dir, "74", "clean1.data")
    if not os.path.exists(musk):  # Unix .Z compressed
        with open(musk, "wb") as out:
            subprocess.run(["gzip", "-dc", musk + ".Z"], stdout=out, check=True)
    scadi = os.path.join(raw_dir, "446", "SCADI-Dataset", "SCADI.csv")
    if not os.path.exists(scadi):  # needs `unar` (apt-get install unar)
        subprocess.run(["unar", "-q", "-f", "-o", os.path.join(raw_dir, "446"),
                        os.path.join(raw_dir, "446", "SCADI-Dataset.rar")], check=True)


def _labels(values):
    _, y = np.unique(np.asarray(values).ravel(), return_inverse=True)
    return y.astype(int)


def _numeric_fill(df: pd.DataFrame) -> np.ndarray:
    df = df.replace("?", np.nan).apply(pd.to_numeric, errors="coerce")
    return df.fillna(df.median()).to_numpy(dtype=float)


def load_uci(name: str, raw_dir: str) -> tuple[np.ndarray, np.ndarray]:
    p = lambda *parts: os.path.join(raw_dir, *parts)  # noqa: E731
    if name == "Heart":  # Cleveland, 14 attrs, target 0..4 (5 classes)
        df = pd.read_csv(p("45", "processed.cleveland.data"), header=None)
        return _numeric_fill(df.iloc[:, :-1]), _labels(df.iloc[:, -1])
    if name == "Zoo":  # drop animal name
        df = pd.read_csv(p("111", "zoo.data"), header=None)
        return df.iloc[:, 1:-1].to_numpy(float), _labels(df.iloc[:, -1])
    if name == "Parkinsons":  # drop name, target = status
        df = pd.read_csv(p("174", "parkinsons.data"))
        return df.drop(columns=["name", "status"]).to_numpy(float), _labels(df["status"])
    if name == "Dermatology":  # age has 8 missing
        df = pd.read_csv(p("33", "dermatology.data"), header=None)
        return _numeric_fill(df.iloc[:, :-1]), _labels(df.iloc[:, -1])
    if name == "Ionosphere":
        df = pd.read_csv(p("52", "ionosphere.data"), header=None)
        return df.iloc[:, :-1].to_numpy(float), _labels(df.iloc[:, -1])
    if name == "LungCancer":  # first column is the class, 5 missing values
        df = pd.read_csv(p("62", "lung-cancer.data"), header=None)
        return _numeric_fill(df.iloc[:, 1:]), _labels(df.iloc[:, 0])
    if name == "Movement-libras":
        df = pd.read_csv(p("181", "movement_libras.data"), header=None)
        return df.iloc[:, :-1].to_numpy(float), _labels(df.iloc[:, -1])
    if name == "Musk1":  # drop molecule + conformation names
        df = pd.read_csv(p("74", "clean1.data"), header=None)
        return df.iloc[:, 2:-1].to_numpy(float), _labels(df.iloc[:, -1])
    if name == "Arrhythmia":  # 408 missing values in total (mostly column 'J')
        df = pd.read_csv(p("5", "arrhythmia.data"), header=None)
        return _numeric_fill(df.iloc[:, :-1]), _labels(df.iloc[:, -1])
    if name == "LSVT":
        xl = p("282", "LSVT_voice_rehabilitation.xlsx")
        X = pd.read_excel(xl, sheet_name="Data").to_numpy(float)
        y = pd.read_excel(xl, sheet_name="Binary response").iloc[:, 0]
        return X, _labels(y)
    if name == "SCADI":
        df = pd.read_csv(p("446", "SCADI-Dataset", "SCADI.csv"))
        return df.drop(columns="Classes").to_numpy(float), _labels(df["Classes"])
    raise ValueError(f"Unknown dataset: {name}")
