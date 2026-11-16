"""Turn the run CSVs of run_uci.py into Markdown comparison tables.

  python scripts/paper_mimpode/summarize.py --out results/paper_mimpode
writes results/paper_mimpode/summary.md (+ summary CSVs).
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
ORDER = ["Heart", "Zoo", "Parkinsons", "Dermatology", "Ionosphere", "LungCancer",
         "Movement-libras", "Musk1", "Arrhythmia", "LSVT", "SCADI"]
OURS = ["DE", "DE + hybrid init", "EvoX-FS (KNN)"]


def ms(s):  # mean ± std
    return f"{s.mean():.4f} ± {s.std(ddof=1):.4f}"


def wil(a, b):
    d = np.asarray(a) - np.asarray(b)
    if np.allclose(d, 0):
        return 1.0
    return wilcoxon(a, b, zero_method="zsplit").pvalue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/paper_mimpode")
    out = ap.parse_args().out
    base = pd.read_csv(os.path.join(out, "baselines_runs.csv"))
    fs = pd.read_csv(os.path.join(out, "fs_runs.csv"))
    paper = pd.read_csv(os.path.join(HERE, "paper_tables_3_4_uci.csv"))
    md = []

    # --- table 1: full-feature baselines
    md += ["## 1. Tüm öznitelikler (öznitelik seçimi yok), 30 koşu, test kümesi\n",
           "| Veri kümesi | Öz. | KNN Acc | KNN Macro-F1 | XGBoost Macro-F1 | LightGBM Macro-F1 | CatBoost Macro-F1 | En iyi (Macro-F1) |",
           "|---|---|---|---|---|---|---|---|"]
    t1 = []
    for d in ORDER:
        b = base[base.Dataset == d]
        g = b.groupby("Model")
        f1 = g["Macro-F1"].mean()
        best = f1.idxmax()
        md.append(f"| {d} | {b.Features.iloc[0]} | {ms(g.get_group('KNN')['Accuracy'])} | "
                  f"{ms(g.get_group('KNN')['Macro-F1'])} | {ms(g.get_group('XGBoost')['Macro-F1'])} | "
                  f"{ms(g.get_group('LightGBM')['Macro-F1'])} | {ms(g.get_group('CatBoost')['Macro-F1'])} | {best} |")
        for m, gg in g:
            t1.append({"Dataset": d, "Model": m, **{k: gg[k].mean() for k in ["Accuracy", "Macro-F1", "Balanced acc", "Time (s)"]}})
    pd.DataFrame(t1).to_csv(os.path.join(out, "summary_baselines.csv"), index=False)
    wins = pd.DataFrame(t1).pivot(index="Dataset", columns="Model", values="Macro-F1").idxmax(axis=1).value_counts()
    md.append("\nEn yüksek ortalama Macro-F1 sayısı: " + ", ".join(f"{k} {v}" for k, v in wins.items()) + "\n")

    # --- table 2: accuracy vs paper
    md += ["## 2. Makaleyle karşılaştırma: KNN doğruluğu (ortalama ± std) ve seçilen öznitelik sayısı\n",
           "Makale sütunları Tablo 3 ve Tablo 4'ten alınmıştır. Bizim sütunlar aynı bütçeyle (NP=10, 1000 değerlendirme, α=0.95, KNN k=5, 30 koşu) çalıştırıldı.\n",
           "| Veri kümesi | Makale DE | Makale MI-MPODE | Makale en iyi | Bizim KNN tüm öz. | Bizim DE | Bizim DE + hibrit başlatma | Bizim EvoX-FS (KNN) |",
           "|---|---|---|---|---|---|---|---|"]
    t2 = []
    for d in ORDER:
        p = paper[paper.dataset == d].set_index("algorithm")
        pb = p.acc_mean.idxmax()
        cell = lambda a: f"{p.loc[a, 'acc_mean']:.4f} ({p.loc[a, 'n_selected']:.1f})"  # noqa: E731
        knn_all = base[(base.Dataset == d) & (base.Model == "KNN")]
        row = [d, cell("DE"), cell("MI-MPODE"), f"{p.loc[pb, 'acc_mean']:.4f} {pb}",
               f"{knn_all.Accuracy.mean():.4f} ({knn_all.Features.iloc[0]})"]
        rec = {"Dataset": d, "Paper DE acc": p.loc["DE", "acc_mean"], "Paper DE nsel": p.loc["DE", "n_selected"],
               "Paper MI-MPODE acc": p.loc["MI-MPODE", "acc_mean"], "Paper MI-MPODE nsel": p.loc["MI-MPODE", "n_selected"],
               "Paper best acc": p.loc[pb, "acc_mean"], "Paper best alg": pb,
               "KNN all acc": knn_all.Accuracy.mean(), "KNN all macroF1": knn_all["Macro-F1"].mean()}
        for m in OURS:
            f = fs[(fs.Dataset == d) & (fs.Method == m)]
            row.append(f"{f.Accuracy.mean():.4f} ± {f.Accuracy.std(ddof=1):.4f} ({f.Selected.mean():.1f})")
            rec.update({f"{m} acc": f.Accuracy.mean(), f"{m} acc std": f.Accuracy.std(ddof=1),
                        f"{m} macroF1": f["Macro-F1"].mean(), f"{m} nsel": f.Selected.mean(),
                        f"{m} cv acc": f["CV acc (train)"].mean(), f"{m} time": f["Search time (s)"].mean()})
        md.append("| " + " | ".join(row) + " |")
        t2.append(rec)
    t2 = pd.DataFrame(t2)
    t2.to_csv(os.path.join(out, "summary_vs_paper.csv"), index=False)

    # --- table 2b: paper numbers vs our in-search (train CV) accuracy
    md += ["\n### 2b. Makale doğruluğu, bizim arama içi (eğitim CV) doğruluğumuza daha yakın\n",
           "| Veri kümesi | Makale DE | Makale MI-MPODE | Bizim DE CV acc | Bizim EvoX-FS CV acc | Bizim EvoX-FS test acc |",
           "|---|---|---|---|---|---|"]
    for _, r in t2.iterrows():
        md.append(f"| {r.Dataset} | {r['Paper DE acc']:.4f} | {r['Paper MI-MPODE acc']:.4f} | {r['DE cv acc']:.4f} | "
                  f"{r['EvoX-FS (KNN) cv acc']:.4f} | {r['EvoX-FS (KNN) acc']:.4f} |")
    gap_test = (t2["Paper MI-MPODE acc"] - t2["EvoX-FS (KNN) acc"]).abs().mean()
    gap_cv = (t2["Paper MI-MPODE acc"] - t2["EvoX-FS (KNN) cv acc"]).abs().mean()
    md.append(f"\nMakale MI-MPODE ile ortalama mutlak fark: test doğruluğumuza {gap_test:.4f}, "
              f"eğitim CV doğruluğumuza {gap_cv:.4f}.\n")

    # --- table 3: our methods, Macro-F1 + Wilcoxon
    md += ["\n## 3. Bizim yöntemler: Macro-F1, seçilen öznitelik oranı, eğitim CV doğruluğu, Wilcoxon (EvoX-FS vs DE, 30 eşli koşu)\n",
           "| Veri kümesi | DE Macro-F1 | DE + hibrit Macro-F1 | EvoX-FS Macro-F1 | DE oran | EvoX-FS oran | DE CV acc | EvoX-FS CV acc | p (test acc) | p (öz. sayısı) |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for d in ORDER:
        f = fs[fs.Dataset == d]
        g = {m: f[f.Method == m].sort_values("Run") for m in OURS}
        n = f.Features.iloc[0]
        md.append(f"| {d} | {ms(g['DE']['Macro-F1'])} | {ms(g['DE + hybrid init']['Macro-F1'])} | "
                  f"{ms(g['EvoX-FS (KNN)']['Macro-F1'])} | {g['DE'].Selected.mean() / n:.3f} | "
                  f"{g['EvoX-FS (KNN)'].Selected.mean() / n:.3f} | {g['DE']['CV acc (train)'].mean():.4f} | "
                  f"{g['EvoX-FS (KNN)']['CV acc (train)'].mean():.4f} | "
                  f"{wil(g['EvoX-FS (KNN)'].Accuracy.values, g['DE'].Accuracy.values):.3g} | "
                  f"{wil(g['EvoX-FS (KNN)'].Selected.values, g['DE'].Selected.values):.3g} |")

    # --- dataset-level counts
    ev, mi, de_p, de_o = (t2["EvoX-FS (KNN) acc"], t2["Paper MI-MPODE acc"], t2["Paper DE acc"], t2["DE acc"])
    md += ["\n## 4. Özet sayımlar (11 veri kümesi)\n",
           f"- Bizim EvoX-FS doğruluğu > makale MI-MPODE: {(ev > mi).sum()} / 11 (veri kümesi düzeyinde Wilcoxon p = {wil(ev, mi):.3g})",
           f"- Bizim EvoX-FS öznitelik sayısı < makale MI-MPODE: {(t2['EvoX-FS (KNN) nsel'] < t2['Paper MI-MPODE nsel']).sum()} / 11",
           f"- Bizim DE doğruluğu > makale DE: {(de_o > de_p).sum()} / 11 (protokol farkını gösterir, aşağıya bakın)",
           f"- Bizim EvoX-FS doğruluğu > bizim DE: {(ev > de_o).sum()} / 11; öznitelik sayısı daha az: {(t2['EvoX-FS (KNN) nsel'] < t2['DE nsel']).sum()} / 11",
           f"- Bizim EvoX-FS > KNN tüm öznitelikler: {(ev > t2['KNN all acc']).sum()} / 11"]
    open(os.path.join(out, "summary.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
