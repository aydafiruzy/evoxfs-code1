"""Tables, figures and statistical tests (same analysis as the original notebooks).

Use:
    res = Results(ctx, experiment_df, masks, histories)
    res.convergence(); res.summaries(); res.statistics()
    res.extra(extra_df); res.save(tuning_df)
"""
from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, wilcoxon

from .config import out_prefix
from .experiment import BASELINE, EVOX, METHODS, sensitivity_names
from .fitness import Context
from .models import evaluate_subset

try:
    from IPython.display import display
except ImportError:                          # plain Python
    display = print

ALPHA = 0.05
METRICS = ["Accuracy", "F1", "AUC", "AUPRC"]
SHORT = {"Classic DE": "DE", "DE + Scenario B fitness": "DE+B",
         "DE + Scenario B + SHAP-guided init": "DE+B+SI", "ODE + Scenario B + hybrid SHAP init": "ODE+B+HI",
         "EvoX-FS": "EvoX-FS", "Classic DE + SHAP-guided init": "DE+SI",
         "EvoX-FS (Scenario A fitness)": "EvoX(A)", "EvoX-FS (classic2 fitness)": "EvoX(c2)"}
FIG_DPI = 300
CONV_FRACTION = 0.99


# ---------------------------------------------------------------- statistics helpers
def rank_biserial(x, y):
    d = np.asarray(x, float) - np.asarray(y, float)
    d = d[d != 0]
    if d.size == 0:
        return 0.0
    r = rankdata(np.abs(d))
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def safe_wilcoxon(x, y):
    return 1.0 if np.allclose(x, y) else float(wilcoxon(x, y, zero_method="wilcox").pvalue)


def holm(p):
    p = np.asarray(p, float)
    out, running = np.empty_like(p), 0.0
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, (len(p) - rank) * p[i])
        out[i] = min(running, 1.0)
    return out


def safe_friedman(groups):
    if all(np.allclose(groups[0], g) for g in groups[1:]):
        return np.nan, 1.0
    stat, p = friedmanchisquare(*groups)
    return float(stat), float(p)


class Results:
    def __init__(self, ctx: Context, experiment_df: pd.DataFrame, masks: dict, histories: dict):
        self.ctx, self.cfg, self.data = ctx, ctx.cfg, ctx.data
        self.df, self.masks, self.histories = experiment_df, masks, histories
        self.pops = self.cfg["population_sizes"]
        self.fig_dir = os.path.join(self.cfg["results_dir"], "figures")
        os.makedirs(self.fig_dir, exist_ok=True)
        self.tables: dict[str, pd.DataFrame] = {}
        self.per_run = (experiment_df.groupby(["Population", "Method", "Run"], sort=False)
                        [["Features", *METRICS, "CV F1 (train)", "Generations", "OBL kept", "Search time (s)"]]
                        .mean().reset_index())
        self.no_fs_df = pd.DataFrame(evaluate_subset(np.ones(self.data.n_features, dtype=bool), ctx.eval_params,
                                                     self.data, self.cfg["eval_models"], self.cfg["seed"]))
        self.tables["no_feature_selection.csv"] = self.no_fs_df
        self.tables["per_run.csv"] = self.per_run

    # -------------------------------------------------------------- helpers
    def save_fig(self, name):
        path = os.path.join(self.fig_dir, f"{self.cfg['dataset']}_{name}.png")
        plt.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
        print("  figure saved:", path)

    def paired(self, metric, pop_size, method):
        pr = self.per_run
        return pr[(pr["Population"] == pop_size) & (pr["Method"] == method)].sort_values("Run")[metric].to_numpy()

    # -------------------------------------------------------------- convergence
    def convergence(self):
        grid = np.linspace(0, self.cfg["max_evals"], self.cfg["max_evals"] // min(self.pops) + 1)

        def curves_for(method, pop_size):
            out = []
            for history in self.histories[(method, pop_size)]:
                evals, values = zip(*history)
                out.append(np.interp(grid, evals, values, left=np.nan))
            return np.array(out)

        def evals_to_fraction(curve, fraction):
            valid = ~np.isnan(curve)
            c, g = curve[valid], grid[valid]
            target = c[0] - fraction * (c[0] - c[-1])
            return float(g[np.argmax(c <= target)]) if c[0] > c[-1] else float(g[0])

        groups = {}
        for m, spec in METHODS.items():
            groups.setdefault(spec["fitness"], []).append(m)
        conv_rows, curve_rows = [], []
        for fit_name, members in groups.items():
            fig, axes = plt.subplots(1, len(self.pops), figsize=(4.5 * len(self.pops), 4), sharey=True)
            for ax, p in zip(np.atleast_1d(axes), self.pops):
                for m in members:
                    C = curves_for(m, p)
                    mean, std = np.nanmean(C, axis=0), np.nanstd(C, axis=0)
                    ax.plot(grid, mean, label=m)
                    ax.fill_between(grid, mean - std, mean + std, alpha=0.15)
                    speeds = [evals_to_fraction(c, CONV_FRACTION) for c in C]
                    conv_rows.append({"Fitness function": fit_name, "Method": m, "Population": p,
                                      "Final fitness (mean)": float(np.nanmean(C[:, -1])),
                                      "Final fitness (std)": float(np.nanstd(C[:, -1])),
                                      f"Evals to {CONV_FRACTION:.0%} (mean)": float(np.mean(speeds))})
                    curve_rows += [{"Method": m, "Population": p, "Evaluations": e, "Mean best fitness": v, "Std": s}
                                   for e, v, s in zip(grid, mean, std)]
                ax.set(title=f"{fit_name} | NP = {p}", xlabel="Fitness evaluations"); ax.grid(alpha=0.3)
            np.atleast_1d(axes)[0].set_ylabel("Best fitness so far (mean of runs)")
            np.atleast_1d(axes)[-1].legend(fontsize=7)
            plt.tight_layout(); self.save_fig(f"convergence_{fit_name}"); plt.show()
        self.tables["convergence_summary.csv"] = pd.DataFrame(conv_rows)
        self.tables["convergence_curves.csv"] = pd.DataFrame(curve_rows)
        display(self.tables["convergence_summary.csv"].round(4))

    # -------------------------------------------------------------- summary tables + figures
    def summaries(self):
        pr, n_feat, labels = self.per_run, self.data.n_features, [SHORT[m] for m in METHODS]
        summary = pr.groupby(["Population", "Method"], sort=False).agg(
            F1_mean=("F1", "mean"), F1_std=("F1", "std"),
            Accuracy_mean=("Accuracy", "mean"), AUC_mean=("AUC", "mean"), AUPRC_mean=("AUPRC", "mean"),
            Features_mean=("Features", "mean"), Features_std=("Features", "std"),
            Generations=("Generations", "mean"), Time_s=("Search time (s)", "mean")).reset_index()
        summary["Reduction (%)"] = 100 * (1 - summary["Features_mean"] / n_feat)
        fmt = lambda m, s, d: f"{m:.{d}f} ± {s:.{d}f}"
        summary["F1 (mean ± std)"] = [fmt(m, s, 4) for m, s in zip(summary["F1_mean"], summary["F1_std"])]
        summary["Features (mean ± std)"] = [fmt(m, s, 2) for m, s in zip(summary["Features_mean"], summary["Features_std"])]
        self.tables["summary.csv"] = summary
        print("No feature selection (all features):"); display(self.no_fs_df.round(4))
        print(f"\nTest F1 (mean ± std over {self.cfg['n_runs']} runs, averaged over the classifiers):")
        display(summary.pivot(index="Method", columns="Population", values="F1 (mean ± std)").loc[list(METHODS)])
        print("Selected features (mean ± std), out of", n_feat, ":")
        display(summary.pivot(index="Method", columns="Population", values="Features (mean ± std)").loc[list(METHODS)])
        per_classifier = (self.df.groupby(["Population", "Method", "Classifier"], sort=False)["F1"]
                          .mean().unstack("Classifier").reset_index())
        self.tables["per_classifier.csv"] = per_classifier
        print("Test F1 per classifier:"); display(per_classifier.round(4))

        for metric, ylabel in [("F1", "Test F1"), ("Features", "Selected features")]:
            fig, axes = plt.subplots(1, len(self.pops), figsize=(4.5 * len(self.pops), 4.5), sharey=True)
            for ax, p in zip(np.atleast_1d(axes), self.pops):
                ax.boxplot([self.paired(metric, p, m) for m in METHODS], tick_labels=labels)
                ax.set(title=f"NP = {p}"); ax.tick_params(axis="x", rotation=60); ax.grid(axis="y", alpha=0.3)
            np.atleast_1d(axes)[0].set_ylabel(ylabel)
            plt.tight_layout(); self.save_fig(f"boxplot_{metric.lower()}"); plt.show()

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        for m in METHODS:
            s = summary[summary["Method"] == m]
            axes[0].errorbar(s["Population"], s["F1_mean"], yerr=s["F1_std"], marker="o", capsize=3, label=SHORT[m])
            axes[1].errorbar(s["Population"], s["Features_mean"], yerr=s["Features_std"], marker="o", capsize=3, label=SHORT[m])
        axes[0].set(xlabel="Population size", ylabel="Test F1", title=f"Test F1 vs population ({self.cfg['max_evals']} evaluations)")
        axes[1].set(xlabel="Population size", ylabel="Selected features", title="Selected features vs population")
        for ax in axes:
            ax.set_xticks(self.pops); ax.grid(alpha=0.3); ax.legend(fontsize=8)
        plt.tight_layout(); self.save_fig("population_effect"); plt.show()

        shap_norm, names = self.ctx.shap_norm, self.data.feature_names
        order = np.argsort(-shap_norm)
        freq = np.array([np.mean([mk for p in self.pops for mk in self.masks[(m, p)]], axis=0)
                         for m in METHODS])[:, order]
        self.tables["selection_frequency.csv"] = pd.DataFrame(freq, index=list(METHODS), columns=names[order]).reset_index(names="Method")
        fig, ax = plt.subplots(figsize=(min(1.0 * n_feat + 3, 40), 0.6 * len(METHODS) + 2))
        im = ax.imshow(np.vstack([shap_norm[order] / shap_norm.max(), freq]), cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_yticks(range(len(METHODS) + 1), ["SHAP (relative)"] + labels)
        ax.set_xticks(range(n_feat), names[order], rotation=60, ha="right")
        if n_feat <= 30:
            for i in range(len(METHODS)):
                for j in range(n_feat):
                    ax.text(j, i + 1, f"{freq[i, j]:.2f}", ha="center", va="center", fontsize=7,
                            color="white" if freq[i, j] > 0.6 else "black")
        ax.axhline(0.5, color="black", linewidth=1)
        ax.set_title("How often each feature is selected (features ordered by SHAP importance)")
        plt.colorbar(im, ax=ax, fraction=0.03)
        plt.tight_layout(); self.save_fig("selection_frequency"); plt.show()

        rng_plot = np.random.default_rng(self.cfg["seed"])
        fig, ax = plt.subplots(figsize=(9, 6))
        for m in METHODS:
            d = pr[pr["Method"] == m]
            ax.scatter(d["Features"] + rng_plot.uniform(-0.2, 0.2, len(d)), d["F1"], s=12, alpha=0.5, label=SHORT[m])
        ax.axhline(self.no_fs_df["F1"].mean(), linestyle="--", color="gray", label="All features")
        ax.set(xlabel="Selected features (jittered)", ylabel="Test F1", title="Trade-off: test F1 vs number of features")
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
        plt.tight_layout(); self.save_fig("f1_vs_features"); plt.show()

    # -------------------------------------------------------------- statistical tests
    def statistics(self):
        P, pops = self.paired, self.pops
        # 1) one change at a time (Wilcoxon + Holm)
        comparisons = [(m, s["previous"]) for m, s in METHODS.items() if s["previous"]] + [(EVOX, BASELINE)]
        step_rows = []
        for p in pops:
            block = []
            for m, prev in comparisons:
                for metric in ["F1", "Features"]:
                    x, y = P(metric, p, m), P(metric, p, prev)
                    block.append({"Population": p, "Method": m, "Compared with": prev, "Metric": metric,
                                  "Method mean": x.mean(), "Previous mean": y.mean(), "Difference": x.mean() - y.mean(),
                                  "Raw p": safe_wilcoxon(x, y), "r": rank_biserial(x, y)})
            for metric in ["F1", "Features"]:
                sub = [b for b in block if b["Metric"] == metric]
                for b, hp in zip(sub, holm([b["Raw p"] for b in sub])):
                    b["Holm p"], b["Significant"] = hp, hp < ALPHA
            step_rows.extend(block)
        steps = pd.DataFrame(step_rows)
        self.tables["ablation_steps.csv"] = steps
        print("1) One change at a time (Wilcoxon + Holm) — significant rows only:")
        display(steps[steps["Significant"]].round(4))

        # 2) SHAP effects vs Classic DE
        rows = []
        for p in pops:
            for label, m in [("SHAP in fitness only", "DE + Scenario B fitness"),
                             ("SHAP in start population only", "Classic DE + SHAP-guided init"),
                             ("SHAP in fitness + start", "DE + Scenario B + SHAP-guided init")]:
                for metric in ["F1", "Features"]:
                    x, y = P(metric, p, m), P(metric, p, BASELINE)
                    rows.append({"Population": p, "Effect": label, "Metric": metric,
                                 "Difference vs Classic DE": x.mean() - y.mean(),
                                 "Raw p": safe_wilcoxon(x, y), "r": rank_biserial(x, y)})
        shap_eff = pd.DataFrame(rows)
        shap_eff["Holm p"] = shap_eff.groupby(["Population", "Metric"])["Raw p"].transform(holm)
        shap_eff["Significant"] = shap_eff["Holm p"] < ALPHA
        self.tables["shap_effects.csv"] = shap_eff
        print("2) SHAP effects vs Classic DE:"); display(shap_eff.round(4))

        # 3) Friedman + mean ranks across all methods
        rank_rows, fr_rows = [], []
        for p in pops:
            for metric, higher_better in [("F1", True), ("Features", False)]:
                M = np.column_stack([P(metric, p, m) for m in METHODS])
                stat, pv = safe_friedman([M[:, j] for j in range(M.shape[1])])
                ranks = np.apply_along_axis(rankdata, 1, -M if higher_better else M).mean(axis=0)
                fr_rows.append({"Population": p, "Metric": metric, "Friedman stat": stat, "p": pv,
                                "Significant": pv < ALPHA, "Best (lowest mean rank)": list(METHODS)[int(np.argmin(ranks))]})
                rank_rows += [{"Population": p, "Metric": metric, "Method": m, "Mean rank": r}
                              for m, r in zip(METHODS, ranks)]
        self.tables["friedman_methods.csv"] = pd.DataFrame(fr_rows)
        self.tables["mean_ranks.csv"] = ranks_df = pd.DataFrame(rank_rows)
        print("3) Friedman test across all methods:"); display(self.tables["friedman_methods.csv"].round(4))
        display(ranks_df.pivot_table(index="Method", columns=["Metric", "Population"], values="Mean rank")
                .loc[list(METHODS)].round(2))

        # 4) population size effect
        pop_rows = []
        for m in METHODS:
            for metric in ["F1", "Features"]:
                groups = [P(metric, p, m) for p in pops]
                stat, pv = safe_friedman(groups)
                means = [g.mean() for g in groups]
                best = pops[int(np.argmax(means) if metric == "F1" else np.argmin(means))]
                pop_rows.append({"Method": m, "Metric": metric, "Friedman stat": stat, "p": pv,
                                 "Significant": pv < ALPHA, "Best population": best})
        self.tables["population_effect.csv"] = pd.DataFrame(pop_rows)
        print("4) Effect of population size (Friedman across NP):"); display(self.tables["population_effect.csv"].round(4))

        # 5) EvoX-FS vs Classic DE win/tie/loss
        wtl = []
        for p in pops:
            for metric, higher_better in [("F1", True), ("Features", False)]:
                d = P(metric, p, EVOX) - P(metric, p, BASELINE)
                d = d if higher_better else -d
                wtl.append({"Population": p, "Metric": metric, "Win": int((d > 0).sum()),
                            "Tie": int((d == 0).sum()), "Loss": int((d < 0).sum())})
        self.tables["evox_vs_classic_wtl.csv"] = pd.DataFrame(wtl)
        print("5) EvoX-FS vs Classic DE (win / tie / loss over paired runs):"); display(self.tables["evox_vs_classic_wtl.csv"])

    # -------------------------------------------------------------- additional experiments
    def extra(self, extra_df: pd.DataFrame):
        e, P = self.cfg["extra"], self.paired
        epr = (extra_df.groupby(["Group", "Method", "Population", "Run"], sort=False)
               [["Features", *METRICS, "CV F1 (train)", "SHAP_norm", "Avg_Corr"]].mean().reset_index())

        def EP(metric, name, pop_size):
            d = epr[(epr["Method"] == name) & (epr["Population"] == pop_size)]
            return d.sort_values("Run")[metric].to_numpy()

        # (a) control fitness without the SHAP term
        rows = []
        for p in self.pops:
            block = []
            for name, spec in e["control_methods"].items():
                ref = spec["compare_with"]
                for metric in ["F1", "Features"]:
                    w, wo = P(metric, p, ref), EP(metric, name, p)
                    block.append({"Population": p, "Method (with SHAP term)": ref, "Metric": metric,
                                  "With SHAP term": w.mean(), "Without SHAP term": wo.mean(),
                                  "SHAP-term effect": w.mean() - wo.mean(),
                                  "Raw p": safe_wilcoxon(w, wo), "r": rank_biserial(w, wo)})
            for metric in ["F1", "Features"]:
                sub = [b for b in block if b["Metric"] == metric]
                for b, hp in zip(sub, holm([b["Raw p"] for b in sub])):
                    b["Holm p"], b["Significant"] = hp, hp < ALPHA
            rows += block
        control = pd.DataFrame(rows)
        self.tables["control_shap_term.csv"] = control
        print("(a) Pure effect of the SHAP term in the fitness (with − without):"); display(control.round(4))

        fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
        for name, spec in e["control_methods"].items():
            ref = spec["compare_with"]
            for ax, metric in zip(axes, ["F1", "Features"]):
                ax.plot(self.pops, [P(metric, p, ref).mean() for p in self.pops], marker="o", label=f"{ref} (with SHAP)")
                ax.plot(self.pops, [EP(metric, name, p).mean() for p in self.pops], marker="s", linestyle="--", label=name)
        for ax, metric in zip(axes, ["Test F1", "Selected features"]):
            ax.set(xlabel="Population size", ylabel=metric, xticks=self.pops); ax.grid(alpha=0.3); ax.legend(fontsize=7)
        axes[0].set_title("Control: with vs without SHAP term"); axes[1].set_title("Selected features")
        plt.tight_layout(); self.save_fig("control_shap_term"); plt.show()

        # (b) sensitivity of one weight
        SP, values, names = e["sensitivity_population"], e["sensitivity_values"], sensitivity_names(self.cfg)
        base_value = self.cfg["fitness_weights"][METHODS[e["sensitivity_method"]]["fitness"]][e["sensitivity_term"]]
        base_name = names[min(values, key=lambda v: abs(v - base_value))]
        rows = []
        for v in values:
            f1, nf = EP("F1", names[v], SP), EP("Features", names[v], SP)
            rows.append({f"w_{e['sensitivity_term']}": v, "F1 mean": f1.mean(), "F1 std": f1.std(ddof=1),
                         "Features mean": nf.mean(), "Features std": nf.std(ddof=1),
                         "Raw p F1 vs default": safe_wilcoxon(f1, EP("F1", base_name, SP)),
                         "Raw p Features vs default": safe_wilcoxon(nf, EP("Features", base_name, SP))})
        sens = pd.DataFrame(rows)
        for col in ["Raw p F1 vs default", "Raw p Features vs default"]:
            sens[col.replace("Raw", "Holm")] = holm(sens[col])
        self.tables["sensitivity_w_size.csv"] = sens
        for metric in ["F1", "Features"]:
            stat, pv = safe_friedman([EP(metric, names[v], SP) for v in values])
            print(f"(b) Friedman across w_{e['sensitivity_term']} values, {metric}: stat = {stat:.3f}, p = {pv:.4g}")
        print(f"(b) Sensitivity of w_{e['sensitivity_term']} for {e['sensitivity_method']} (NP = {SP}):")
        display(sens.round(4))
        diff = np.abs(EP("F1", base_name, SP) - P("F1", SP, e["sensitivity_method"])).max()
        print(f"Reproducibility check (default weight vs main experiment), max |ΔF1| = {diff:.2e}")

        fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
        for ax, metric, col in [(axes[0], "Test F1", "F1"), (axes[1], "Selected features", "Features")]:
            ax.errorbar(values, sens[f"{col} mean"], yerr=sens[f"{col} std"], marker="o", capsize=3,
                        label=e["sensitivity_method"])
            ax.axhline(P(col, SP, BASELINE).mean(), linestyle="--", color="gray", label=f"{BASELINE} (classic2)")
            ax.set(xlabel=f"w_{e['sensitivity_term']}", ylabel=metric, xticks=values,
                   title=f"{metric} vs w_{e['sensitivity_term']} (NP = {SP})")
            ax.grid(alpha=0.3); ax.legend(fontsize=8)
        plt.tight_layout(); self.save_fig("sensitivity_w_size"); plt.show()

    # -------------------------------------------------------------- save
    def save(self, tuning_df: pd.DataFrame | None = None):
        if tuning_df is not None:
            self.tables["tuning_options.csv"] = tuning_df.assign(params=tuning_df["params"].map(json.dumps))
        prefix = out_prefix(self.cfg)
        for name, frame in self.tables.items():
            frame.to_csv(prefix + name, index=False)
            print(" -", prefix + name)
        print("\nFigures in:", self.fig_dir)
