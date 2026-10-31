"""Differential Evolution with switchable components.

One DE (DE/rand/1/bin) whose parts can be switched on/off, so every
comparison isolates ONE change:
  init     : "random" | "shap" (SHAP-guided) | "hybrid" (+ opposite individuals)
  obl      : opposition-based jumping (ODE) with dynamic bounds
  adaptive : self-adaptive F and CR
"""
from __future__ import annotations

import time

import numpy as np

from .fitness import Context, CountingFitness
from .shap_tools import top_k


def repair_bounds(vector, rng):
    vector = vector.copy()
    bad = (vector < 0.0) | (vector > 1.0)
    vector[bad] = rng.random(int(bad.sum()))
    return vector


def make_population(ctx: Context, init: str, pop_size: int, rng):
    cfg, n = ctx.cfg, ctx.n_features
    if init == "random":
        return rng.random((pop_size, n))
    ratios = cfg["init_ratios_hybrid"] if init == "hybrid" else cfg["init_ratios_shap"]
    n_shap = int(round(ratios["shap"] * pop_size))
    n_opp_shap = min(n_shap, int(round(ratios.get("opp_shap", 0.0) * pop_size)))
    n_opp_random = int(round(ratios.get("opp_random", 0.0) * pop_size))
    n_random = pop_size - n_shap - n_opp_shap - n_opp_random

    k = top_k(cfg, n)
    top_idx = np.argsort(-ctx.shap_norm)[:k]
    random_part = rng.random((n_random, n))
    shap_part = rng.uniform(*cfg["shap_low_range"], (n_shap, n))
    shap_part[:, top_idx] = rng.uniform(*cfg["shap_high_range"], (n_shap, k))
    return np.vstack([random_part, shap_part,
                      1.0 - shap_part[:n_opp_shap],
                      1.0 - random_part[:n_opp_random]])


def opposition_jump(population, fitness, F, CR, fitness_fn):
    low, high = population.min(axis=0), population.max(axis=0)
    opposite = low + high - population
    union = np.vstack([population, opposite])
    union_fitness = np.concatenate([fitness, [fitness_fn(v)[0] for v in opposite]])
    keep = np.argsort(union_fitness, kind="stable")[:len(population)]
    return (union[keep], union_fitness[keep],
            np.concatenate([F, F])[keep], np.concatenate([CR, CR])[keep],
            int((keep >= len(population)).sum()))


def switchable_de(ctx: Context, fitness_fn, init: str, obl: bool, adaptive: bool, pop_size: int, seed: int):
    cfg = ctx.cfg
    rng = np.random.default_rng(seed)
    fitness_fn = CountingFitness(fitness_fn)
    start = time.perf_counter()
    budget, n = cfg["max_evals"], ctx.n_features

    population = make_population(ctx, init, pop_size, rng)
    fitness = np.array([fitness_fn(v)[0] for v in population])
    if adaptive:
        # NOTE: F = F_low + rand * F_up lies in [0.1, 1.0] (jDE convention)
        F = cfg["F_low"] + rng.random(pop_size) * cfg["F_up"]
        CR = rng.random(pop_size)
    else:
        F, CR = np.full(pop_size, cfg["F_fixed"]), np.full(pop_size, cfg["CR_fixed"])

    history = [(fitness_fn.n_evals, float(fitness.min()))]
    generations, obl_kept = 0, 0
    while fitness_fn.n_evals + pop_size <= budget:
        generations += 1
        f_best, f_worst = float(fitness.min()), float(fitness.max())
        new_pop, new_fit, new_F, new_CR = population.copy(), fitness.copy(), F.copy(), CR.copy()
        for i in range(pop_size):
            F_try, CR_try = F[i], CR[i]
            if adaptive:
                # worse individuals change their F/CR more often
                tau = (cfg["degenerate_tau"] if np.isclose(f_best, f_worst)
                       else 1.0 - (fitness[i] - f_worst) / (f_best - f_worst))
                if rng.random() < tau:
                    F_try = cfg["F_low"] + rng.random() * cfg["F_up"]
                if rng.random() < tau:
                    CR_try = rng.random()
            r1, r2, r3 = rng.choice(np.delete(np.arange(pop_size), i), 3, replace=False)
            mutant = repair_bounds(population[r1] + F_try * (population[r2] - population[r3]), rng)
            cross = rng.random(n) <= CR_try
            cross[rng.integers(n)] = True
            trial = np.where(cross, mutant, population[i])
            trial_fitness = fitness_fn(trial)[0]
            if trial_fitness < fitness[i]:
                new_pop[i], new_fit[i], new_F[i], new_CR[i] = trial, trial_fitness, F_try, CR_try
        population, fitness, F, CR = new_pop, new_fit, new_F, new_CR
        if obl and rng.random() < cfg["jump_rate"] and fitness_fn.n_evals + pop_size <= budget:
            population, fitness, F, CR, kept = opposition_jump(population, fitness, F, CR, fitness_fn)
            obl_kept += kept
        history.append((fitness_fn.n_evals, float(fitness.min())))

    best = int(np.argmin(fitness))
    return {"mask": population[best] >= cfg["threshold"], "fitness": float(fitness[best]),
            "details": fitness_fn.fn(population[best])[1], "n_evals": fitness_fn.n_evals,
            "generations": generations, "obl_kept": obl_kept, "history": history,
            "search_time": time.perf_counter() - start}
