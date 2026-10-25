"""Simulation of GARCH processes for recovery experiments.

Why simulate? With simulated data we KNOW the true (p, q), so we can test
whether the ACF/PACF rules recover it. Real data never tells us the truth.

The recovery harness: simulate a known process R times, run the selector on
each series, and count how often it returns the true order. It is used to calibrate two
rules (fade tolerance, mean trigger) on simulated data only.

The final recovery experiment: a general GARCH(p, q) simulator (sim_garch), the
recovery.* true models, the metrics (exact, nesting, family, ...) and the table builder
recovery_table. It runs on the EVAL seeds only; the frozen rules are never changed here.

The last section is a post-hoc diagnostic: why ARCH(1) exact recovery falls with n. It reuses the
recovery code on its own seed range and measures how often a chance PACF-of-squares spike appears at lags
2..5. It reports only; it never changes a rule.
"""
import copy
import itertools
import math

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.baselines import grid_bic
from src.correlogram import acf_pacf_band, significant_lags
from src.rules import mean_order_rule
from src.selector import select_and_fit


def sim_garch11(n: int, omega: float, alpha: float, beta: float, seed: int,
                burn: int = 500, dist: str = "normal", nu: float = 8.0) -> np.ndarray:
    """Simulate zero-mean GARCH(1,1) shocks e_t = sqrt(h_t) * z_t, length n after burn-in.

    Variance recursion (hand-rolled):
        h_t = omega + alpha * e_{t-1}^2 + beta * h_{t-1}
    Intuition: today's variance = a baseline (omega) + a reaction to yesterday's
    surprise (alpha) + memory of yesterday's variance (beta).

    dist: "normal" or "t" (Student-t with nu dof, rescaled to unit variance).
    """
    # --- 1. Validate inputs: the process only has a finite variance if alpha + beta < 1.
    if omega <= 0 or alpha < 0 or beta < 0:
        raise ValueError("need omega > 0, alpha >= 0, beta >= 0")
    if alpha + beta >= 1:
        raise ValueError("alpha + beta must be < 1 (stationarity)")

    # --- 2. Draw the iid standardised noise z_t (seeded, so runs are reproducible).
    rng = np.random.default_rng(seed)
    total = n + burn  # extra "burn" draws are thrown away so the start-up does not matter

    if dist == "normal":
        z = rng.standard_normal(total)
    elif dist == "t":
        if nu <= 2:
            raise ValueError("nu must be > 2")
        # A raw t(nu) has variance nu/(nu-2); multiplying by sqrt((nu-2)/nu) makes it 1.
        z = rng.standard_t(nu, total) * np.sqrt((nu - 2) / nu)
    else:
        raise ValueError(f"unknown dist {dist!r}")

    # --- 3. Run the recursion, starting from the unconditional (long-run) variance.
    e = np.empty(total)
    h = omega / (1 - alpha - beta)

    for t in range(total):
        e[t] = np.sqrt(h) * z[t]                 # today's shock, scaled by today's volatility
        h = omega + alpha * e[t] ** 2 + beta * h  # tomorrow's variance

    # --- 4. Drop the burn-in.
    return e[burn:]


def sim_arch1(n: int, omega: float, alpha: float, seed: int, burn: int = 500,
              dist: str = "normal", nu: float = 8.0) -> np.ndarray:
    """Simulate ARCH(1): a GARCH(1,1) with beta = 0 (no variance memory)."""
    return sim_garch11(n, omega, alpha, 0.0, seed, burn, dist, nu)


def sim_arch(n: int, omega: float, alphas: list[float], seed: int, burn: int = 500) -> np.ndarray:
    """Simulate zero-mean normal ARCH(p) shocks e_t = sqrt(h_t) * z_t, p = len(alphas).

    Variance recursion (hand-rolled):
        h_t = omega + alphas[0] * e_{t-1}^2 + alphas[1] * e_{t-2}^2 + ...
    Intuition: today's variance = a baseline plus a reaction to each of the last p surprises.
    Used only for the calibration-only true model arch2.
    Raises ValueError if omega <= 0, any alpha < 0, alphas is empty or sum(alphas) >= 1.
    """
    # --- 1. Validate: the variance is finite only if the alphas sum to less than 1.
    alphas = np.asarray(alphas, dtype=float)
    if omega <= 0 or alphas.size == 0 or (alphas < 0).any():
        raise ValueError("need omega > 0 and a non-empty alphas list with alpha >= 0")
    if alphas.sum() >= 1:
        raise ValueError("sum(alphas) must be < 1 (stationarity)")

    # --- 2. Draw the iid standard normal noise z_t (seeded) with burn extra draws to discard.
    total = n + burn
    z = np.random.default_rng(seed).standard_normal(total)

    # --- 3. Run the recursion. Start every remembered squared shock at the long-run variance
    # so the first h_t is already sensible; the burn-in removes what is left of the start-up.
    past_sq = np.full(alphas.size, omega / (1 - alphas.sum()))   # past_sq[0] = e_{t-1}^2, [1] = e_{t-2}^2, ...
    e = np.empty(total)

    for t in range(total):
        h = omega + alphas @ past_sq                    # today's variance from the last p squared shocks
        e[t] = np.sqrt(h) * z[t]                        # today's shock, scaled by today's volatility
        past_sq = np.concatenate(([e[t] ** 2], past_sq[:-1]))   # push today's e^2 in, drop the oldest

    # --- 4. Drop the burn-in.
    return e[burn:]


def _push(buf: np.ndarray, value: float) -> np.ndarray:
    """Put value at the front of a short history buffer and drop its oldest entry.

    Input: buf 1-D array (may be empty), value float. Output: new array of the same length.
    An empty buffer (no lags of that kind) is returned unchanged.
    """
    if buf.size == 0:
        return buf

    return np.concatenate(([value], buf[:-1]))


def sim_garch(n: int, omega: float, alphas, betas, seed: int, burn: int = 500,
              dist: str = "normal", nu: float = 8.0) -> np.ndarray:
    """Simulate zero-mean GARCH(p, q) shocks e_t = sqrt(h_t) * z_t, length n after burn-in.

    p = len(alphas) lags of squared shocks, q = len(betas) lags of variances (arch convention).
    Variance recursion (hand-rolled):
        h_t = omega + sum_i alphas[i] * e_{t-1-i}^2 + sum_j betas[j] * h_{t-1-j}
    Intuition: today's variance = a baseline + a reaction to each of the last p surprises + a
    memory of each of the last q variances. Empty alphas and betas give constant variance omega (iid).
    dist: "normal" or "t" (Student-t with nu dof, rescaled to unit variance, so only the tails change).
    Raises ValueError if omega <= 0, any coefficient < 0, or sum(alphas) + sum(betas) >= 1
    (the variance would not exist). Same seed -> same series.
    """
    # --- 1. Validate: a finite unconditional variance needs persistence = sum of all coefficients < 1.
    alphas = np.asarray(alphas, dtype=float).ravel()
    betas = np.asarray(betas, dtype=float).ravel()

    if omega <= 0 or (alphas < 0).any() or (betas < 0).any():
        raise ValueError("need omega > 0 and all alphas, betas >= 0")

    persistence = alphas.sum() + betas.sum()
    if persistence >= 1:
        raise ValueError("persistence (sum of alphas and betas) must be < 1 (stationarity)")

    # --- 2. Draw the iid standardised noise z_t (seeded); burn extra draws are thrown away later.
    rng = np.random.default_rng(seed)
    total = n + burn

    if dist == "normal":
        z = rng.standard_normal(total)
    elif dist == "t":
        if nu <= 2:
            raise ValueError("nu must be > 2")
        # A raw t(nu) has variance nu/(nu-2); multiplying by sqrt((nu-2)/nu) makes it 1.
        z = rng.standard_t(nu, total) * np.sqrt((nu - 2) / nu)
    else:
        raise ValueError(f"unknown dist {dist!r}")

    # --- 3. Run the recursion. Every remembered squared shock and variance starts at the long-run
    # variance omega / (1 - persistence), so the first h is already sensible; the burn-in removes the rest.
    long_run = omega / (1.0 - persistence)
    past_sq = np.full(alphas.size, long_run)   # past_sq[0] = e_{t-1}^2, past_sq[1] = e_{t-2}^2, ...
    past_h = np.full(betas.size, long_run)     # past_h[0] = h_{t-1}, past_h[1] = h_{t-2}, ...
    e = np.empty(total)

    for t in range(total):
        h = omega + alphas @ past_sq + betas @ past_h   # today's variance from the remembered history
        e[t] = np.sqrt(h) * z[t]                        # today's shock, scaled by today's volatility
        past_sq = _push(past_sq, e[t] ** 2)             # remember today's squared shock
        past_h = _push(past_h, h)                       # and today's variance

    # --- 4. Drop the burn-in.
    return e[burn:]


def true_spec(par: dict) -> dict:
    """True mean order m and variance order (p, q) of a recovery.models entry.

    Input: par = dict with alphas, betas and optionally phi. Output: dict(m, p, q) where p = number
    of alphas, q = number of betas and m = 1 if an AR(1) coefficient phi is present and non-zero, else 0.
    """
    return dict(m=1 if par.get("phi", 0.0) != 0.0 else 0, p=len(par["alphas"]), q=len(par["betas"]))


def simulate_recovery_model(true_model: str, n: int, seed: int, cfg: dict) -> np.ndarray:
    """Simulate one series from a recovery.models true process.

    Input: true_model name in recovery.models, n length, seed, cfg. Output: ndarray of length n.
    omega = 1 - persistence, so the unconditional variance of the shocks is 1. With phi in the
    entry the shocks drive an AR(1) mean x_t = phi * x_{t-1} + e_t. Same inputs -> same series.
    """
    models = cfg["recovery"]["models"]
    if true_model not in models:
        raise ValueError(f"unknown recovery model {true_model!r}; choose from {sorted(models)}")
    par = models[true_model]

    # Unit unconditional variance: omega = 1 - (sum of alphas + sum of betas).
    omega = 1.0 - (sum(par["alphas"]) + sum(par["betas"]))
    dist, nu, burn = par.get("dist", "normal"), par.get("nu", 8.0), cfg["sim"]["burn"]

    if not par.get("phi"):
        return sim_garch(n, omega, par["alphas"], par["betas"], seed, burn, dist, nu)

    # AR(1) mean: simulate burn extra shocks and drop them so the start x_0 = 0 does not bias the sample.
    e = sim_garch(n + burn, omega, par["alphas"], par["betas"], seed, burn, dist, nu)
    x, prev = np.empty(n + burn), 0.0

    for t, shock in enumerate(e):
        prev = par["phi"] * prev + shock   # yesterday's value, shrunk by phi, plus today's shock
        x[t] = prev

    return x[burn:]


def simulate_true_model(true_model: str, n: int, seed: int, cfg: dict) -> np.ndarray:
    """Simulate one series from a named true process; parameters come from calibration.models.

    true_model: "arch1", "arch2" (calibration-only), "garch11", "ar1_garch11" (AR(1) mean
    with GARCH(1,1) shocks) or "iid".
    Output: ndarray of length n. Same (true_model, n, seed, cfg) always gives the same series.
    """
    models, burn = cfg["calibration"]["models"], cfg["sim"]["burn"]
    if true_model not in models:
        raise ValueError(f"unknown true_model {true_model!r}; choose from {sorted(models)}")
    par = models[true_model]

    # iid normal: no mean structure and no volatility clustering (the null case).
    if true_model == "iid":
        return par["sigma"] * np.random.default_rng(seed).standard_normal(n)

    if true_model == "arch1":
        return sim_arch1(n, par["omega"], par["alpha"], seed, burn)

    if true_model == "garch11":
        return sim_garch11(n, par["omega"], par["alpha"], par["beta"], seed, burn)

    if true_model == "arch2":
        return sim_arch(n, par["omega"], par["alphas"], seed, burn)   # calibration-only

    if true_model == "ar1_garch11":
        # GARCH(1,1) shocks e_t, then the AR(1) recursion x_t = phi * x_{t-1} + e_t
        # (hand-rolled). Simulate burn extra points and drop them so
        # the AR start-up (x_0 = 0) does not bias the first observations.
        e = sim_garch11(n + burn, par["omega"], par["alpha"], par["beta"], seed, burn)
        x, prev = np.empty(n + burn), 0.0

        for t, shock in enumerate(e):
            prev = par["phi"] * prev + shock   # yesterday's value, shrunk by phi, plus today's shock
            x[t] = prev

        return x[burn:]

    raise ValueError(f"true_model {true_model!r} has parameters but no simulator")


def _one_replication(true_model: str, n: int, seed: int, cfg: dict, selector: str = "hybrid",
                     model_set: str = "calibration") -> dict:
    """Simulate one series, run the chosen selector, return a flat row of what it chose.

    selector: "hybrid" (select_and_fit: rules, gate, escalation) or "grid_bic" (baselines.grid_bic).
    model_set: "calibration" (calibration.models) or "recovery" (recovery.models).
    Module-level (not nested) so joblib can pickle it for worker processes.
    """
    # --- 1. The series: the same name can mean different parameters in the two model sets.
    if model_set == "calibration":
        x = simulate_true_model(true_model, n, seed, cfg)
    elif model_set == "recovery":
        x = simulate_recovery_model(true_model, n, seed, cfg)
    else:
        raise ValueError(f"unknown model_set {model_set!r}")

    # --- 2. grid_bic: no rule, no gate, no escalation, so there is no start order and no gate verdict.
    if selector == "grid_bic":
        g = grid_bic(x, cfg)

        # Same frozen mean rule as every selector; it is read directly because the grid
        # result only carries m inside spec, which is None when no fit was valid.
        m = mean_order_rule(x, cfg)["m"]

        return dict(seed=seed, m=m, p=g["order"][0], q=g["order"][1],
                    start_p=float("nan"), start_q=float("nan"), dist=cfg["fit"]["dist"],
                    models_fitted=g["models_fitted"], flagged=g["flagged"],
                    no_valid_fit=g["no_valid_fit"], gate_passed=float("nan"))

    if selector != "hybrid":
        raise ValueError(f"unknown selector {selector!r}; choose hybrid or grid_bic")

    # --- 3. hybrid: the full selector.
    r = select_and_fit(x, cfg)

    # The rule's starting order is the first fit attempted (escalation_log[0]); comparing it
    # with the final order shows whether escalation repaired or changed the rule's proposal.
    start_p, start_q = r["escalation_log"][0]["order"]

    return dict(seed=seed, m=r["mean_spec"]["m"], p=r["order"][0], q=r["order"][1],
                start_p=start_p, start_q=start_q, dist=r["dist"], models_fitted=r["models_fitted"], flagged=r["flagged"],
                no_valid_fit=r["no_valid_fit"], gate_passed=bool(r["gate"]["passed"]))


def order_recovery_experiment(true_model: str, n: int, R: int, seed_base: int,
                              cfg: dict, selector: str = "hybrid",
                              model_set: str = "calibration") -> pd.DataFrame:
    """Monte Carlo: run a selector on R simulated series from a known process.

    Replication i uses seed seed_base + i (callers pass seed_base from config), so the
    experiment is reproducible and independent of how many workers joblib uses.
    selector: "hybrid" (default) or "grid_bic". model_set: "calibration" (default, calibration.models
    and calibration.n_jobs) or "recovery" (recovery.models and recovery.n_jobs).
    Output: DataFrame, one row per replication, columns seed, m (chosen AR order),
    p, q (chosen variance order), start_p, start_q (the rule's proposal before escalation;
    NaN for grid_bic), dist, models_fitted, flagged, no_valid_fit, gate_passed (final fit passed
    the gate; NaN for grid_bic).
    When no_valid_fit is True, (p, q) is only the rule's start order (grid_bic: (1, 1)), not a fitted model.
    """
    seeds = [seed_base + i for i in range(R)]

    # Replications are independent, so they run in parallel; results come back in seed order.
    rows = Parallel(n_jobs=cfg[model_set]["n_jobs"])(
        delayed(_one_replication)(true_model, n, s, cfg, selector, model_set) for s in seeds)

    return pd.DataFrame(rows)


def recovery_summary(df: pd.DataFrame, true_model: str, cfg: dict) -> dict:
    """Turn an experiment DataFrame into recovery rates against the known truth.

    exact_order  = share with (p, q) equal to the true order AND a valid fit
                   (a no_valid_fit row never counts as recovered).
    any_ar_rate  = share given any AR mean (m > 0); on iid data this is the false-AR rate.
    m_correct    = share with m equal to the true m; on ar1_garch11 this is the AR(1) detection rate.
    Also mean models_fitted and the flagged / no_valid_fit shares.
    """
    truth = cfg["calibration"]["truth"][true_model]
    true_p, true_q = truth["order"]

    exact = (df["p"] == true_p) & (df["q"] == true_q) & ~df["no_valid_fit"]

    return dict(exact_order=float(exact.mean()),
                any_ar_rate=float((df["m"] > 0).mean()),
                m_correct=float((df["m"] == truth["m"]).mean()),
                mean_models_fitted=float(df["models_fitted"].mean()),
                flagged_rate=float(df["flagged"].mean()),
                no_valid_fit_rate=float(df["no_valid_fit"].mean()))


def apply_candidates(cfg: dict, fade: str | None = None, mean_trigger: str | None = None,
                     cutoff: str | None = None) -> dict:
    """Return a deep copy of cfg with a named fade, mean-trigger and/or cut-off candidate switched on.

    fade: "F0".."F4" sets variance.fade_max_spikes; mean_trigger: "M0"/"M1" sets mean.trigger;
    cutoff: "C0"/"C1" sets variance.cutoff_rule.
    None leaves that rule at its config value. The input cfg is never modified.
    """
    out = copy.deepcopy(cfg)  # deep copy: nested dicts must not be shared with the caller
    cands = cfg["calibration"]["candidates"]

    if fade is not None:
        out["variance"]["fade_max_spikes"] = cands["fade"][fade]

    if mean_trigger is not None:
        out["mean"]["trigger"] = cands["mean_trigger"][mean_trigger]

    if cutoff is not None:
        out["variance"]["cutoff_rule"] = cands["cutoff"][cutoff]

    return out


def check_seed_ranges(cfg: dict) -> None:
    """Raise if calibration seeds could ever coincide with evaluation seeds.

    Calibration uses calibration.seed_base .. +calibration.R, evaluation uses
    eval.seed_base .. +sim.R. Tuning a rule on the seeds it is later scored on would leak.
    """
    cal, ev = cfg["calibration"], cfg["eval"]
    cal_lo, cal_hi = cal["seed_base"], cal["seed_base"] + cal["R"]
    ev_lo, ev_hi = ev["seed_base"], ev["seed_base"] + cfg["sim"]["R"]

    if cal_lo < ev_hi and ev_lo < cal_hi:   # the two half-open intervals overlap
        raise ValueError("calibration and eval seed ranges overlap")


def calibration_table(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the calibration; return (table, orders).

    table: one row per (rule, candidate, true model).
      Fade candidates (every name in calibration.candidates.fade) are scored on arch1 and
      garch11 with the mean trigger at its config value.
      Mean-trigger candidates are scored on all four true models (iid, arch1, garch11,
      ar1_garch11) with fade at its config value. Every candidate sees the SAME seeds
      (calibration.seed_base + i), so candidates are paired replication by replication.
      Columns: rule, candidate, true_model, n, R, then the recovery_summary rates.
    orders: for fade candidates only, how often each (start order, final order) pair
      occurred. Columns: candidate, true_model, start_order, final_order, count.
    Simulated data only; uses calibration seeds, never eval seeds.
    """
    check_seed_ranges(cfg)
    cal = cfg["calibration"]
    rows, order_rows = [], []

    jobs = [("fade", name, ("arch1", "garch11")) for name in cal["candidates"]["fade"]] + \
           [("mean_trigger", name, ("iid", "arch1", "garch11", "ar1_garch11"))
            for name in cal["candidates"]["mean_trigger"]]

    for rule, name, true_models in jobs:
        # Switch exactly one candidate on; every other setting stays as in config.
        run_cfg = apply_candidates(cfg, **{rule: name})

        for tm in true_models:
            df = order_recovery_experiment(tm, cal["n"], cal["R"], cal["seed_base"], run_cfg)
            rows.append(dict(rule=rule, candidate=name, true_model=tm, n=cal["n"], R=cal["R"],
                             **recovery_summary(df, tm, run_cfg)))

            # Order counts: what the rule proposed vs what survived the gate and escalation.
            if rule == "fade":
                order_rows.append(order_counts(df).assign(candidate=name, true_model=tm))

    orders = pd.concat(order_rows, ignore_index=True)[
        ["candidate", "true_model", "start_order", "final_order", "count"]]

    return pd.DataFrame(rows), orders


def cutoff_calibration_table(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the cut-off grid: cut-off candidates x fade candidates x true models; return (table, orders).

    Grid = every cutoff name in calibration.candidates.cutoff crossed with the fade names in
    calibration.cutoff_grid.fade, scored on calibration.cutoff_grid.models (arch1, arch2, garch11).
    Every cell uses the SAME calibration seeds (paired), n = calibration.n, R = calibration.R;
    eval seeds are never used. The mean trigger stays at its config value (M0 by default).
    table columns: cutoff, fade, true_model, exact_order, flagged_rate.
    orders columns: cutoff, fade, true_model, start_order, final_order, count.
    """
    check_seed_ranges(cfg)
    cal = cfg["calibration"]
    rows, order_rows = [], []

    for cutoff in cal["candidates"]["cutoff"]:
        for fade in cal["cutoff_grid"]["fade"]:
            # Switch exactly this (cutoff, fade) pair on; every other setting stays as in config.
            run_cfg = apply_candidates(cfg, fade=fade, cutoff=cutoff)

            for tm in cal["cutoff_grid"]["models"]:
                df = order_recovery_experiment(tm, cal["n"], cal["R"], cal["seed_base"], run_cfg)
                s = recovery_summary(df, tm, run_cfg)
                rows.append(dict(cutoff=cutoff, fade=fade, true_model=tm,
                                 exact_order=s["exact_order"], flagged_rate=s["flagged_rate"]))
                order_rows.append(order_counts(df).assign(cutoff=cutoff, fade=fade, true_model=tm))

    orders = pd.concat(order_rows, ignore_index=True)[
        ["cutoff", "fade", "true_model", "start_order", "final_order", "count"]]

    return pd.DataFrame(rows), orders


def summarise_cutoff(table: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Reshape the grid table into one row per (cutoff, fade) for the criterion.

    Columns: one per true model (exact-order recovery), score (their mean), garch11_ge_090
    (the floor), gain_vs_C0F3 (score minus the (C0, F3) score) and clears_margin (the floor
    holds AND the gain is at least calibration.cutoff_grid.margin). It reports only; picking
    the winner (ties to C0, then fewer spikes; no candidate clears -> keep C0, F3) is the
    separate, user-triggered freeze step.
    """
    models = cfg["calibration"]["cutoff_grid"]["models"]
    wide = table.pivot(index=["cutoff", "fade"], columns="true_model", values="exact_order")[models]

    wide["score"] = wide[models].mean(axis=1)             # criterion: mean over the true models
    wide["garch11_ge_090"] = wide["garch11"] >= 0.90      # floor used in calibration
    wide["gain_vs_C0F3"] = wide["score"] - wide.loc[("C0", "F3"), "score"]
    wide["clears_margin"] = wide["garch11_ge_090"] & (wide["gain_vs_C0F3"] >= cfg["calibration"]["cutoff_grid"]["margin"])

    return wide


def order_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Count replications by (start order, final order), orders written like "(1, 1)".

    Input: experiment DataFrame with start_p, start_q, p, q. Output: DataFrame with
    start_order, final_order, count, sorted by count (largest first), ties by label.
    A no_valid_fit row shows its start order as the final order (nothing was fitted).
    When start_p is NaN everywhere (grid_bic has no start order) start_order is the text "n/a".
    """
    # grid_bic rows carry no start order; writing NaN as "(nan, nan)" would be misleading.
    if df["start_p"].isna().all():
        start = pd.Series("n/a", index=df.index)
    else:
        start = "(" + df["start_p"].astype(str) + ", " + df["start_q"].astype(str) + ")"

    out = pd.DataFrame({"start_order": start,
                        "final_order": "(" + df["p"].astype(str) + ", " + df["q"].astype(str) + ")"})

    counts = out.value_counts().reset_index(name="count")   # one row per distinct pair

    return counts.sort_values(["count", "start_order", "final_order"],
                              ascending=[False, True, True], ignore_index=True)


def summarise_calibration(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reshape the calibration table into the two comparison tables used by the calibration criteria.

    Returns (fade_df, mean_df):
      fade_df: index candidate, columns arch1, garch11 (exact-order recovery), score (their
               mean) and garch11_ge_090 (the floor: GARCH(1,1) recovery >= 0.90).
      mean_df: index candidate, columns any_ar_<model> for iid, arch1, garch11 (true m = 0, so
               these are false-AR rates) and ar1_garch11 (true m = 1, so ~1 is good), plus
               ar1_detect = share given exactly AR(1) on ar1_garch11.
    It reports; it does not pick winners (that is a separate, user-triggered step).
    """
    fade = table[table["rule"] == "fade"]
    fade_df = fade.pivot(index="candidate", columns="true_model", values="exact_order")
    fade_df["score"] = fade_df[["arch1", "garch11"]].mean(axis=1)   # criterion: mean of the two
    fade_df["garch11_ge_090"] = fade_df["garch11"] >= 0.90          # floor used in calibration

    mt = table[table["rule"] == "mean_trigger"]
    mean_df = mt.pivot(index="candidate", columns="true_model", values="any_ar_rate")
    mean_df.columns = [f"any_ar_{c}" for c in mean_df.columns]
    mean_df["ar1_detect"] = mt[mt["true_model"] == "ar1_garch11"].set_index("candidate")["m_correct"]

    return fade_df, mean_df


# ---------------------------------------------------------------------------------------
# Order-recovery experiment on the EVAL seeds.
# ---------------------------------------------------------------------------------------

def nests_true(final_p, final_q, true_p: int, true_q: int):
    """True where the chosen GARCH(final_p, final_q) contains the true (true_p, true_q) model.

    Input: final_p, final_q scalars or arrays; true_p, true_q ints. Output: bool (or bool array).
    Rule: final_p >= true_p AND final_q >= true_q. A bigger model contains a smaller one
    (set the extra coefficients to zero), so e.g. (2, 0) nests ARCH(1) = (1, 0), (3, 1) nests (2, 1),
    but (1, 1) does not nest (2, 1). It is weaker than an exact match: over-fitting still counts.
    """
    return np.logical_and(np.asarray(final_p) >= true_p, np.asarray(final_q) >= true_q)


def family_match(start_q, true_q: int):
    """True where the rule's START order is in the right family (variance lag or not).

    Input: start_q scalar or array (q of the start order), true_q int. Output: bool (or bool array).
    Rule: (start_q >= 1) == (true_q >= 1), i.e. the rule chose pure ARCH (q = 0) exactly
    when the truth is pure ARCH, and a GARCH with a variance lag exactly when the truth has one.
    """
    return (np.asarray(start_q) >= 1) == (true_q >= 1)


def mc_se(rate: float, R: int) -> float:
    """Monte Carlo standard error sqrt(p (1 - p) / R) of a share p measured on R replications.

    Input: rate in [0, 1] (NaN passes through as NaN), R int > 0. Output: float.
    Idea: each replication is a yes/no draw, so the share's sampling spread shrinks like 1/sqrt(R).
    """
    return math.sqrt(rate * (1.0 - rate) / R)


def recovery_metrics(df: pd.DataFrame, true_model: str, cfg: dict, selector: str = "hybrid") -> dict:
    """Turn one experiment DataFrame into the recovery rates, each with its Monte Carlo SE.

    Input: df from order_recovery_experiment(..., model_set="recovery"), true_model name in
    recovery.models, cfg, selector ("hybrid" or "grid_bic"). Output: flat dict with
    exact, nesting, family, any_ar_rate, gate_pass, escalation_rate, flagged, no_valid_fit (shares),
    their _se columns (family -> family_se etc.; any_ar_rate -> any_ar_se; escalation_rate ->
    escalation_se) and mean_fits.
    - exact = final (p, q) equals the truth AND a valid fit existed; nesting = nests_true AND a valid fit.
    - family uses the START order (all replications); NaN for grid_bic (no start order).
    - any_ar_rate = share with m > 0 (false-AR rate when the true m is 0; detection rate when it is 1).
    - gate_pass and escalation_rate are NaN for grid_bic (no gate; it always fits many models).
    """
    spec = true_spec(cfg["recovery"]["models"][true_model])
    R = len(df)
    valid = ~df["no_valid_fit"].to_numpy(dtype=bool)   # a replication with no valid fit never counts as recovered
    nan = float("nan")

    # Exact: the chosen (p, q) is the true one, and the fit behind it was valid.
    same_order = ((df["p"] == spec["p"]) & (df["q"] == spec["q"])).to_numpy()

    rates = dict(
        exact=float((same_order & valid).mean()),
        nesting=float((nests_true(df["p"], df["q"], spec["p"], spec["q"]) & valid).mean()),   # array in, array out
        family=nan if selector == "grid_bic" else float(family_match(df["start_q"], spec["q"]).mean()),
        any_ar_rate=float((df["m"] > 0).mean()),
        gate_pass=nan if selector == "grid_bic" else float(df["gate_passed"].astype(bool).mean()),
        escalation_rate=nan if selector == "grid_bic" else float((df["models_fitted"] > 1).mean()),
        flagged=float(df["flagged"].astype(bool).mean()),
        no_valid_fit=float((~valid).mean()))

    # Short names for the SE columns; every rate gets one.
    se_name = dict(exact="exact_se", nesting="nesting_se", family="family_se", any_ar_rate="any_ar_se",
                   gate_pass="gate_pass_se", escalation_rate="escalation_se", flagged="flagged_se",
                   no_valid_fit="no_valid_fit_se")
    out = {}

    for key, rate in rates.items():
        out[key] = rate
        out[se_name[key]] = mc_se(rate, R)

    out["mean_fits"] = float(df["models_fitted"].mean())

    return out


def recovery_table(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the experiment; return (table, orders). Eval seeds only, rules as frozen.

    For every true model in recovery.models and every length n: the hybrid, plus grid_bic when n is in
    recovery.grid_n. recovery.smoke true -> R = recovery.smoke_R and n in recovery.smoke_n; false ->
    R = sim.R and n in sim.n. Replication i uses seed eval.seed_base + i (the same seeds for every
    model and n, so cells are paired). check_seed_ranges guards against overlap with calibration seeds.
    table: one row per (true_model, n, selector): R plus the recovery_metrics columns.
    orders: true_model, n, selector, start_order, final_order, count (grid_bic start_order = "n/a";
    for a no_valid_fit replication the order is the rule's proposal, not a fitted model).
    """
    check_seed_ranges(cfg)
    rec = cfg["recovery"]

    if rec["smoke"]:
        R, lengths = rec["smoke_R"], rec["smoke_n"]
    else:
        R, lengths = cfg["sim"]["R"], cfg["sim"]["n"]

    # check_seed_ranges only covers sim.R eval seeds, so R must not exceed it.
    if R > cfg["sim"]["R"]:
        raise ValueError(f"R = {R} exceeds sim.R = {cfg['sim']['R']}: the eval seed range was not checked for it")

    rows, order_rows = [], []

    for true_model in rec["models"]:
        for n in lengths:
            # The grid costs 30 fits per series, so it only runs at the lengths listed in recovery.grid_n.
            selectors = ["hybrid"] + (["grid_bic"] if n in rec["grid_n"] else [])

            for sel in selectors:
                df = order_recovery_experiment(true_model, n, R, cfg["eval"]["seed_base"], cfg,
                                               selector=sel, model_set="recovery")
                rows.append(dict(true_model=true_model, n=n, selector=sel, R=R,
                                 **recovery_metrics(df, true_model, cfg, sel)))
                order_rows.append(order_counts(df).assign(true_model=true_model, n=n, selector=sel))

    orders = pd.concat(order_rows, ignore_index=True)[
        ["true_model", "n", "selector", "start_order", "final_order", "count"]]

    return pd.DataFrame(rows), orders


def order_count_matrix(orders: pd.DataFrame, true_model: str, n: int, selector: str,
                       pmax: int, qmax: int) -> pd.DataFrame:
    """Final-order counts as a (p x q) grid, for the heatmap.

    Input: orders from recovery_table (final_order written like "(2, 1)"), the cell to show
    (true_model, n, selector) and the grid size pmax, qmax. Output: DataFrame, index p = 0..pmax,
    columns q = 0..qmax, entries = number of replications that ended on that order (0 if none).
    Raises ValueError if some observed order lies outside the grid (it would be dropped silently).
    """
    sub = orders[(orders["true_model"] == true_model) & (orders["n"] == n) & (orders["selector"] == selector)]

    # Pull p and q out of the "(p, q)" text.
    pq = sub["final_order"].str.extract(r"\((\d+), (\d+)\)").astype(int)
    pq.columns = ["p", "q"]
    pq["count"] = sub["count"].to_numpy()

    if len(pq) and (pq["p"].max() > pmax or pq["q"].max() > qmax):
        raise ValueError(f"an observed order exceeds the grid ({pmax}, {qmax})")

    mat = pq.pivot_table(index="p", columns="q", values="count", aggfunc="sum", fill_value=0)

    return mat.reindex(index=range(pmax + 1), columns=range(qmax + 1), fill_value=0).astype(int)


# ---------------------------------------------------------------------------------------
# Post-hoc diagnostic: why does ARCH(1) exact recovery fall with n?
# Hypothesis: the 1.96 / sqrt(n) band is too narrow for the PACF of squares when the 8th moment of the
# returns is infinite (normal ARCH(1): alpha > (1/105)^(1/4), about 0.31). Reports only; no rule changes.
# ---------------------------------------------------------------------------------------

def diagnostic_model_name(alpha: float) -> str:
    """Name under which the diagnostic's ARCH(1) with this alpha is stored in recovery.models.

    Input: alpha float. Output: str such as "arch1_a0.4". The name only has to be unique and stable.
    """
    return f"arch1_a{alpha}"


def diagnostic_cfg(cfg: dict) -> dict:
    """Copy of cfg whose recovery.models also holds one ARCH(1) per diagnostic.alphas entry.

    Input: cfg dict. Output: deep copy (the input is never modified). Each added entry is
    {alphas: [alpha], betas: []}; simulate_recovery_model then sets omega = 1 - alpha, so the
    unconditional variance is 1, as for every model. This lets the diagnostic reuse the
    recovery code (order_recovery_experiment, recovery_metrics) unchanged.
    """
    out = copy.deepcopy(cfg)   # deep copy: the nested recovery.models dict must not be shared with the caller

    for alpha in cfg["diagnostic"]["alphas"]:
        out["recovery"]["models"][diagnostic_model_name(alpha)] = dict(alphas=[alpha], betas=[])

    return out


def seed_ranges(cfg: dict) -> dict:
    """Half-open seed intervals [lo, hi) used by the three Monte Carlo blocks.

    Input: cfg dict. Output: dict name -> (lo, hi) for calibration (seed_base .. +calibration.R),
    eval (seed_base .. +sim.R) and diagnostic (seed_base .. +diagnostic.R).
    """
    cal, ev, dg = cfg["calibration"], cfg["eval"], cfg["diagnostic"]

    return dict(calibration=(cal["seed_base"], cal["seed_base"] + cal["R"]),
                eval=(ev["seed_base"], ev["seed_base"] + cfg["sim"]["R"]),
                diagnostic=(dg["seed_base"], dg["seed_base"] + dg["R"]))


def check_diagnostic_seed_ranges(cfg: dict) -> None:
    """Raise ValueError if any two of the calibration, eval and diagnostic seed ranges overlap.

    Input: cfg dict. Output: None. A diagnostic that reused calibration or eval seeds would not be
    an independent look at the frozen rules, so every pair is compared (not only calibration vs eval).
    """
    for (a, (a_lo, a_hi)), (b, (b_lo, b_hi)) in itertools.combinations(seed_ranges(cfg).items(), 2):
        if a_lo < b_hi and b_lo < a_hi:   # two half-open intervals overlap exactly when each starts before the other ends
            raise ValueError(f"{a} and {b} seed ranges overlap")


def pacf_squares_spike(x, cfg: dict) -> bool:
    """True if the PACF of the squared, demeaned series has a significant lag in lag_min..lag_max.

    Input: x return series (array-like), cfg dict. Output: bool.
    Steps (same as the variance rule, rules.variance_order_rule, but with no model fitted): subtract the
    sample mean (a constant-mean residual), square it (a proxy for variance), take the PACF and the
    white-noise band 1.96 / sqrt(n) from the same acf_pacf_band call, and look for a bar outside the band
    in diagnostic.lag_min..lag_max. In an ARCH(1) the true PACF of squares is zero beyond lag 1, so a spike
    there is a false alarm caused by sampling noise (a band too narrow for heavy-tailed squares gives more of them).
    """
    dg, vc = cfg["diagnostic"], cfg["variance"]

    if dg["lag_max"] > vc["corr_lags"]:
        raise ValueError("diagnostic.lag_max must be <= variance.corr_lags")

    # --- 1. Constant-mean shocks: with no AR term the residual is the series minus its mean.
    shocks = np.asarray(x, dtype=float) - np.mean(x)

    # --- 2. PACF of the squares and the same band the rules use.
    cor = acf_pacf_band(shocks ** 2, vc["corr_lags"], cfg["band_confidence"])
    spikes = significant_lags(cor["pacf"], cor["band"])

    # --- 3. Any bar outside the band in the lag window is a (false) spike.
    return any(dg["lag_min"] <= lag <= dg["lag_max"] for lag in spikes)


def _spike_replication(true_model: str, n: int, seed: int, cfg: dict) -> bool:
    """Simulate one recovery.models series and report whether it has a PACF-of-squares spike (lags 2..5).

    Input: true_model name in recovery.models, length n, seed, cfg. Output: bool.
    Module-level (not nested) so joblib can pickle it for worker processes.
    """
    return pacf_squares_spike(simulate_recovery_model(true_model, n, seed, cfg), cfg)


def false_spike_rate(true_model: str, n: int, R: int, seed_base: int, cfg: dict) -> float:
    """Share of R simulated series with at least one significant PACF-of-squares lag in lag_min..lag_max.

    Input: true_model in recovery.models, n, R, seed_base (replication i uses seed_base + i), cfg.
    Output: float in [0, 1]. For white noise each lag is outside a 95% band 5% of the time, so with four
    lags about 1 - 0.95**4 = 0.19 is expected by chance (my arithmetic, assumes independent lags); the
    diagnostic asks whether ARCH(1) with a large alpha gives MORE than that, and more as n grows.
    """
    # Replications are independent, so they run in parallel; the result does not depend on the worker count.
    flags = Parallel(n_jobs=cfg["recovery"]["n_jobs"])(
        delayed(_spike_replication)(true_model, n, seed_base + i, cfg) for i in range(R))

    return float(np.mean(flags))


def diagnostic_arch1_table(cfg: dict) -> pd.DataFrame:
    """Run the diagnostic: hybrid recovery and false-spike rate for ARCH(1) at each (alpha, n).

    Input: cfg dict (needs the diagnostic section). Output: DataFrame, one row per (alpha, n) with
    columns alpha, n, R, exact, nesting, false_spike_rate, exact_se, nesting_se, false_spike_se.
    exact and nesting come from the existing recovery code (frozen rules, hybrid selector); the false
    spike rate is measured on the same simulated series (same seeds). Seeds are diagnostic.seed_base + i;
    check_diagnostic_seed_ranges guards against overlap with the calibration and eval ranges.
    SE = sqrt(p (1 - p) / R). No rule, threshold or existing result is changed.
    """
    check_diagnostic_seed_ranges(cfg)
    dg = cfg["diagnostic"]
    dcfg = diagnostic_cfg(cfg)   # recovery.models gains the diagnostic ARCH(1) models, on a copy
    rows = []

    for alpha in dg["alphas"]:
        name = diagnostic_model_name(alpha)

        for n in dg["n"]:
            # Recovery of the true order by the frozen hybrid, exactly as the experiment does it.
            df = order_recovery_experiment(name, n, dg["R"], dg["seed_base"], dcfg,
                                           selector="hybrid", model_set="recovery")
            met = recovery_metrics(df, name, dcfg, "hybrid")

            # Chance-spike rate of the PACF of squares on the SAME series (same seeds, no model fitted).
            spike = false_spike_rate(name, n, dg["R"], dg["seed_base"], dcfg)

            rows.append(dict(alpha=alpha, n=n, R=dg["R"], exact=met["exact"], nesting=met["nesting"],
                             false_spike_rate=spike, exact_se=met["exact_se"], nesting_se=met["nesting_se"],
                             false_spike_se=mc_se(spike, dg["R"])))

    return pd.DataFrame(rows)


def diagnostic_verdict(table: pd.DataFrame, cfg: dict) -> dict:
    """Apply the fixed interpretation to a diagnostic table.

    Input: table from diagnostic_arch1_table (or its CSV), cfg. Output: dict with the three changes
    (value at the largest n minus value at the smallest n), the three booleans and verdict
    ("supported" or "unexplained"). With tol = diagnostic.change_tol:
      low alpha (smallest in diagnostic.alphas): exact does not fall, change >= -tol;
      high alpha (largest): exact falls, change < -tol, AND false_spike_rate rises, change > +tol.
    Supported only if all three hold. 1e-9 absorbs floating-point error at the boundary.
    "Supported" means consistent with the hypothesis, not proved.
    """
    dg = cfg["diagnostic"]
    tol, eps = dg["change_tol"], 1e-9
    low_a, high_a = min(dg["alphas"]), max(dg["alphas"])
    n_lo, n_hi = min(dg["n"]), max(dg["n"])
    cells = table.set_index(["alpha", "n"])

    def change(alpha: float, col: str) -> float:
        """Value of col at the largest n minus its value at the smallest n, for one alpha."""
        return float(cells.loc[(alpha, n_hi), col] - cells.loc[(alpha, n_lo), col])

    low_exact = change(low_a, "exact")
    high_exact = change(high_a, "exact")
    high_spike = change(high_a, "false_spike_rate")

    low_flat = low_exact >= -tol - eps          # low alpha does not fall
    high_falls = high_exact < -tol - eps        # high alpha falls by more than tol
    spike_rises = high_spike > tol + eps        # chance spikes become more frequent as n grows

    return dict(change_exact_low_alpha=low_exact, change_exact_high_alpha=high_exact,
                change_spike_high_alpha=high_spike, low_alpha_flat=bool(low_flat),
                high_alpha_falls=bool(high_falls), high_alpha_spike_rises=bool(spike_rises),
                verdict="supported" if (low_flat and high_falls and spike_rises) else "unexplained")
