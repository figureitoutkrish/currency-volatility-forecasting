"""Forecast evaluation: QLIKE and MSE losses, Diebold-Mariano test, and the result tables.

What this file is for:
- qlike, mse, diebold_mariano are hand-rolled.
- loss_table and dm_table turn the per-day forecast CSVs of src/forecast.py into the two result
  tables; cumulative_qlike_diff prepares the data of the cumulative-loss figure; max_abs_h_diff is the
  regression check between two runs; block_qlike_diff splits one pair's hybrid-vs-fixed QLIKE
  difference by refit block. scripts/run_forecasts.py only calls these and writes the files.

Statistics idea in one paragraph: we never observe the true variance, only the squared return r2
of each day, which is a very noisy copy of it. A loss function scores a forecast h against r2.
QLIKE = ln(h) + r2/h is robust to that noise (it is minimised in expectation by the true
variance and punishes under-forecasting more than over-forecasting). The Diebold-Mariano test
asks whether the average loss of selector A differs from selector B by more than chance, using a
long-run variance (Newey-West) because daily loss differences are correlated over time.
Lower loss is better everywhere. Negative DM means A (the first-named selector) is better.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

from src.data import load_config


def _check_inputs(h, r2) -> tuple[np.ndarray, np.ndarray]:
    """Validate forecasts and proxies and return them as float arrays.

    Input: h variance forecasts, r2 squared returns (array-like, same length).
    Output: (h, r2) as 1-D float arrays. Raises ValueError if the lengths differ, anything is
    non-finite, any h <= 0 (ln(h) and r2/h are undefined) or any r2 < 0 (a square cannot be negative).
    """
    h = np.atleast_1d(np.asarray(h, dtype=float))
    r2 = np.atleast_1d(np.asarray(r2, dtype=float))

    # Same number of forecasts and targets, otherwise the pairs make no sense.
    if h.shape != r2.shape:
        raise ValueError(f"h and r2 must have the same shape, got {h.shape} and {r2.shape}")

    # NaN/inf would silently poison every average downstream, so stop here.
    if not (np.isfinite(h).all() and np.isfinite(r2).all()):
        raise ValueError("h and r2 must be finite")

    # A variance forecast must be strictly positive: ln(0) = -inf and r2/0 = inf.
    if (h <= 0).any():
        raise ValueError("every variance forecast h must be > 0")

    if (r2 < 0).any():
        raise ValueError("r2 (a squared return) must be >= 0")

    return h, r2


def qlike(h, r2) -> np.ndarray:
    """Per-day QLIKE loss ln(h) + r2 / h (primary loss).

    Input: h variance forecasts (> 0), r2 squared returns (>= 0). Output: array of daily losses.
    Idea: if the forecast is exactly right on average (E[r2] = h) the expected loss is smallest;
    a forecast that is too small is punished harder through the r2/h term than one too large.
    r2 = 0 is fine (the loss is then ln(h)).
    """
    h, r2 = _check_inputs(h, r2)

    # Elementwise: the natural log of the forecast plus the proxy relative to the forecast.
    return np.log(h) + r2 / h


def mse(h, r2) -> np.ndarray:
    """Per-day squared error (r2 - h)^2 (secondary loss).

    Input: h variance forecasts (> 0), r2 squared returns (>= 0). Output: array of daily losses.
    Idea: plain distance to the proxy. It is dominated by the few days with huge r2, so it is
    only a robustness check next to QLIKE.
    """
    h, r2 = _check_inputs(h, r2)

    return (r2 - h) ** 2


# Name -> loss function; config evaluate.losses uses these names.
LOSSES = dict(qlike=qlike, mse=mse)


def nw_lag(T: int, cfg: dict) -> int:
    """Newey-West bandwidth L = floor(lag_c * (T / lag_ref) ** lag_exp) (defaults 4, 100, 2/9).

    Input: T number of days, cfg (evaluate.dm.* parameters). Output: int L.
    Idea: more days allow a longer memory in the long-run variance; the exponent 2/9 is the
    usual rule of thumb for the Bartlett kernel.
    """
    dm = cfg["evaluate"]["dm"]

    return int(np.floor(dm["lag_c"] * (T / dm["lag_ref"]) ** dm["lag_exp"]))


def newey_west_lrv(d, L: int) -> float:
    """Newey-West long-run variance of a series with Bartlett weights.

    LRV = gamma_0 + 2 * sum_{k=1..L} (1 - k/(L+1)) * gamma_k, where
    gamma_k = (1/T) * sum_{t=k+1..T} (d_t - dbar)(d_{t-k} - dbar).
    Input: d 1-D array, L bandwidth (0 <= L < T). Output: float.
    Idea: the variance of the MEAN of d is not var(d)/T when d is autocorrelated; the weighted
    sum of autocovariances corrects that, and the Bartlett weights keep it non-negative.
    """
    d = np.asarray(d, dtype=float)
    T = len(d)

    # Centre once; every gamma_k uses deviations from the sample mean.
    dev = d - d.mean()

    # gamma_0 is the ordinary (divide-by-T) variance.
    lrv = float(dev @ dev) / T

    # Add the weighted autocovariances at lags 1..L (counted twice: lag +k and lag -k).
    for k in range(1, L + 1):
        gamma_k = float(dev[k:] @ dev[:-k]) / T
        lrv += 2.0 * (1.0 - k / (L + 1.0)) * gamma_k

    return lrv


def diebold_mariano(loss_a, loss_b, L: int | None = None, cfg: dict | None = None) -> dict:
    """Diebold-Mariano test of equal predictive accuracy of A and B (two-sided).

    Input: loss_a, loss_b daily losses of the same days (lower = better); L Newey-West bandwidth
    (None = nw_lag rule from cfg; cfg None = load config.yaml). Output: dict(mean_diff, dm, p, L,
    status). d_t = loss_a - loss_b; DM = mean(d) / sqrt(LRV / T); p = 2 * (1 - Phi(|DM|)) from
    the standard normal, with no small-sample correction. Negative DM means A has the lower loss.
    status is "ok", "identical" (the differential has zero variance, so the two forecasts score
    the same every day and no test exists) or "nonpositive_lrv" (numerical guard). dm and p are
    NaN unless status is "ok".
    """
    a = np.atleast_1d(np.asarray(loss_a, dtype=float))
    b = np.atleast_1d(np.asarray(loss_b, dtype=float))

    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("loss_a and loss_b must be 1-D arrays of the same length")
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("losses must be finite")
    T = len(a)
    if T < 2:
        raise ValueError("need at least 2 days")

    # Bandwidth: given, or the config rule. It must be shorter than the sample.
    if L is None:
        L = nw_lag(T, cfg if cfg is not None else load_config())
    if not 0 <= L < T:
        raise ValueError(f"bandwidth L must satisfy 0 <= L < T, got L={L}, T={T}")

    # Loss differential: negative on days where A beat B.
    d = a - b
    mean_diff = float(d.mean())

    # Zero variance (all differences equal, in practice all zero): same forecasts, no test.
    if np.ptp(d) == 0.0:
        return dict(mean_diff=mean_diff, dm=float("nan"), p=float("nan"), L=L, status="identical")

    lrv = newey_west_lrv(d, L)
    if not lrv > 0.0:
        return dict(mean_diff=mean_diff, dm=float("nan"), p=float("nan"), L=L,
                    status="nonpositive_lrv")

    # Statistic: the mean difference in units of its own standard error.
    dm = mean_diff / np.sqrt(lrv / T)

    # Two-sided p-value from the standard normal (sf = 1 - cdf, accurate in the tails).
    p = 2.0 * float(norm.sf(abs(dm)))

    return dict(mean_diff=mean_diff, dm=float(dm), p=p, L=L, status="ok")


def _paired(fc: pd.DataFrame, a: str, b: str, sample: str) -> pd.DataFrame:
    """Line up selector a and selector b day by day for one pair.

    Input: fc stacked forecast table of ONE pair, selector names a and b, sample = "all" (every
    test day) or "common_valid" (only days where BOTH had fit_status == valid).
    Output: DataFrame[h_a, h_b, r2] indexed by date. Raises if the two selectors do not cover the
    same days or disagree on r2 (that would be a bug upstream).
    """
    da = fc[fc["selector"] == a].set_index("date")
    db = fc[fc["selector"] == b].set_index("date")

    # Both selectors must forecast the same target days with the same target r2.
    if not da.index.equals(db.index):
        raise ValueError(f"{a} and {b} do not cover the same days")
    if not np.array_equal(da["r2"].to_numpy(), db["r2"].to_numpy()):
        raise ValueError(f"{a} and {b} disagree on r2")

    out = pd.DataFrame({"h_a": da["h"], "h_b": db["h"], "r2": da["r2"]})

    if sample == "common_valid":
        # Drop days on which either selector used the rolling-variance fallback.
        keep = (da["fit_status"] == "valid") & (db["fit_status"] == "valid")
        out = out[keep.to_numpy()]
    elif sample != "all":
        raise ValueError(f"unknown sample {sample!r}")

    return out


def dm_table(forecasts: dict[str, pd.DataFrame], cfg: dict) -> pd.DataFrame:
    """Diebold-Mariano table for every pair x loss x sample x comparison.

    Input: forecasts = dict pair -> stacked forecast DataFrame (from forecast.run_all or CSVs),
    cfg. Output: DataFrame[pair, loss, sample, A, B, mean_diff, dm, p, L, status, n_days, dm_L20, p_L20].
    Statuses: ok, identical (h series identical to the last digit, or zero-variance differential),
    too_few_days (fewer than evaluate.dm.min_days common days). Raw p-values; no multiplicity
    correction is applied.
    dm_L20 / p_L20 = the same test with bandwidth L = forecast.refit_every (the name assumes 20),
    a sensitivity check: a result is "robust" only if p and p_L20 are both below 0.05. They are NaN
    unless status is "ok" and the bandwidth is shorter than the sample.
    """
    ev = cfg["evaluate"]
    alt_L = cfg["forecast"]["refit_every"]   # sensitivity bandwidth: one refit block
    rows = []

    for pair, fc in forecasts.items():
        for loss in ev["losses"]:
            for sample in ev["samples"]:
                for a, b in ev["comparisons"]:
                    pr = _paired(fc, a, b, sample)
                    T = len(pr)
                    row = dict(pair=pair, loss=loss, sample=sample, A=a, B=b, n_days=T)

                    # Too short for a meaningful test: say so instead of printing a shaky p.
                    if T < ev["dm"]["min_days"]:
                        row.update(mean_diff=float("nan"), dm=float("nan"), p=float("nan"),
                                   L=float("nan"), status="too_few_days",
                                   dm_L20=float("nan"), p_L20=float("nan"))
                        rows.append(row)
                        continue

                    la = LOSSES[loss](pr["h_a"], pr["r2"])
                    lb = LOSSES[loss](pr["h_b"], pr["r2"])
                    res = diebold_mariano(la, lb, cfg=cfg)

                    # Identical forecasts (max abs difference exactly 0) never get a test.
                    if float(np.max(np.abs(pr["h_a"].to_numpy() - pr["h_b"].to_numpy()))) == 0.0:
                        res.update(dm=float("nan"), p=float("nan"), status="identical")

                    # Sensitivity: same losses, bandwidth = one refit block. Only when the
                    # primary test ran and the bandwidth is shorter than the sample (else it would raise).
                    alt_dm = alt_p = float("nan")
                    if res["status"] == "ok" and alt_L < T:
                        alt = diebold_mariano(la, lb, L=alt_L)
                        if alt["status"] == "ok":
                            alt_dm, alt_p = alt["dm"], alt["p"]

                    row.update(res)
                    row.update(dm_L20=alt_dm, p_L20=alt_p)
                    rows.append(row)

    cols = ["pair", "loss", "sample", "A", "B", "mean_diff", "dm", "p", "L", "status", "n_days",
            "dm_L20", "p_L20"]

    return pd.DataFrame(rows)[cols]


def loss_table(forecasts: dict[str, pd.DataFrame], cfg: dict) -> pd.DataFrame:
    """Loss and cost table, one row per pair x selector.

    Input: forecasts = dict pair -> stacked forecast DataFrame, cfg.
    Output: DataFrame[pair, selector, qlike, mse, models_fitted, refits, escalation_rate,
    modal_order_share, agree_grid, n_both_valid, fallback_share, valid_day_qlike]:
    - qlike, mse: mean over ALL test days (fallback days included).
    - models_fitted: total fits over the test period; refits: number of refit blocks.
    - escalation_rate: share of refits that needed more than one fit; NaN for the grid selector
      (it always fits many models, so the share would be 1 by construction).
    - modal_order_share: share of refits using the selector's most common order (NaN: no order).
    - agree_grid: share of refits whose order equals grid_bic's at the same refit, counted ONLY on
      refits where both selectors had a valid fit (NaN: no order, or no such refit).
    - n_both_valid: how many refits that agree_grid is based on (NaN: no order).
    - fallback_share: share of refits with no valid fit (rolling variance used instead).
    - valid_day_qlike: mean QLIKE over the selector's own valid days only.
    For a no_valid_fit refit the order is the rule's proposal, not a fitted model.
    """
    grid_name = cfg["evaluate"]["grid_selector"]
    rows = []

    for pair, fc in forecasts.items():
        # One row per refit block: the refit columns repeat inside a block, so keep the first row.
        refits = fc.drop_duplicates(["selector", "refit_date"])

        # grid_bic's order and fit status at each refit date, to measure agreement with it.
        grid_rows = refits[refits["selector"] == grid_name].set_index("refit_date")
        grid = grid_rows["order"].fillna("")
        grid_status = grid_rows["fit_status"]

        for sel, g in fc.groupby("selector", sort=False):
            r = refits[refits["selector"] == sel]
            qd = qlike(g["h"], g["r2"])
            valid_day = (g["fit_status"] == "valid").to_numpy()

            # Orders are strings like "(1, 1)"; an empty string means no model (rolling_var).
            order = r["order"].fillna("").astype(str)
            has_order = bool((order != "").any())

            if has_order:
                modal = float(order.value_counts(normalize=True).max())

                # Agreement only where BOTH had a valid fit: a no_valid_fit refit logs a proposal, not a model.
                both = ((r["fit_status"].to_numpy() == "valid")
                        & (grid_status.reindex(r["refit_date"]).to_numpy() == "valid"))
                n_both = int(both.sum())
                same = order.to_numpy() == grid.reindex(r["refit_date"]).to_numpy()
                agree = float(same[both].mean()) if n_both > 0 else float("nan")
            else:
                modal = agree = n_both = float("nan")

            rows.append(dict(
                pair=pair, selector=sel,
                qlike=float(qd.mean()),
                mse=float(mse(g["h"], g["r2"]).mean()),
                models_fitted=int(g["models_fitted"].sum()),
                refits=int(len(r)),
                # grid_bic always fits many models, so its escalation share is 1 by construction: NaN.
                escalation_rate=float("nan") if sel == grid_name else float((r["refit_models"] > 1).mean()),
                modal_order_share=modal,
                agree_grid=agree,
                n_both_valid=n_both,
                fallback_share=float((r["fit_status"] == "no_valid_fit").mean()),
                valid_day_qlike=float(qd[valid_day].mean()) if valid_day.any() else float("nan")))

    return pd.DataFrame(rows)


def cumulative_qlike_diff(fc: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Cumulative QLIKE difference of every selector against the baseline, for one pair.

    Input: fc stacked forecast table of ONE pair, cfg (evaluate.figure_baseline).
    Output: DataFrame indexed by date, one column per selector except the baseline, holding the
    running sum of (QLIKE_selector - QLIKE_baseline). A line that falls means the selector beats
    the baseline; its final value is T times the mean difference.
    """
    base = cfg["evaluate"]["figure_baseline"]

    # Daily QLIKE per selector, one column each (rows = target dates).
    daily = {sel: pd.Series(qlike(g["h"], g["r2"]), index=g["date"].to_numpy())
             for sel, g in fc.groupby("selector", sort=False)}
    wide = pd.DataFrame(daily)

    # Subtract the baseline column, drop it, then accumulate over time.
    diff = wide.drop(columns=base).sub(wide[base], axis=0)

    return diff.cumsum()


def block_qlike_diff(fc: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Split the QLIKE difference of two selectors by refit block, for one pair.

    Input: fc stacked forecast table of ONE pair (needs the dist
    column), cfg (evaluate.block_compare = [A, B]). Output: one row per refit block in which BOTH
    selectors had fit_status valid, columns refit_date, <A>_order, <A>_dist, <B>_order, n_days,
    block_qlike_diff (SUM over the block's days of QLIKE_A - QLIKE_B; negative = A better) and
    share_of_total (block value / sum over the listed blocks; NaN if that sum is exactly 0).
    Idea: shows whether a pooled difference comes from many blocks or from a few; it is a
    description, not a test. Sums rather than means, so the shares add up to 1.
    """
    a, b = cfg["evaluate"]["block_compare"]

    if "dist" not in fc.columns:
        raise ValueError("forecast table has no 'dist' column: rerun the forecasts")

    da = fc[fc["selector"] == a].set_index("date")
    db = fc[fc["selector"] == b].set_index("date")

    # Both selectors must cover the same days and the same refit schedule, otherwise blocks do not line up.
    if not da.index.equals(db.index):
        raise ValueError(f"{a} and {b} do not cover the same days")
    if not np.array_equal(da["refit_date"].to_numpy(), db["refit_date"].to_numpy()):
        raise ValueError(f"{a} and {b} do not share the same refit dates")

    # Daily QLIKE difference A - B (the same sign convention as the DM table).
    diff = qlike(da["h"], da["r2"]) - qlike(db["h"], db["r2"])

    daily = pd.DataFrame({
        "refit_date": da["refit_date"].to_numpy(), "diff": diff,
        # fit_status, order and dist repeat on every row of a block, so "first" below reads the block's value.
        "both_valid": (da["fit_status"] == "valid").to_numpy() & (db["fit_status"] == "valid").to_numpy(),
        "order_a": da["order"].to_numpy(), "dist_a": da["dist"].to_numpy(), "order_b": db["order"].to_numpy()})

    blocks = daily.groupby("refit_date", sort=True).agg(
        order_a=("order_a", "first"), dist_a=("dist_a", "first"), order_b=("order_b", "first"),
        both_valid=("both_valid", "first"), n_days=("diff", "size"),
        block_qlike_diff=("diff", "sum")).reset_index()

    # Keep blocks where both had a valid fit; the share is relative to the total over exactly these blocks.
    blocks = blocks[blocks["both_valid"]].drop(columns="both_valid")
    total = float(blocks["block_qlike_diff"].sum())
    blocks["share_of_total"] = blocks["block_qlike_diff"] / total if total != 0.0 else float("nan")

    return blocks.rename(columns={"order_a": f"{a}_order", "dist_a": f"{a}_dist",
                                  "order_b": f"{b}_order"}).reset_index(drop=True)


def max_abs_h_diff(new: pd.DataFrame, old: pd.DataFrame) -> float:
    """Largest absolute difference between the h columns of two runs (regression check).

    Input: new and old stacked forecast tables (columns date, selector, h). Output: float; 0.0
    means every forecast is unchanged. Raises if the two runs do not cover the same
    (date, selector) keys, because then the comparison would be meaningless.
    """
    key = ["date", "selector"]
    n = new[key + ["h"]].copy()
    o = old[key + ["h"]].copy()

    # Same date type on both sides (one may come from a CSV, where dates are strings).
    n["date"] = pd.to_datetime(n["date"])
    o["date"] = pd.to_datetime(o["date"])

    # Inner join on (date, selector); equal sizes prove that nothing was dropped.
    m = n.merge(o, on=key, suffixes=("_new", "_old"))
    if not (len(m) == len(n) == len(o)):
        raise ValueError(f"runs cover different rows: new {len(n)}, old {len(o)}, matched {len(m)}")

    return float((m["h_new"] - m["h_old"]).abs().max())
