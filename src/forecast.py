"""Rolling-origin one-day-ahead variance forecasts with periodic refit. No look-ahead.

Design:
- The test period is the last `test_fraction` of the sample. The training window EXPANDS:
  the forecast for day i uses returns[:i] only, never anything later.
- Every `forecast.refit_every` test days the WHOLE selector is re-run on the data so far.
- Between refits the order and parameters are kept, but the conditional variance is
  re-filtered through the newest returns (baselines.next_variance), so yesterday's shock
  still moves today's forecast.
- A refit with no valid fit uses the rolling variance for its whole block and says so.
- r2 is the squared return of the target day: a NOISY proxy for the true variance (one draw
  of e_t^2 has expectation h_t but a huge spread), so single-day errors mean little and only
  averages over many days (QLIKE, DM tests) are informative.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src import baselines
from src.baselines import next_variance, rolling_variance
from src.data import ROOT, load_fx, to_log_returns

COLUMNS = ["date", "pair", "selector", "h", "r2", "order", "models_fitted", "flagged",
           "no_valid_fit", "refit_date",
           # Observation only: what the refit in force cost and why fits were rejected.
           "refit_models", "fit_status", "reject_reason", "rejected_persistence",
           # observation only: error distribution of the model in force ("" when there is no model).
           "dist"]


def n_test_days(n: int, cfg: dict) -> int:
    """Number of test days: the last test_fraction of n observations (rounded, not truncated).

    Input: n sample size, cfg. Output: int. round() avoids float slips such as 290 * 0.2 = 57.999...
    """
    return int(round(n * cfg["test_fraction"]))


def _run_block(x: np.ndarray, selector_name: str, cfg: dict, i0: int, i1: int) -> list[dict]:
    """Refit once on x[:i0], then forecast days i0..i1-1 (one refit block).

    Input: x all returns (only x[:i] is ever used for day i), selector name, cfg, block bounds.
    Output: list of row dicts (position i, h, r2, order, models_fitted, flagged, no_valid_fit,
    refit_models, fit_status, reject_reason, rejected_persistence, dist).
    Module-level so joblib can pickle it for worker processes.
    """
    window = cfg["baselines"]["rolling_window"]

    # --- Refit: the whole selector sees the data up to (not including) day i0.
    fit = baselines.SELECTORS[selector_name](x[:i0], cfg)
    order = "" if fit["order"] is None else str(tuple(fit["order"]))

    rows = []
    for i in range(i0, i1):
        # --- Variance forecast for day i, from returns[:i] only.
        if i == i0:
            h = fit["h_next"]                                # already computed at the refit
        elif fit["spec"] is None:
            h = rolling_variance(x[:i], window)              # rolling baseline or fallback
        else:
            h = next_variance(x[:i], fit["spec"])            # fixed params, fresh variance filter

        rows.append(dict(
            i=i, h=h, r2=float(x[i] ** 2), order=order,
            # Cost is counted once per refit (first day of the block), so a column sum = total fits.
            models_fitted=fit["models_fitted"] if i == i0 else 0,
            # flagged / no_valid_fit describe the model in force, so they repeat on every row.
            flagged=fit["flagged"], no_valid_fit=fit["no_valid_fit"],
            # The four columns below describe the refit in force, so they repeat on every row.
            # refit_models is NOT zeroed after day one (unlike models_fitted): it is per-refit cost.
            refit_models=fit["models_fitted"],
            fit_status="no_valid_fit" if fit["no_valid_fit"] else "valid",
            # .get so a selector without these keys (e.g. a test stub) still works: "" and NaN.
            reject_reason=fit.get("reject_reason", ""),
            rejected_persistence=fit.get("rejected_persistence", float("nan")),
            # dist is read from the chosen model's spec (never recomputed); no model -> "".
            dist="" if fit["spec"] is None else fit["spec"]["dist"]))

    return rows


def rolling_refit_forecast(returns: pd.Series, selector_name: str, cfg: dict) -> pd.DataFrame:
    """Expanding-window rolling forecast of the next-day variance for one pair and one selector.

    Input: returns = log returns x100 as a pd.Series with a date index (its name is the pair),
    selector_name in baselines.SELECTORS, cfg. Output: DataFrame with COLUMNS, one row per test
    day: date (the forecast TARGET day), h (forecast variance), r2 (that day's squared return),
    order, models_fitted, flagged, no_valid_fit, refit_date (last training date of the refit
    in force), refit_models (fits spent in that refit), fit_status (valid | no_valid_fit),
    reject_reason and rejected_persistence (see baselines.py), dist (error distribution of the model
    in force, "" if none). With forecast.smoke true only the last forecast.smoke_days test days are used.
    """
    fc = cfg["forecast"]
    x = returns.to_numpy(dtype=float)
    n = len(x)

    # --- Test period: the last test_fraction of the sample (shortened in smoke mode).
    n_test = n_test_days(n, cfg)
    if fc["smoke"]:
        n_test = min(n_test, fc["smoke_days"])
    start = n - n_test

    # --- Refit blocks of refit_every days; the last block may be shorter.
    step = fc["refit_every"]
    bounds = [(i0, min(i0 + step, n)) for i0 in range(start, n, step)]

    # Blocks are independent (each refits on its own data), so run them in parallel.
    # No randomness is involved, so the result does not depend on n_jobs.
    blocks = Parallel(n_jobs=fc["n_jobs"])(
        delayed(_run_block)(x, selector_name, cfg, i0, i1) for i0, i1 in bounds)

    # --- Attach dates: the target day, and the last training day of each block's refit.
    out = []
    for (i0, _), rows in zip(bounds, blocks):
        for r in rows:
            r["date"] = returns.index[r.pop("i")]
            r["refit_date"] = returns.index[i0 - 1]
            r["pair"] = returns.name if returns.name is not None else ""
            r["selector"] = selector_name
            out.append(r)

    return pd.DataFrame(out)[COLUMNS]


def pairs_to_run(cfg: dict) -> list[str]:
    """Pairs to forecast: all of cfg['pairs'], or only the first one in smoke mode."""
    return cfg["pairs"][:1] if cfg["forecast"]["smoke"] else list(cfg["pairs"])


def run_pair(pair: str, cfg: dict) -> pd.DataFrame:
    """Forecast one pair with all five selectors and write results/forecasts_<pair>.csv.

    Input: FRED series id, cfg. Output: the stacked DataFrame (also saved as CSV, no index).
    """
    # Load the cached price series and convert to log returns x100 (project convention).
    returns = to_log_returns(load_fx(pair, cfg["start"], cfg=cfg))

    frames = [rolling_refit_forecast(returns, name, cfg) for name in baselines.SELECTORS]
    df = pd.concat(frames, ignore_index=True)

    # CSV, not parquet: no new dependency.
    path = Path(ROOT) / cfg["paths"]["results_dir"] / f"forecasts_{pair}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

    return df


def run_all(cfg: dict) -> dict[str, pd.DataFrame]:
    """Run run_pair for every pair in pairs_to_run(cfg). Output: dict pair -> DataFrame."""
    return {pair: run_pair(pair, cfg) for pair in pairs_to_run(cfg)}
