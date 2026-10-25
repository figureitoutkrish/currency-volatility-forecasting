"""Pure helpers for the Streamlit dashboard: everything app.py needs that is not a widget.

app.py may only create widgets and call functions; slicing by the as-of date, building tables,
turning rule output into sentences and reading saved CSVs all live here, so they can be unit
tested without Streamlit. Nothing in this file changes a selector rule (the rules are frozen) and nothing
fits a model except live_recovery (a thin wrapper over simulate.order_recovery_experiment).

No look-ahead: every function that takes data takes only the series it is given; the as-of cut is
made once, by slice_asof, before anything else sees the data.
"""
import datetime as dt
import math

import re

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.data import ROOT
from src.rules import mean_order_rule, variance_order_rule
from src.selector import _mean_resid
from src.simulate import order_counts, order_recovery_experiment, recovery_metrics, true_spec
from src.stat_tests import adf_kpss, arch_lm, ljung_box

# (rate column, its Monte Carlo SE column) in the order the recovery table shows them.
RATES = [("exact", "exact_se"), ("nesting", "nesting_se"), ("family", "family_se"),
         ("any_ar_rate", "any_ar_se"), ("gate_pass", "gate_pass_se"),
         ("escalation_rate", "escalation_se"), ("flagged", "flagged_se"),
         ("no_valid_fit", "no_valid_fit_se")]


# ---------------------------------------------------------------------------------------
# As-of cut and the Data tab
# ---------------------------------------------------------------------------------------

def slice_asof(x, asof):
    """Keep only the observations dated on or before `asof` (the no-look-ahead cut).

    Input: x Series or DataFrame with a DatetimeIndex, asof a date, datetime or "YYYY-MM-DD" text.
    Output: the same type, last index <= asof. Idea: anything chosen or fitted "as of" a day may
    use that day's data and earlier, never later; making the cut once, here, keeps the tabs honest.
    """
    return x.loc[:pd.Timestamp(asof)]


def asof_bounds(returns: pd.Series, min_obs: int) -> tuple[dt.date, dt.date]:
    """Earliest and latest allowed as-of dates for the slider.

    Input: returns Series (DatetimeIndex), min_obs the fewest returns the rules need (app.min_obs).
    Output: (first_date, last_date) as datetime.date; the first date is the min_obs-th return, so
    the slider can never leave fewer than min_obs observations. Raises ValueError if there are
    fewer than min_obs returns in total.
    """
    if len(returns) < min_obs:
        raise ValueError(f"only {len(returns)} returns, need at least {min_obs}")

    return returns.index[min_obs - 1].date(), returns.index[-1].date()


def summary_table(returns: pd.Series) -> pd.DataFrame:
    """One-row summary of the returns: n, mean, std, skew, excess kurtosis, zero-return count.

    Input: returns Series (percent log returns). Output: DataFrame with one row and columns
    n, mean, std, skew, excess_kurtosis, zero_returns. Same pandas conventions as the exploratory analysis
    (std with ddof = 1, bias-adjusted skew and kurtosis; kurtosis is already "excess", 0 = normal).
    Idea: fat tails (large excess kurtosis) and many exact-zero days (stale quotes) matter for the
    ACF band and for the likelihood, so they are shown before any modelling.
    """
    return pd.DataFrame([dict(n=int(returns.count()), mean=returns.mean(), std=returns.std(),
                              skew=returns.skew(), excess_kurtosis=returns.kurt(),
                              zero_returns=int((returns == 0).sum()))], index=["returns (x100)"])


def stationarity_table(price: pd.Series, returns: pd.Series, cfg: dict) -> pd.DataFrame:
    """ADF and KPSS on the price level and on the returns, one row each.

    Input: price and returns Series, cfg (app.value_labels). Output: DataFrame with columns series, adf_stat,
    adf_p, kpss_stat, kpss_p, verdict (from stat_tests.adf_kpss: stationary / unit_root / inconclusive,
    shown as Stationary / Unit root / Inconclusive).
    Idea: a price level normally looks like a unit root and returns look stationary; that is the
    reason the study models returns.
    """
    rows = [dict(series=name, **adf_kpss(x)) for name, x in [("Price level", price), ("Returns", returns)]]

    return label_values(pd.DataFrame(rows), cfg)


# ---------------------------------------------------------------------------------------
# Diagnose tab
# ---------------------------------------------------------------------------------------

def diagnostic_table(returns, cfg: dict) -> pd.DataFrame:
    """Ljung-Box (returns and squares) and ARCH-LM numbers, with the same lags as the gate.

    Input: returns array-like, cfg (gate.lb_lags, gate.lm_lags). Output: DataFrame with columns
    test, series, lag, statistic, p_value. Squares are those of the demeaned returns (EDA convention).
    Idea: a small Ljung-Box p on returns says the mean is not constant; a small p on the squares and a
    small ARCH-LM p say the variance clusters. These are computed on the raw returns, whereas the
    rules apply them to mean-model residuals, so the numbers can differ slightly when m > 0.
    """
    x = np.asarray(returns, dtype=float)
    g = cfg["gate"]
    rows = []

    # Ljung-Box on the returns and on their squares, at every gate lag.
    for name, series in [("returns", x), ("squared returns", (x - x.mean()) ** 2)]:
        lb = ljung_box(series, g["lb_lags"])
        rows += [dict(test="Ljung-Box", series=name, lag=int(r.lag), statistic=float(r.lb_stat),
                      p_value=float(r.lb_pvalue)) for r in lb.itertuples()]

    # ARCH-LM: regress squared (demeaned) returns on their own lags; small p = ARCH effect.
    lm = arch_lm(x, g["lm_lags"])
    rows.append(dict(test="ARCH-LM", series="returns", lag=g["lm_lags"],
                     statistic=float(lm["lm_stat"]), p_value=float(lm["lm_p"])))

    return pd.DataFrame(rows)


def rule_readout(returns, cfg: dict) -> dict:
    """Run the frozen mean and variance rules on `returns` and say what they decided, in words.

    Input: returns array-like (already cut at the as-of date), cfg. Output: dict with
      mean     = dict(m, text, reason)         m = AR order chosen by mean_order_rule
      variance = dict(order, stop, text, reason)  order = START order (p, q) from variance_order_rule
    `reason` is the rule's own reason string, unchanged; `text` is one plain sentence built from the
    rule's structured output. No model is fitted (the fit and the gate are the Select tab).
    The variance rule reads the mean-model residuals, so the same residual step as
    selector.select_and_fit is used (selector._mean_resid).
    Note: under mean.trigger = lb_only the mean reason still says "no ACF/PACF spike" for m = 0,
    although spikes are not checked there; the text states this so the wording is not misread.
    """
    x = np.asarray(returns, dtype=float)

    # --- Mean rule, then the residuals the variance rule inspects.
    mean = mean_order_rule(x, cfg)
    var = variance_order_rule(_mean_resid(x, mean["m"]), cfg)

    if mean["m"] == 0:
        mean_text = "The mean is treated as constant (no autocorrelation evidence)."
        if cfg["mean"]["trigger"] == "lb_only":
            mean_text += (" Under the frozen mean rule only the Ljung-Box test decides; "
                          "single ACF/PACF spikes are not checked, whatever the reason string below says.")
    else:
        mean_text = f"The mean gets an AR({mean['m']}) term: the returns show autocorrelation."

    p, q = var["start_order"]
    if var["stop_constant_variance"]:
        var_text = "No ARCH effect: the variance is treated as constant, so no GARCH model is started."
    elif q == 0:
        var_text = f"Start from ARCH({p}): the squares' PACF cuts off at lag {p} and the ACF fades."
    else:
        var_text = (f"Start from the GARCH({p},{q}) default: the squares' ACF/PACF pattern is not a "
                    f"clean ARCH cut-off.")

    return dict(mean=dict(m=mean["m"], text=mean_text, reason=mean["reason"]),
                variance=dict(order=(p, q), stop=var["stop_constant_variance"], text=var_text,
                              reason=var["reason"]))


# ---------------------------------------------------------------------------------------
# Select tab
# ---------------------------------------------------------------------------------------

def _fmt(v) -> str:
    """Text for one table cell: floats to 4 significant digits, everything else via str()."""
    return f"{v:.4g}" if isinstance(v, float) else str(v)


def selection_tables(result: dict, cfg: dict) -> dict:
    """Split a select_and_fit result into display-ready tables.

    Input: result from selector.select_and_fit, cfg (gate.alpha, gate.lm_lags, app labels). Output:
    dict of DataFrames: summary (item, value), params (parameter, value), gate (test, p_value, passes)
    and log (the escalation_log, one row per fit). Orders become text like "(1, 1)" and mixed columns
    become strings, so Streamlit can show them without type errors. Parameter names, test names and
    coded cell values (step, distribution, reason, gate verdict) are replaced by the plain labels in
    app.param_labels / app.value_labels. The start order is the first escalation_log entry.
    """
    log = result["escalation_log"]
    gate = result["gate"]

    summary = [("mean spec", result["mean_spec"]["kind"]),
               ("mean-residual LB p", result["mean_resid_lb_p"]),
               ("start order (p, q)", str(log[0]["order"])),
               ("final order (p, q)", str(result["order"])),
               ("error distribution", cfg["app"]["value_labels"]["dist"].get(result["dist"], result["dist"])),
               ("persistence (alpha + beta)", result["persistence"]),
               ("BIC", result["bic"]),
               ("gate passed", gate["passed"]),
               ("models fitted", result["models_fitted"]),
               ("flagged", result["flagged"]),
               ("no valid fit", result["no_valid_fit"]),
               ("reason", result["reason"])]

    # Gate p-values: each must exceed gate.alpha for the fit to pass.
    alpha = cfg["gate"]["alpha"]
    gate_rows = [dict(test=f"Squared residuals: Ljung-Box, lag {lag}", p_value=p, passes=bool(p > alpha))
                 for lag, p in gate["lb_p"].items()]
    if not math.isnan(gate["lm_p"]):
        gate_rows.append(dict(test=f"ARCH-LM test, {cfg['gate']['lm_lags']} lags",
                              p_value=gate["lm_p"], passes=bool(gate["lm_p"] > alpha)))

    # gate_passed is None when a fit was rejected as invalid and so never gate-tested.
    shown = {True: "pass", False: "fail", None: "not tested"}
    log_rows = [dict(step=e["step"], order=str(e["order"]), dist=e["dist"], bic=e["bic"], valid=e["valid"],
                     why=e["why"], persistence=e["persistence"], gate=shown[e["gate_passed"]])
                for e in log]

    return dict(summary=pd.DataFrame(summary, columns=["item", "value"]).assign(
                    value=lambda d: d["value"].map(_fmt)),
                params=pd.DataFrame([(plain_param(k, cfg), v) for k, v in result["params"].items()],
                                    columns=["parameter", "value"]),
                gate=pd.DataFrame(gate_rows, columns=["test", "p_value", "passes"]),
                log=label_values(pd.DataFrame(log_rows), cfg))


def grid_comparison(result: dict, grid: dict, cfg: dict) -> pd.DataFrame:
    """Side-by-side row for the hybrid selection and the full-grid BIC choice.

    Input: result from select_and_fit, grid from baselines.grid_bic (fit_forecast dict), cfg (labels).
    Output: DataFrame with one row per selector: order, dist, models_fitted, flagged, no_valid_fit,
    with selector and distribution shown by their plain names.
    Idea: the same frozen mean rule feeds both; they differ only in how the variance order is found
    (rules + gate versus fitting every (p, q) and taking the lowest BIC), and in cost (fits).
    When no_valid_fit is True the order shown was not a fitted model.
    """
    return label_values(pd.DataFrame([
        dict(selector="hybrid", order=str(result["order"]), dist=result["dist"],
             models_fitted=result["models_fitted"], flagged=result["flagged"],
             no_valid_fit=result["no_valid_fit"]),
        dict(selector="grid_bic", order=str(grid["order"]), dist=grid["spec"]["dist"] if grid["spec"] else "",
             models_fitted=grid["models_fitted"], flagged=grid["flagged"],
             no_valid_fit=grid["no_valid_fit"])]), cfg)


# ---------------------------------------------------------------------------------------
# Forecast tab (saved CSVs only)
# ---------------------------------------------------------------------------------------

def read_result(cfg: dict, name: str) -> pd.DataFrame | None:
    """Read results/<name> (path from paths.results_dir); None if the file does not exist.

    Input: cfg, file name such as "loss_table.csv". Output: DataFrame, or None so the caller can show
    the command that produces the file instead of crashing. A `date` column is parsed to datetime.
    """
    path = ROOT / cfg["paths"]["results_dir"] / name
    if not path.exists():
        return None

    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    return df


def result_mtime(cfg: dict, name: str) -> float | None:
    """Last-modified time of results/<name>, or None if it does not exist.

    Input: cfg, file name. Output: float seconds or None. Used as part of a cache key in app.py, so a
    cached read is refreshed when the file is created or rewritten (a cached "missing" would
    otherwise stick until the server restarts).
    """
    path = ROOT / cfg["paths"]["results_dir"] / name

    return path.stat().st_mtime if path.exists() else None


def pair_rows(df: pd.DataFrame, pair: str) -> pd.DataFrame:
    """Rows of a results table that belong to one pair (column `pair`), index reset."""
    return df[df["pair"] == pair].reset_index(drop=True)


# ---------------------------------------------------------------------------------------
# Simulate tab
# ---------------------------------------------------------------------------------------

def true_order_label(cfg: dict, true_model: str) -> str:
    """The true variance order of a recovery.models entry as text, e.g. "(2, 1)"."""
    spec = true_spec(cfg["recovery"]["models"][true_model])

    return f"({spec['p']}, {spec['q']})"


def live_recovery(true_model: str, n: int, R: int, cfg: dict) -> tuple[dict, pd.DataFrame]:
    """Run the hybrid selector on R simulated series from a known model (small live version).

    Input: true_model name in recovery.models, n series length, R replications (<= app.simulate.max_live_R),
    cfg. Output: (metrics, counts): metrics = simulate.recovery_metrics dict (rates and their
    Monte Carlo SEs), counts = simulate.order_counts table (start_order, final_order, count).
    Seeds are eval.seed_base + i, as in the saved table, and the rules are the frozen ones.
    Raises ValueError if R is outside 1..max_live_R, before anything is computed.
    """
    cap = cfg["app"]["simulate"]["max_live_R"]
    if not 1 <= R <= cap:
        raise ValueError(f"R = {R} outside 1..{cap} (live Monte Carlo is capped, app.simulate.max_live_R)")

    df = order_recovery_experiment(true_model, n, R, cfg["eval"]["seed_base"], cfg,
                                   selector="hybrid", model_set="recovery")

    return recovery_metrics(df, true_model, cfg, "hybrid"), order_counts(df)


def saved_recovery_row(table: pd.DataFrame, true_model: str, n: int, selector: str) -> dict | None:
    """The saved recovery_table.csv row for (true_model, n, selector) as a dict; None if absent."""
    hit = table[(table["true_model"] == true_model) & (table["n"] == n) & (table["selector"] == selector)]

    return None if hit.empty else hit.iloc[0].to_dict()


def rates_table(metrics: dict, cfg: dict) -> pd.DataFrame:
    """Recovery rates with their Monte Carlo SE as a table, in percent, with plain names.

    Input: dict with the rate and *_se keys of simulate.recovery_metrics (a live result, or a saved
    recovery_table.csv row as a dict), cfg (app.rate_labels). Output: DataFrame with columns rate (plain
    name), share and mc_se (both in percent, 0-100), in RATES order. NaN means "not defined"
    (grid_bic has no gate or start order).
    """
    labels = cfg["app"]["rate_labels"]

    return pd.DataFrame([dict(rate=labels[r], share=100 * metrics[r], mc_se=100 * metrics[se]) for r, se in RATES])


def rate_specs(cfg: dict, keys: list[str]) -> list[tuple[str, str, str]]:
    """(rate key, SE key, plain label) for the chosen rates, as plots.recovery_bars expects.

    Input: cfg (app.rate_labels), keys such as app.simulate.chart_rates. Output: list of tuples in that
    order. Raises KeyError for a key that is not one of the recovery rates.
    """
    se = dict(RATES)

    return [(k, se[k], cfg["app"]["rate_labels"][k]) for k in keys]


def order_shares(counts: pd.DataFrame) -> pd.Series:
    """Percent of runs that ended on each final order.

    Input: DataFrame with columns final_order and count (order_counts output, or recovery_orders.csv rows
    for one cell); extra columns such as start_order are summed away. Output: Series indexed by the order
    text (like "(1, 1)"), values in percent that add up to 100.
    """
    by_final = counts.groupby("final_order")["count"].sum()

    return 100 * by_final / by_final.sum()


def saved_order_counts(orders: pd.DataFrame, true_model: str, n: int, selector: str) -> pd.DataFrame:
    """The rows of recovery_orders.csv for one (true model, length, selector) cell."""
    return orders[(orders["true_model"] == true_model) & (orders["n"] == n) & (orders["selector"] == selector)]


# ---------------------------------------------------------------------------------------
# Plain labels for tables
# ---------------------------------------------------------------------------------------

def label_values(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Replace coded cell values by their plain names (app.value_labels), column by column.

    Input: DataFrame, cfg. Output: a copy; a value with no entry stays as it is. Example: in a column
    named selector, "fixed_garch11" becomes "Fixed GARCH(1,1)". Only columns listed in value_labels change.
    """
    out = df.copy()

    for col, mapping in cfg["app"]["value_labels"].items():
        if col in out.columns:
            out[col] = out[col].map(lambda v, m=mapping: m.get(v, v))

    return out


def scale_percent(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Multiply the listed columns by 100 (fractions to percent); columns not in df are skipped. Returns a copy."""
    out = df.copy()

    for col in cols:
        if col in out.columns:
            out[col] = out[col] * 100

    return out


def plain_param(name: str, cfg: dict) -> str:
    """Plain name of a fitted parameter from arch, e.g. "alpha[2]" -> "Reaction to a past shock (alpha), lag 2".

    Input: arch parameter name, cfg (app.param_labels). Output: text; a name with no entry is returned as is.
    """
    labels = cfg["app"]["param_labels"]
    match = re.fullmatch(r"(\w+)\[(\d+)\]", name)

    if match and match.group(1) in labels:
        return f"{labels[match.group(1)]}, lag {match.group(2)}"

    return labels.get(name, name)


def results_table(df: pd.DataFrame, pair: str, cfg: dict) -> pd.DataFrame:
    """One pair's rows of loss_table.csv or dm_table.csv, ready to show.

    Input: the full table, pair (FRED id), cfg. Output: rows of that pair without the pair column,
    fraction columns (app.percent_columns) turned into percent, coded values (selector, loss, sample,
    outcome) replaced by plain names.
    """
    rows = pair_rows(df, pair).drop(columns=["pair"])

    return label_values(scale_percent(rows, cfg["app"]["percent_columns"]), cfg)


# ---------------------------------------------------------------------------------------
# Select tab: step table
# ---------------------------------------------------------------------------------------

PASS, FAIL, SKIP = "✅", "❌", "➖"   # icons of the step table: passed, failed, not needed


def step_table(result: dict, cfg: dict) -> pd.DataFrame:
    """The selector's pipeline as a table of steps with a pass/fail icon and a one-line explanation.

    Input: result from select_and_fit, cfg. Output: DataFrame with columns step, result (icon + short
    outcome) and detail. Steps: mean rule, Check 1 (is there volatility clustering?), start order, first
    fit, validity of the final fit, diagnostic gate, escalation. Everything is read from the result
    dict (nothing is recomputed). The mean and variance rule reasons are the first and second parts of
    result["reason"], which select_and_fit writes as "mean | variance | step".
    """
    log, gate = result["escalation_log"], result["gate"]
    parts = result["reason"].split(" | ")
    mean_reason, var_reason = (parts[0], parts[1]) if len(parts) == 3 else ("", result["reason"])
    constant = log[0]["step"] == "constant_variance"
    alpha = cfg["gate"]["alpha"]
    first = log[0]
    why_names, step_names = cfg["app"]["value_labels"]["why"], cfg["app"]["value_labels"]["step"]   # plain names

    rows = [dict(step="1. Mean model (ACF/PACF rule)", result=f"{PASS} {result['mean_spec']['kind']}",
                 detail=mean_reason)]

    # Check 1: ARCH-LM and Ljung-Box on squared residuals; no effect means no GARCH model at all.
    rows.append(dict(step="2. Check 1: is there volatility clustering?",
                     result=f"{SKIP} No, constant variance" if constant else f"{PASS} Yes",
                     detail=var_reason if constant else "Tests on the squared residuals reject constant variance."))

    rows.append(dict(step="3. Starting order from the ACF/PACF pattern",
                     result=f"{SKIP} Not needed" if constant else f"{PASS} {first['order']}",
                     detail="" if constant else var_reason))

    rows.append(dict(step="4. First model fitted",
                     result=f"{PASS} Fitted" if first["valid"] else f"{FAIL} Rejected",
                     detail=f"Order {first['order']}, persistence {first['persistence']:.4g}."
                            + ("" if first["valid"] else f" Rejected because: {why_names.get(first['why'], first['why'])}.")))

    # Validity of the model finally used (omega > 0, no negative coefficients, persistence < 1).
    rejected = [e for e in log if not e["valid"]]
    rows.append(dict(step="5. Valid model (positive variance, persistence below 1)",
                     result=f"{FAIL} No model was valid" if result["no_valid_fit"] else f"{PASS} Yes",
                     detail=f"{len(rejected)} of {len(log)} fits rejected." if rejected else "No fit was rejected."))

    # Gate: squared standardised residuals must look like white noise.
    if result["no_valid_fit"]:
        gate_row = dict(result=f"{SKIP} Not tested", detail="There was no valid model to test.")
    else:
        worst = min(list(gate["lb_p"].values()) + [gate["lm_p"]])
        gate_row = dict(result=f"{PASS} Passed" if gate["passed"] else f"{FAIL} Failed",
                        detail=f"Smallest p-value {worst:.4g}; each must exceed {alpha}.")
    rows.append(dict(step="6. Diagnostic gate", **gate_row))

    # Escalation: extra fits tried when the first model did not pass.
    extra = [step_names.get(e["step"], e["step"]) for e in log[1:]]
    if not extra:
        esc = dict(result=f"{SKIP} Not needed", detail="The first model was accepted.")
    elif result["flagged"] or result["no_valid_fit"]:
        esc = dict(result=f"{FAIL} Used, gate still failed", detail=f"Tried: {', '.join(extra)}.")
    else:
        esc = dict(result=f"{PASS} Used, gate passed", detail=f"Tried: {', '.join(extra)}.")
    rows.append(dict(step="7. Escalation", **esc))

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------
# Explanation text
# ---------------------------------------------------------------------------------------

class _SafeDict(dict):
    """dict for str.format_map that answers "n/a" for a placeholder with no value."""

    def __missing__(self, key):
        return "n/a"


def render_text(template: str, values: dict) -> str:
    """Fill the {placeholders} of a config text from `values`; a missing one prints "n/a" (never raises)."""
    return template.format_map(_SafeDict(values))


def _pct(x: float) -> str:
    """A fraction as text with one decimal, e.g. 0.658 -> "65.8%"."""
    return f"{100 * x:.1f}%"


def explain_values(cfg: dict, pair: str, returns: pd.Series, tables: dict,
                   live_metrics: dict | None = None, live_R: int | None = None,
                   saved_row: dict | None = None) -> dict:
    """Numbers for the placeholders in app.explain, taken from data, results CSVs and config.

    Input: cfg, pair (FRED id), returns (already cut at the as-of date), tables = dict with optional
    DataFrames under the keys diagnostic, recovery, loss, dm (None = file missing), live_metrics and
    live_R (the live Monte Carlo, or None), saved_row (the saved recovery_table row, or None).
    Output: dict placeholder -> text. A number whose source is missing is simply left out, so
    render_text prints "n/a" for it; nothing is typed in by hand.
    """
    app, conf = cfg["app"], cfg["band_confidence"]
    stats = summary_table(returns).iloc[0]

    vals = dict(pair_label=app["pair_labels"].get(pair, pair),
                first_date=str(returns.index[0].date()), last_date=str(returns.index[-1].date()),
                n_obs=f"{len(returns):,}", excess_kurtosis=f"{stats['excess_kurtosis']:.2f}",
                zero_days=str(int(stats["zero_returns"])),
                band=f"{norm.ppf(0.5 + conf / 2) / np.sqrt(len(returns)):.3f}", band_pct=f"{100 * conf:.0f}",
                gate_alpha=str(cfg["gate"]["alpha"]), alpha_level=str(cfg["gate"]["alpha"]),
                frozen_id=cfg["frozen"]["decision_id"])

    # spurious PACF spikes for the larger and the smaller ARCH(1) coefficient.
    diag = tables.get("diagnostic")
    if diag is not None and len(diag):
        a_hi, a_lo, n_lo, n_hi = diag["alpha"].max(), diag["alpha"].min(), diag["n"].min(), diag["n"].max()

        def spike(a, n):
            return diag[(diag["alpha"] == a) & (diag["n"] == n)]["false_spike_rate"].iloc[0]

        vals.update(alpha_high=str(a_hi), alpha_low=str(a_lo), n_small=f"{n_lo:,}", n_large=f"{n_hi:,}",
                    spike_rate_small_n=_pct(spike(a_hi, n_lo)), spike_rate_large_n=_pct(spike(a_hi, n_hi)),
                    spike_rate_low_alpha=_pct(spike(a_lo, n_hi)))

    # Cost of the hybrid versus the full grid in the saved recovery run (at the length where both ran).
    rec = tables.get("recovery")
    if rec is not None and len(rec):
        at = rec[rec["n"] == cfg["recovery"]["grid_n"][0]]
        for sel, key in [("hybrid", "hybrid_fits"), ("grid_bic", "grid_fits")]:
            sub = at[at["selector"] == sel]
            if len(sub):
                vals[key] = f"{sub['mean_fits'].mean():.1f}"

    # Forecast tab: fallback shares of this pair and the number of test days.
    loss, dm = tables.get("loss"), tables.get("dm")
    if loss is not None:
        mine = loss[loss["pair"] == pair]
        for sel, key in [("hybrid", "fallback_hybrid"), ("fixed_garch11", "fallback_fixed")]:
            sub = mine[mine["selector"] == sel]
            if len(sub):
                vals[key] = _pct(sub["fallback_share"].iloc[0])
    if dm is not None:
        sub = dm[(dm["pair"] == pair) & (dm["sample"] == "all")]
        if len(sub):
            vals["n_days"] = f"{int(sub['n_days'].iloc[0]):,}"

    # Simulate tab: the saved row and the live run.
    if saved_row is not None:
        vals.update(saved_R=f"{int(saved_row['R']):,}", saved_exact=_pct(saved_row["exact"]),
                    saved_nesting=_pct(saved_row["nesting"]))
    if live_R is not None:
        vals["live_R"] = str(live_R)
    if live_metrics is not None:
        vals["live_se_exact"] = f"+/- {100 * live_metrics['exact_se']:.1f} percentage points"

    return vals
