"""The five variance selectors compared in the study, all behind one interface.

Every selector is a function fit_forecast(train, cfg) -> dict with the keys
    h_next        one-step-ahead conditional variance for the day AFTER the last train value
    order         (p, q) of the variance model (None for the rolling variance)
    models_fitted number of GARCH fits this call needed (cost of the selection)
    flagged       True when the fit is not trustworthy (no gate for the baselines: only
                  True when no valid fit exists)
    no_valid_fit  True when every fit was rejected; h_next is then the rolling variance
    spec          dict(m, order, dist, params) of the chosen model, or None. forecast.py
                  reuses it to update the variance between refits without refitting.
    reject_reason distinct reasons among the fits REJECTED in this call, ";"-separated ("" if none)
    rejected_persistence  largest alpha+beta among fits rejected for persistence (NaN if none)
The last two are observation only: they never change which fit is chosen.

Fairness rule: every GARCH-based selector uses the SAME frozen mean rule
(rules.mean_order_rule), so the only thing that differs between them is the VARIANCE-order
choice, which is what the study is about.

No look-ahead: each function only sees the `train` array it is given.
"""
import numpy as np
from arch import arch_model

from src.rules import mean_order_rule, variance_order_rule
from src.selector import _fit, _mean_resid, _persistence, _valid, select_and_fit


def rolling_variance(x, window: int) -> float:
    """Variance of the last `window` values of x (np.var convention, ddof = 0).

    Input: x array-like of returns, window int. Output: float.
    Idea: a model-free volatility estimate; if fewer than `window` values exist, all are used.
    """
    x = np.asarray(x, dtype=float)

    # Slice the most recent window; ddof=0 divides by the window length (np.var default).
    return float(np.var(x[-window:]))


def next_variance(x, spec: dict) -> float:
    """One-step-ahead variance of a GARCH-type model with FIXED parameters, filtered through x.

    Input: x returns up to day t (array-like); spec = dict(m, order, dist, params) where
    params are the fitted values in arch's own order. Output: variance for day t + 1.
    Idea: the parameters stay as estimated at the last refit, but the conditional variance
    recursion h_t = omega + alpha * e_{t-1}^2 + beta * h_{t-1} is run over the NEW data, so
    yesterday's shock and variance are up to date without re-estimating anything.
    arch calls used (read from installed source, not executed):
    ARCHModel.fix(params) at arch/univariate/base.py:526, ARCHModelFixedResult.forecast
    at base.py:1444, ARCHModelForecast.variance at base.py:2224.
    """
    x = np.asarray(x, dtype=float)
    m, (p, q), dist = spec["m"], spec["order"], spec["dist"]

    # (0, 0) is the constant-variance model the selector uses when Check 1 sees no ARCH effect.
    vol = "constant" if (p, q) == (0, 0) else "GARCH"

    # Build the SAME model structure as _fit does (rescale=False keeps the x100 scale), so the
    # parameter vector lines up with the model's parameter order.
    mean = "Constant" if m == 0 else "AR"
    model = arch_model(x, mean=mean, lags=m if m else 0, vol=vol, p=p, q=q,
                       dist=dist, rescale=False)

    # dict preserves arch's parameter order (mean, volatility, distribution).
    res = model.fix(np.array(list(spec["params"].values()), dtype=float))

    # horizon=1 with the default start gives one row: the forecast made from the last value of x.
    return float(res.forecast(horizon=1).variance.to_numpy()[-1, 0])


def _reject_summary(rejects) -> tuple[str, float]:
    """Summarise rejected fits for the output (observation only).

    Input: rejects = list of (why, persistence) pairs, one per REJECTED fit (why is the string
    from selector._valid). Output: (reason, rejected_persistence): the distinct reasons in
    first-seen order joined by ";" ("" if none), and the largest persistence among fits
    rejected for persistence (NaN if there was none).
    """
    # dict.fromkeys drops duplicates but keeps the first-seen order.
    reason = ";".join(dict.fromkeys(why for why, _ in rejects))

    # Persistence is only meaningful as "how close to 1" for the persistence rejections.
    pers = [p for why, p in rejects if why.startswith("persistence")]

    return reason, (max(pers) if pers else float("nan"))


def _pack(x, cfg: dict, m: int, order, dist: str, params: dict, models_fitted: int,
          rejects=()) -> dict:
    """Build the common output dict from a chosen model (empty params = no valid fit).

    Input: x train returns, cfg, m mean order, order (p, q), dist, params dict, models_fitted,
    rejects = (why, persistence) pairs of the rejected fits (for the reject columns only).
    Output: the fit_forecast dict described in the module docstring.
    """
    reason, rej_pers = _reject_summary(rejects)

    # No valid fit: fall back to the rolling variance and flag it.
    if not params:
        return dict(h_next=rolling_variance(x, cfg["baselines"]["rolling_window"]), order=order,
                    models_fitted=models_fitted, flagged=True, no_valid_fit=True, spec=None,
                    reject_reason=reason, rejected_persistence=rej_pers)

    # Valid fit: forecast with the fitted parameters filtered through the training data.
    spec = dict(m=m, order=order, dist=dist, params=params)

    return dict(h_next=next_variance(x, spec), order=order, models_fitted=models_fitted,
                flagged=False, no_valid_fit=False, spec=spec,
                reject_reason=reason, rejected_persistence=rej_pers)


def rolling_var(train, cfg: dict) -> dict:
    """Baseline 1: sample variance of the last baselines.rolling_window (60) returns.

    Input: train returns, cfg. Output: fit_forecast dict (no model, so order None, 0 fits).
    """
    x = np.asarray(train, dtype=float)

    # No model is fitted, so nothing can be rejected: empty reason, NaN persistence.
    return dict(h_next=rolling_variance(x, cfg["baselines"]["rolling_window"]), order=None,
                models_fitted=0, flagged=False, no_valid_fit=False, spec=None,
                reject_reason="", rejected_persistence=float("nan"))


def fixed_garch11(train, cfg: dict) -> dict:
    """Baseline 2: frozen mean rule + GARCH(1,1) with fit.dist errors, no gate, no search.

    Input: train returns, cfg. Output: fit_forecast dict (1 fit). The standard textbook model.
    """
    x = np.asarray(train, dtype=float)
    dist = cfg["fit"]["dist"]

    # Same mean model as every other selector (fairness rule).
    m = mean_order_rule(x, cfg)["m"]

    # One GARCH(1,1) fit; p = lag of squared shocks, q = lag of variances (arch convention).
    res = _fit(x, m, 1, 1, dist)

    # Same validity rules as the selector: omega > 0, alpha/beta >= 0, persistence < 1 - tol.
    ok, why = _valid(res.params, cfg["fit"]["persistence_tol"])

    # Keep the rejection reason (if any) for the output columns; the decision above is unchanged.
    rejects = [] if ok else [(why, _persistence(res.params))]

    return _pack(x, cfg, m, (1, 1), dist, res.params.to_dict() if ok else {}, 1, rejects)


def acf_only(train, cfg: dict) -> dict:
    """Baseline 3: frozen mean rule + the ACF/PACF variance rule's start order, fitted ONCE.

    No diagnostic gate and no escalation (so no escalation_log in the output): this isolates
    what the correlogram reading alone buys. Input: train returns, cfg. Output: dict (1 fit).
    """
    x = np.asarray(train, dtype=float)
    dist = cfg["fit"]["dist"]

    # Mean model, then the squared residuals the variance rule reads its correlograms from.
    m = mean_order_rule(x, cfg)["m"]
    var = variance_order_rule(_mean_resid(x, m), cfg)

    # Check 1 says "no ARCH effect" -> constant variance; otherwise the rule's (p, q).
    if var["stop_constant_variance"]:
        p, q, vol = 0, 0, "constant"
    else:
        (p, q), vol = var["start_order"], "GARCH"

    # Fit exactly once and accept it if valid; nothing is re-tried.
    res = _fit(x, m, p, q, dist, vol)
    ok, why = _valid(res.params, cfg["fit"]["persistence_tol"])

    # Keep the rejection reason (if any) for the output columns; the decision above is unchanged.
    rejects = [] if ok else [(why, _persistence(res.params))]

    return _pack(x, cfg, m, (p, q), dist, res.params.to_dict() if ok else {}, 1, rejects)


def grid_bic(train, cfg: dict) -> dict:
    """Baseline 4: frozen mean rule + every (p, q), p in 1..grid_max, q in 0..grid_max, lowest BIC.

    BIC = k ln(n) - 2 ln(L) trades fit against parameter count; lower is better. Invalid fits
    (same rules as the selector) never win. models_fitted counts ALL fits, valid or not.
    Input: train returns, cfg. Output: fit_forecast dict.
    """
    x = np.asarray(train, dtype=float)
    dist, tol = cfg["fit"]["dist"], cfg["fit"]["persistence_tol"]
    gmax = cfg["baselines"]["grid_max"]

    m = mean_order_rule(x, cfg)["m"]

    best = None   # (bic, order, params) of the best valid fit so far
    fitted = 0    # number of fits attempted
    rejects = []  # (why, persistence) of every rejected fit, for the output columns only

    # q = 0 is ARCH(p); p starts at 1 because arch GARCH needs at least one shock lag.
    for p in range(1, gmax + 1):
        for q in range(0, gmax + 1):
            res = _fit(x, m, p, q, dist)
            fitted += 1

            ok, why = _valid(res.params, tol)
            if not ok:
                rejects.append((why, _persistence(res.params)))  # observation only
            if ok and (best is None or res.bic < best[0]):
                best = (float(res.bic), (p, q), res.params.to_dict())

    # No valid fit anywhere on the grid: order is reported as (1, 1) but nothing was fitted.
    if best is None:
        return _pack(x, cfg, m, (1, 1), dist, {}, fitted, rejects)

    return _pack(x, cfg, m, best[1], dist, best[2], fitted, rejects)


def hybrid(train, cfg: dict) -> dict:
    """Selector 5 (the study's method): selector.select_and_fit (rules + gate + escalation).

    Input: train returns, cfg. Output: fit_forecast dict; models_fitted is the selector's own
    count, flagged is its gate-failure flag.
    """
    x = np.asarray(train, dtype=float)

    # The selector does mean rule, order rule, fit, gate and escalation by itself.
    out = select_and_fit(x, cfg)

    # Same packing as the baselines, then restore the selector's own flag (gate failure).
    # Rejected fits come from the selector's own log (valid False), read only, never altered.
    rejects = [(e["why"], e["persistence"]) for e in out["escalation_log"] if not e["valid"]]
    packed = _pack(x, cfg, out["mean_spec"]["m"], out["order"], out["dist"],
                   {} if out["no_valid_fit"] else out["params"], out["models_fitted"], rejects)
    packed["flagged"] = bool(out["flagged"])

    return packed


# Name -> function, used by forecast.py and simulate.py (order = column order in tables).
SELECTORS = dict(rolling_var=rolling_var, fixed_garch11=fixed_garch11, acf_only=acf_only,
                 grid_bic=grid_bic, hybrid=hybrid)
