"""Selector: apply rules, fit, run diagnostic gate, escalate on failure.

Pipeline: mean rule -> residuals -> Check 1 -> start order -> fit -> validity
-> gate -> escalate. Every fit is counted so the study can compare cost with grid search.
"""
import numpy as np
from arch import arch_model

from src.rules import mean_order_rule, variance_order_rule
from src.stat_tests import arch_lm, ljung_box


def _mean_resid(x: np.ndarray, m: int) -> np.ndarray:
    """Residuals of a constant-mean (m=0) or AR(m) OLS fit; length n - m.

    These are the "shocks" whose squares we inspect for volatility clustering.
    """
    if m == 0:
        return x - x.mean()

    # Regress x_t on a constant and x_{t-1}..x_{t-m}.
    y = x[m:]
    lags = [x[m - k:len(x) - k] for k in range(1, m + 1)]
    X = np.column_stack([np.ones(len(y))] + lags)

    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ coef


def _fit(x: np.ndarray, m: int, p: int, q: int, dist: str, vol: str = "GARCH"):
    """Fit mean + GARCH(p, q) with arch_model.

    arch convention: p = lags of squared shocks, q = lags of variances.
    q = 0 gives ARCH(p). rescale=False keeps the x100 return scale untouched.
    """
    mean = "Constant" if m == 0 else "AR"
    model = arch_model(x, mean=mean, lags=m if m else 0, vol=vol, p=p, q=q,
                       dist=dist, rescale=False)
    return model.fit(disp="off", show_warning=False)


def _persistence(params) -> float:
    """Persistence = sum of all alpha[i] and beta[j]; < 1 means a stationary variance."""
    return float(sum(v for k, v in params.items() if k.startswith(("alpha[", "beta["))))


def _valid(params, tol: float) -> tuple[bool, str]:
    """A fit is usable only if omega > 0, all alpha/beta >= 0 and persistence < 1 - tol.

    Input: fitted params (mapping), tol = fit.persistence_tol from config.
    Output: (is_valid, reason). arch constrains alpha+beta <= 1, so an optimum on the
    IGARCH boundary shows up as persistence ~ 1; the tolerance catches values that
    sit within tol of 1 (an infinite-variance model is not usable for forecasting).
    """
    if "omega" in params and params["omega"] <= 0:
        return False, "omega<=0"

    if any(v < 0 for k, v in params.items() if k.startswith(("alpha[", "beta["))):
        return False, "negative alpha/beta"

    # Reject at the boundary, not just beyond it: >= 1 - tol (the value is logged by the caller).
    if _persistence(params) >= 1 - tol:
        return False, "persistence>=1-tol"

    return True, "ok"


def _gate(res, cfg: dict) -> dict:
    """Diagnostic gate: has the model absorbed the volatility clustering?

    If it has, the squared standardised residuals look like white noise:
    Ljung-Box at each gate lag and ARCH-LM must all give p > alpha.
    No df adjustment for estimated parameters.
    """
    g = cfg["gate"]

    # Standardised residuals = shock / fitted volatility. Drop NaNs (AR start-up).
    z = np.asarray(res.std_resid, dtype=float)
    z = z[np.isfinite(z)]

    lb = ljung_box(z ** 2, g["lb_lags"], model_df=0)
    lb_p = {int(lag): float(p) for lag, p in zip(lb["lag"], lb["lb_pvalue"])}
    lm_p = float(arch_lm(z, g["lm_lags"])["lm_p"])

    passed = all(p > g["alpha"] for p in lb_p.values()) and lm_p > g["alpha"]
    return dict(lb_p=lb_p, lm_p=lm_p, passed=bool(passed))


def select_and_fit(returns, cfg: dict) -> dict:
    """Run the full selection on `returns` and return a result dict.

    Uses only the series it is given (no look-ahead). Escalation when the gate fails:
      (a) try (p+1, q) and (p, q+1) within the cap, keep the lowest BIC;
      (b) refit that order with Student-t errors;
      (c) otherwise return the lowest-BIC valid fit and flag it.
    If EVERY fit is rejected as invalid, nothing is raised: the result has
    no_valid_fit=True (persistence/bic NaN, params empty) and the caller must fall back
    (forecast.py will use the rolling 60-day variance and count the substitutions).
    Each escalation_log entry also records the fit's persistence, so a rejected value
    (e.g. 1.0000) can be read back.
    """
    x = np.asarray(returns, dtype=float)

    # --- Step 1: mean model, then the residuals the variance rule will inspect.
    mean = mean_order_rule(x, cfg)
    m = mean["m"]
    mean_resid = _mean_resid(x, m)
    var = variance_order_rule(mean_resid, cfg)

    # Did the mean model remove the autocorrelation? LB(10) p on its residuals (report only).
    lb_lag = cfg["mean"]["lb_lags"]
    mean_resid_lb_p = float(ljung_box(mean_resid, [lb_lag])["lb_pvalue"].iloc[0])

    cap, dist0 = cfg["escalation"]["cap"], cfg["fit"]["dist"]
    tol = cfg["fit"]["persistence_tol"]  # boundary margin for the validity check
    log: list[dict] = []   # one entry per fit, in order (this is also the fit counter)
    fits: list[dict] = []  # valid candidates only

    def attempt(step: str, p: int, q: int, dist: str, vol: str = "GARCH"):
        """Fit one candidate, log it, and keep it if it passes the validity checks."""
        res = _fit(x, m, p, q, dist, vol)
        ok, why = _valid(res.params, tol)
        gate = _gate(res, cfg) if ok else None

        # persistence is logged for every fit, so a fit rejected for sitting on the
        # boundary still shows how close to 1 it was.
        log.append(dict(step=step, order=(p, q), dist=dist, bic=float(res.bic), valid=ok,
                        why=why, persistence=_persistence(res.params),
                        gate_passed=None if gate is None else gate["passed"]))

        if not ok:
            return None
        fits.append(dict(res=res, p=p, q=q, dist=dist, gate=gate))
        return fits[-1]

    def pack(c, flagged: bool, reason: str) -> dict:
        """Build the output dict from the chosen candidate c (None = no valid fit)."""
        res = c["res"] if c else None
        return dict(
            mean_spec=dict(m=m, kind="constant" if m == 0 else f"AR({m})", reason=mean["reason"]),
            mean_resid_lb_p=mean_resid_lb_p,
            order=(c["p"], c["q"]) if c else var["start_order"],
            dist=c["dist"] if c else dist0,
            params=res.params.to_dict() if c else {},
            persistence=_persistence(res.params) if c else float("nan"),
            bic=float(res.bic) if c else float("nan"),
            gate=c["gate"] if c else dict(lb_p={}, lm_p=float("nan"), passed=False),
            escalation_log=log,
            models_fitted=len(log),
            flagged=flagged,
            no_valid_fit=c is None,  # True = every fit was rejected; caller must fall back
            reason=f"{mean['reason']} | {var['reason']} | {reason}",
        )

    # --- Step 2: Check 1 says no ARCH effect -> constant variance, nothing to escalate.
    if var["stop_constant_variance"]:
        c = attempt("constant_variance", 0, 0, dist0, vol="constant")
        return pack(c, flagged=not (c and c["gate"]["passed"]), reason="constant variance")

    # --- Step 3: fit the rule's starting order.
    p, q = var["start_order"]
    best = attempt("start", p, q, dist0)
    if best and best["gate"]["passed"]:
        return pack(best, False, "start order passed gate")

    # --- Step 4a: neighbours within the cap; the lowest-BIC valid fit is gate-tested.
    for pp, qq in [(p + 1, q), (p, q + 1)]:
        if pp <= cap and qq <= cap:
            attempt("neighbour", pp, qq, dist0)

    if fits:
        best = min(fits, key=lambda c: c["res"].bic)
        if best["gate"]["passed"]:
            return pack(best, False, "passed after neighbour step")

    # --- Step 4b: heavier tails. Student-t at the best order so far (or the start order).
    bp, bq = (best["p"], best["q"]) if fits else (p, q)
    t_fit = attempt("student_t", bp, bq, "t")
    if t_fit and t_fit["gate"]["passed"]:
        return pack(t_fit, False, "passed with Student-t")

    # --- Step 4c: give up gracefully. Lowest-BIC valid fit, flagged as failing the gate.
    best = min(fits, key=lambda c: c["res"].bic) if fits else None
    return pack(best, True, "gate failed after escalation" if best else "no valid fit")
