"""Order-selection rules driven by ACF/PACF patterns. Thresholds come from config.yaml.

Reading guide:
- ACF at lag k: correlation of the series with itself k days earlier.
- PACF at lag k: the same, after removing the effect of lags 1..k-1.
  An AR(m) process has a PACF that "cuts off" after lag m, which is how we read m.
- A spike = a bar outside the 95% white-noise band (about +-1.96/sqrt(n)).
"""
import numpy as np

from src.correlogram import acf_pacf_band, significant_lags
from src.stat_tests import arch_lm, ljung_box


def mean_order_rule(x, cfg: dict) -> dict:
    """Choose the AR order m of the mean: 0 (constant) or the PACF cut-off, m <= max_m.

    The trigger that rejects "constant" is config-driven (mean.trigger):
      - "spike_or_lb" (M0, current): reject if any ACF/PACF spike at lags 1..spike_lags
        OR Ljung-Box(lb_lags) p <= alpha.
      - "lb_only" (M1): reject only if Ljung-Box(lb_lags) p <= alpha.
    When rejected, m = the last significant PACF lag within 1..max_m (falling back
    to 1 if no PACF spike sits in that range).
    Input: x returns (array-like), cfg dict. Output: dict(m, reason).
    """
    mc = cfg["mean"]

    # Correlogram of the returns themselves, only the first few lags matter for the mean.
    cor = acf_pacf_band(x, mc["spike_lags"], cfg["band_confidence"])
    acf_spikes = significant_lags(cor["acf"], cor["band"])
    pacf_spikes = significant_lags(cor["pacf"], cor["band"])

    # Joint test: any autocorrelation up to lag lb_lags? (small p = yes)
    lb_p = float(ljung_box(x, [mc["lb_lags"]])["lb_pvalue"].iloc[0])

    # Decide whether there is NO evidence of autocorrelation, under the configured trigger.
    # M0 also treats a single chance spike as evidence (a 5% band spikes by chance often);
    # M1 trusts only the joint Ljung-Box test, which pools lags and is less spike-prone.
    if mc["trigger"] == "spike_or_lb":
        no_evidence = not acf_spikes and not pacf_spikes and lb_p > mc["alpha"]
    elif mc["trigger"] == "lb_only":
        no_evidence = lb_p > mc["alpha"]
    else:
        raise ValueError(f"unknown mean.trigger {mc['trigger']!r}")

    # No evidence -> a constant mean is enough.
    if no_evidence:
        return dict(m=0, reason=f"constant: no ACF/PACF spike, LB({mc['lb_lags']}) p={lb_p:.3f}")

    # Otherwise read the AR order off the PACF cut-off, capped at max_m.
    in_range = [lag for lag in pacf_spikes if lag <= mc["max_m"]]
    m = max(in_range) if in_range else 1

    return dict(m=m, reason=f"AR({m}): ACF spikes {acf_spikes}, PACF spikes {pacf_spikes}, "
                            f"LB({mc['lb_lags']}) p={lb_p:.3g}")


def pacf_cutoff(pacf_spikes: list[int], max_p: int, rule: str) -> int:
    """Read the PACF cut-off lag p of the squared residuals (0 = no usable cut-off).

    pacf_spikes: 1-indexed lags where the PACF is outside the band. max_p: largest p allowed.
    rule (variance.cutoff_rule):
      - "last_significant" (C0, current): p = the last spike at lag <= max_p.
      - "contiguous" (C1): p = the largest k <= max_p with a spike at EVERY lag 1..k, so a
        lone chance spike at lag 3 cannot lift p over a gap at lag 2.
    Output: int p in 0..max_p; the caller treats 0 as "fall back to GARCH(1,1)".
    """
    spikes = set(pacf_spikes)

    if rule == "last_significant":
        # Any spike counts, gaps allowed; take the highest one that is still small enough.
        return max((lag for lag in spikes if lag <= max_p), default=0)

    if rule == "contiguous":
        # Walk up from lag 1 and stop at the first lag with no spike: the run must be unbroken.
        k = 0
        while k < max_p and (k + 1) in spikes:
            k += 1
        return k

    raise ValueError(f"unknown variance.cutoff_rule {rule!r}")


def variance_order_rule(resid, cfg: dict) -> dict:
    """Decide on constant variance, else a starting (p, q) from squared-residual ACF/PACF.

    Check 1: ARCH-LM and LB on squared residuals both p > alpha -> stop (constant variance).
    Otherwise ARCH(p), i.e. (p, 0), when ALL of these hold:
      - the PACF of squares has a cut-off p >= 1, p <= pacf_cut_max_p, read by
        pacf_cutoff (variance.cutoff_rule: last significant lag, or contiguous run),
      - no PACF spike in p+1..p+quiet_lags (a clean cut-off),
      - the ACF of squares "fades fast" (at most variance.fade_max_spikes spikes
        over lags fade_lag..corr_lags; 0 = the original strict rule).
    Anything else -> (1, 1), the default GARCH.
    Input: resid mean-model residuals, cfg dict.
    Output: dict(stop_constant_variance, start_order=(p, q), reason).
    """
    resid = np.asarray(resid, dtype=float)
    ac, vc = cfg["arch_check"], cfg["variance"]

    # --- Check 1: is there any volatility clustering at all?
    lm_p = arch_lm(resid, ac["lm_lags"])["lm_p"]
    lb_p = float(ljung_box(resid ** 2, [ac["lb_lags"]])["lb_pvalue"].iloc[0])

    if lm_p > ac["alpha"] and lb_p > ac["alpha"]:
        return dict(stop_constant_variance=True, start_order=(0, 0),
                    reason=f"Check 1: ARCH-LM p={lm_p:.3f}, LB sq p={lb_p:.3f}, no ARCH effect")

    # --- Correlogram of the SQUARED residuals (a proxy for variance).
    cor = acf_pacf_band(resid ** 2, vc["corr_lags"], cfg["band_confidence"])
    pacf_spikes = significant_lags(cor["pacf"], cor["band"])
    acf_spikes = significant_lags(cor["acf"], cor["band"])

    # --- PACF cut-off: the ARCH order p, read as configured (last significant lag, or the
    # unbroken run from lag 1), never above pacf_cut_max_p. p = 0 means no usable cut-off.
    p = pacf_cutoff(pacf_spikes, vc["pacf_cut_max_p"], vc["cutoff_rule"])

    # "Quiet" = no PACF spike right after the cut-off, so it really is a cut-off.
    quiet = not any(p < lag <= p + vc["quiet_lags"] for lag in pacf_spikes)

    # "Fades fast" = few ACF-of-squares spikes in the late lags fade_lag..corr_lags.
    # Squares are heavy-tailed, so the white-noise band is too narrow and a chance spike
    # is likely; fade_max_spikes lets us tolerate a small number of them (0 = none).
    late_spikes = [lag for lag in acf_spikes if vc["fade_lag"] <= lag <= vc["corr_lags"]]
    fades = len(late_spikes) <= vc["fade_max_spikes"]

    if p >= 1 and quiet and fades:
        return dict(stop_constant_variance=False, start_order=(p, 0),
                    reason=f"ARCH({p}): PACF cut-off at {p}, quiet after, "
                           f"ACF has {len(late_spikes)} late spike(s) from lag {vc['fade_lag']}, "
                           f"<= {vc['fade_max_spikes']} allowed")

    return dict(stop_constant_variance=False, start_order=(1, 1),
                reason=f"GARCH(1,1) default: PACF cut-off p={p}, quiet={quiet}, fades={fades}")
