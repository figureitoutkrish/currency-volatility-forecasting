"""Statistical tests: ADF/KPSS, Ljung-Box, ARCH-LM."""
import warnings

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.stattools import adfuller, kpss


def adf_kpss(x, alpha: float = 0.05) -> dict:
    """ADF (H0: unit root) and KPSS (H0: stationary) with a combined verdict.

    kpss_p is interpolated from a lookup table, so it is bounded by the table
    range; the InterpolationWarning is suppressed.
    """
    x = np.asarray(x, dtype=float)
    adf_stat, adf_p = adfuller(x, autolag="AIC", result_object=False)[:2]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kpss_stat, kpss_p = kpss(x, regression="c", nlags="auto")[:2]
    adf_rej, kpss_rej = adf_p < alpha, kpss_p < alpha
    if adf_rej and not kpss_rej:
        verdict = "stationary"
    elif not adf_rej and kpss_rej:
        verdict = "unit_root"
    else:
        verdict = "inconclusive"
    return dict(adf_stat=adf_stat, adf_p=adf_p, kpss_stat=kpss_stat,
                kpss_p=kpss_p, verdict=verdict)


def ljung_box(x, lags: list[int], model_df: int = 0) -> pd.DataFrame:
    """Ljung-Box test at the given lags: columns lag, lb_stat, lb_pvalue."""
    out = acorr_ljungbox(np.asarray(x, dtype=float), lags=lags, model_df=model_df)
    return out.rename_axis("lag").reset_index()[["lag", "lb_stat", "lb_pvalue"]]


def arch_lm(x, nlags: int) -> dict:
    """Engle ARCH-LM test on the demeaned series."""
    x = np.asarray(x, dtype=float)
    lm, lm_p, f, f_p = het_arch(x - x.mean(), nlags=nlags, result_object=False)
    return dict(lm_stat=lm, lm_p=lm_p, f_stat=f, f_p=f_p)
