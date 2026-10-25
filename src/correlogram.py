"""Correlograms: ACF/PACF values, confidence bands, significant lags."""
import numpy as np
from scipy.stats import norm
from statsmodels.tsa.stattools import acf, pacf


def acf_pacf_band(x, nlags: int, confidence: float = 0.95) -> dict:
    """ACF, PACF (Yule-Walker MLE) for lags 1..nlags and the white-noise band."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    return dict(
        acf=acf(x, nlags=nlags, fft=True)[1:],
        pacf=pacf(x, nlags=nlags, method="ywm")[1:],
        band=norm.ppf(0.5 + confidence / 2) / np.sqrt(n),
        n=n,
    )


def significant_lags(values, band: float) -> list[int]:
    """1-indexed lags where |value| > band."""
    return [i + 1 for i, v in enumerate(values) if abs(v) > band]
