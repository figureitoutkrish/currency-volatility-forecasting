"""Data layer: load FRED FX series and compute log returns x100."""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path = "config.yaml") -> dict:
    """Read the YAML config. Relative paths resolve against the project root."""
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    with open(p) as f:
        return yaml.safe_load(f)


def load_fx(series_id: str, start: str, end: str | None = None,
            cfg: dict | None = None) -> pd.Series:
    """Load a FRED daily series, downloading and caching the raw CSV if absent.

    Columns are read positionally (date, value); FRED holidays ('.') are dropped.
    """
    cfg = cfg or load_config()
    raw_dir = ROOT / cfg["paths"]["raw_dir"]
    csv = raw_dir / f"{series_id}.csv"
    if not csv.exists():
        url = cfg["fred"]["url_template"].format(series_id=series_id)
        raw_dir.mkdir(parents=True, exist_ok=True)
        pd.read_csv(url).to_csv(csv, index=False)
    df = pd.read_csv(csv)
    idx = pd.DatetimeIndex(pd.to_datetime(df.iloc[:, 0]))
    s = pd.Series(pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy(),
                  index=idx, name=series_id)
    return s.dropna().sort_index().loc[start:end]


def to_log_returns(price: pd.Series) -> pd.Series:
    """Daily log returns in percent: 100 * diff(log price)."""
    if (price <= 0).any():
        raise ValueError("prices must be strictly positive")
    return (100 * np.log(price).diff()).dropna()
