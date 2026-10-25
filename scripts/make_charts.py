"""Draw the README charts into images/.

Usage (from the project root):  python scripts/make_charts.py
The data charts need only data/raw; the forecast charts also need results/ from
scripts/run_forecasts.py (they are skipped if it is missing).
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.correlogram import acf_pacf_band
from src.data import ROOT, load_config, load_fx, to_log_returns
from src.evaluate import qlike

OUT = ROOT / "images"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
PAIRS = {"DEXUSEU": ("EUR/USD", "#2a78d6"), "DEXJPUS": ("USD/JPY", "#eb6834"),
         "DEXINUS": ("USD/INR", "#1baf7a")}
SELECTORS = {"rolling_var": "Rolling 60-day variance", "fixed_garch11": "Fixed GARCH(1,1)",
             "acf_only": "ACF/PACF rules only", "grid_bic": "Full-grid BIC (30 models)",
             "hybrid": "Hybrid (this project)"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.size": 10, "text.color": INK,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "xtick.color": INK2, "ytick.color": INK2, "legend.frameon": False,
})


def save(fig, name):
    fig.savefig(OUT / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def headline(fig, title, subtitle):
    # Fixed offsets in inches above the plot area, so short and tall figures look alike.
    h = fig.get_figheight()
    fig.text(0.01, 1 + 0.45 / h, title, ha="left", va="bottom", fontsize=14, fontweight="bold", color=INK)
    fig.text(0.01, 1 + 0.15 / h, subtitle, ha="left", va="bottom", fontsize=10, color=INK2)


def load_all(cfg):
    prices = {p: load_fx(p, cfg["start"], cfg=cfg) for p in PAIRS}
    return prices, {p: to_log_returns(s) for p, s in prices.items()}


def chart_prices(prices, returns, cfg):
    """One panel per pair (different units, so no shared axis), test period shaded."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    for ax, (pair, s) in zip(axes, prices.items()):
        name, color = PAIRS[pair]
        ax.plot(s.index, s.values, color=color, lw=1.4)
        test_start = returns[pair].index[-int(round(len(returns[pair]) * cfg["test_fraction"]))]
        ax.axvspan(test_start, s.index[-1], color=GRID, alpha=0.6, lw=0)
        ax.set_title(name)
    axes[0].text(test_start, axes[0].get_ylim()[1], "  test period (last 20%)", va="top",
                 fontsize=9, color=INK2)
    headline(fig, "The three exchange rates, 2010 to 2026",
             "Daily noon buying rates in New York (Federal Reserve H.10, via FRED)")
    fig.tight_layout()
    save(fig, "exchange_rates.png")


def chart_returns(returns):
    """Daily returns: calm and wild periods bunch together (volatility clustering)."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    for ax, (pair, r) in zip(axes, returns.items()):
        name, color = PAIRS[pair]
        ax.vlines(r.index, 0, r.values, color=color, lw=0.5)
        ax.axhline(0, color=INK2, lw=0.6)
        ax.set_title(name)
        ax.set_ylabel("% per day")
    headline(fig, "Daily returns: big moves come in bunches",
             "Log return x 100. Quiet stretches follow quiet ones, and wild stretches follow wild ones")
    fig.tight_layout()
    save(fig, "daily_returns.png")


def chart_acf(returns, cfg):
    """The core idea: returns look uncorrelated, their squares do not."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    r = returns["DEXUSEU"].to_numpy()
    for ax, (series, title) in zip(axes, [(r, "Returns"), (r ** 2, "Squared returns (a stand-in for volatility)")]):
        cor = acf_pacf_band(series - (series.mean() if title == "Returns" else 0), 20, cfg["band_confidence"])
        lags = np.arange(1, 21)
        inside = np.abs(cor["acf"]) <= cor["band"]
        ax.bar(lags[inside], cor["acf"][inside], color="#b5b4ad", width=0.6, zorder=3)
        ax.bar(lags[~inside], cor["acf"][~inside], color=PAIRS["DEXUSEU"][1], width=0.6, zorder=3)
        ax.axhspan(-cor["band"], cor["band"], color=GRID, alpha=0.7, lw=0, zorder=1)
        ax.axhline(0, color=INK2, lw=0.6)
        ax.set_title(title)
        ax.set_xlabel("lag (days)")
        ax.set_xticks([1, 5, 10, 15, 20])
    axes[0].set_ylabel("autocorrelation (ACF)")
    axes[0].text(20.5, cor["band"], "95% noise band", ha="right", va="bottom", fontsize=8, color=INK2)
    headline(fig, "EUR/USD: today's size predicts tomorrow's size",
             "Blue bars sit outside the band that pure noise would stay inside")
    fig.tight_layout()
    save(fig, "acf_returns_vs_squares.png")


def chart_cost():
    """Model fits needed per pair over the 42 refits (from the report)."""
    fig, ax = plt.subplots(figsize=(8, 2.6))
    labels = ["Hybrid (this project)", "Full-grid BIC"]
    vals = [42, 1260]
    ax.barh(labels, vals, color=["#2a78d6", "#b5b4ad"], height=0.5)
    for i, v in enumerate(vals):
        ax.text(v + 15, i, f"{v:,} fits", va="center", color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 1450)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("GARCH models fitted on EUR/USD (42 refits)")
    headline(fig, "30x fewer model fits for the same forecast",
             "On EUR/USD and USD/JPY both methods picked GARCH(1,1) at every refit")
    fig.tight_layout()
    save(fig, "cost_comparison.png")


def chart_recovery():
    """Order recovery from the report: share of simulated series where the true order was found."""
    models = ["ARCH(1)", "GARCH(1,1)", "GARCH(2,1)", "GARCH(1,2)", "GARCH(1,1)\nStudent-t", "iid noise", "AR(1) +\nGARCH(1,1)"]
    rates = {500: [0.658, 0.766, 0.046, 0.000, 0.660, 0.926, 0.782],
             1000: [0.606, 0.938, 0.036, 0.006, 0.878, 0.912, 0.952],
             4000: [0.452, 0.996, 0.224, 0.038, 0.992, 0.930, 0.996]}
    shades = {500: "#a9c9f0", 1000: "#5b9be3", 4000: "#1d5aa6"}
    fig, ax = plt.subplots(figsize=(10, 4))
    x, w = np.arange(len(models)), 0.26
    for k, (n, vals) in enumerate(rates.items()):
        ax.bar(x + (k - 1) * w, vals, width=w - 0.03, color=shades[n], label=f"n = {n:,} days")
    ax.set_xticks(x, models)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("exact order found")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    headline(fig, "How often the rules find the true model",
             "500 simulated series per bar. Strong for GARCH(1,1), weak for higher orders")
    fig.tight_layout()
    save(fig, "order_recovery.png")


def chart_losses(results):
    """QLIKE per pair and selector, relative to the rolling-variance baseline (lower is better)."""
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharey=True)
    for ax, pair in zip(axes, PAIRS):
        fc = pd.read_csv(results / f"forecasts_{pair}.csv")
        mean = fc.groupby("selector").apply(lambda d: qlike(d["h"], d["r2"]).mean(), include_groups=False)
        diff = (mean - mean["rolling_var"]).reindex(list(SELECTORS))
        colors = ["#b5b4ad", "#b5b4ad", "#b5b4ad", "#b5b4ad", PAIRS[pair][1]]
        ax.barh([SELECTORS[s] for s in diff.index], diff.values, color=colors, height=0.6)
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set_title(PAIRS[pair][0])
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("QLIKE vs rolling variance")
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    axes[0].invert_yaxis()
    headline(fig, "Forecast error by method (left of zero = better than the simple baseline)",
             "Mean QLIKE over 838 out-of-sample days, rerun from the code in this repo")
    fig.tight_layout()
    save(fig, "forecast_losses.png")


def chart_forecast(results):
    """Forecast volatility against what happened, USD/JPY test period."""
    fc = pd.read_csv(results / "forecasts_DEXJPUS.csv", parse_dates=["date"])
    h = fc[fc["selector"] == "hybrid"].set_index("date")
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.vlines(h.index, 0, np.sqrt(h["r2"]), color="#c9c8c1", lw=0.7, label="actual move (|return|)")
    ax.plot(h.index, np.sqrt(h["h"]), color=PAIRS["DEXJPUS"][1], lw=2, label="forecast volatility (hybrid)")
    ax.set_ylabel("% per day")
    ax.legend(loc="upper left")
    headline(fig, "USD/JPY: the forecast rises and falls with the market",
             "One-day-ahead forecast, made each day using only data up to the day before")
    fig.tight_layout()
    save(fig, "forecast_vs_actual.png")


def main():
    cfg = load_config()
    OUT.mkdir(exist_ok=True)
    prices, returns = load_all(cfg)
    chart_prices(prices, returns, cfg)
    chart_returns(returns)
    chart_acf(returns, cfg)
    chart_cost()
    chart_recovery()

    results = ROOT / cfg["paths"]["results_dir"]
    if all((results / f"forecasts_{p}.csv").exists() for p in PAIRS):
        chart_losses(results)
        chart_forecast(results)
    else:
        print("results/ missing: run scripts/run_forecasts.py first for the forecast charts")


if __name__ == "__main__":
    main()
