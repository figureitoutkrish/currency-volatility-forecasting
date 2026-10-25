"""Run the rolling forecasts for every pair and selector, then score them.

Usage (from the project root):  python scripts/run_forecasts.py
Writes results/forecasts_<pair>.csv, results/loss_table.csv and results/dm_table.csv.
Takes a few minutes: the full-grid baseline fits 30 models at each of the 42 refits per pair.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import ROOT, load_config
from src.evaluate import dm_table, loss_table
from src.forecast import run_all


def main():
    cfg = load_config()
    out = ROOT / cfg["paths"]["results_dir"]

    forecasts = run_all(cfg)
    loss_table(forecasts, cfg).to_csv(out / "loss_table.csv", index=False)
    dm_table(forecasts, cfg).to_csv(out / "dm_table.csv", index=False)
    print(f"done: results in {out}")


if __name__ == "__main__":
    main()
