"""Audit R2K-Means against its paper, the FX data and forecast-pooling role."""

import argparse
from pathlib import Path

from fx_forecasting.config import PROJECT_ROOT
from fx_forecasting.evaluation.r2k_audit import run_methodology_audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "reports/clustering/r2kmeans/audit")
    parser.add_argument("--seed-count", type=int, default=10)
    args = parser.parse_args()
    result = run_methodology_audit(**vars(args))
    print(result["partition_comparison"].to_string(index=False))
    screen = result["forecast_summary"]
    print(screen.loc[screen.reporting_group.eq("all"),
                     ["model", "horizon", "pooled_r2_os_pct", "equal_currency_r2_os_pct", "fallback_rate"]].to_string(index=False))
    print(f"Saved audit evidence to {args.output_dir}")


if __name__ == "__main__":
    main()
