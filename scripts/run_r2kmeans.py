"""Run the paper-inspired currency experiment on local ingested BIS data."""

import argparse
from pathlib import Path

from fx_forecasting.config import INGESTED_DIR, PROJECT_ROOT
from fx_forecasting.evaluation.rolling_clustering import run_bis_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ingested-dir", type=Path, default=INGESTED_DIR)
    parser.add_argument("--sample-path", type=Path, default=PROJECT_ROOT / "config/research_sample.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "reports/clustering/r2kmeans")
    parser.add_argument("--start", default="1995-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--evaluation-start", default="2010-01")
    parser.add_argument("--n-clusters", type=int, default=3)
    parser.add_argument("--windows", nargs="+", type=int, default=[1, 6, 12])
    parser.add_argument("--lookback", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tune-k", action="store_true", help="Save a pre-evaluation K=2..6 diagnostic")
    args = vars(parser.parse_args())
    args["windows"] = tuple(args["windows"])
    _, summary, manifest = run_bis_experiment(**args)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"Saved experiment to {args['output_dir']}")
    print(f"Observed window: {manifest['settings']['start']} to {manifest['settings']['end']}")


if __name__ == "__main__":
    main()
