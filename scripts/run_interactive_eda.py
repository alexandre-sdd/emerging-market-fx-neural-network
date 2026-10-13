#!/usr/bin/env python3
"""Build matching Plotly EDA and quality reports from local BIS evidence."""

import argparse
from pathlib import Path

from fx_forecasting.data.interactive_eda import build_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    print(build_report(output_dir=args.output_dir))
