#!/usr/bin/env python3
"""Build the focused research-choice report and its report-only eligibility audit."""

import argparse
from pathlib import Path

from fx_forecasting.data.decision_report import build_decision_report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=None)
    args = parser.parse_args()
    print(build_decision_report(output_dir=args.output_dir))
