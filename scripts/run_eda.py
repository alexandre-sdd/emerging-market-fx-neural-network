#!/usr/bin/env python3
"""Regenerate the basic BIS EDA report, tables, and figures from local exports."""

from fx_forecasting.data.eda import run_eda


if __name__ == "__main__":
    result = run_eda()
    print(result["tables"]["overview"].to_string(index=False))
    print(f"\nReport: {result['report']}")
