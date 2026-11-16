"""Build frozen groups, multivariate datasets and a shareable comparison report."""

import argparse
from pathlib import Path
import sys

# A checkout command also works before editable-install path hooks are loaded.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))

from fx_forecasting.config import INGESTED_DIR, PROJECT_ROOT
from fx_forecasting.evaluation.clustering_handoff import build_clustering_handoff, DEFAULT_OUTPUT, DEFAULT_SETTINGS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--settings-path", type=Path, default=DEFAULT_SETTINGS)
    parser.add_argument("--ingested-dir", type=Path, default=INGESTED_DIR)
    parser.add_argument("--sample-path", type=Path, default=PROJECT_ROOT/"config/research_sample.json")
    parser.add_argument("--bootstrap-reps", type=int, default=None, help="Explicit smoke-run override; default50")
    parser.add_argument("--no-dtw", action="store_true", help="Explicit partial comparison for smoke tests")
    parser.add_argument("--skip-report", action="store_true", help="Export data/tables without rendering the report")
    args = vars(parser.parse_args())
    args["include_dtw"] = not args.pop("no_dtw")
    args["write_report"] = not args.pop("skip_report")
    result = build_clustering_handoff(**args)
    print(f"Saved {len(result['outputs'])} verified outputs to {args['output_dir']}")
    print("X_local:", result["array_shapes"]["X_local"], "Y:", result["array_shapes"]["Y"])


if __name__ == "__main__":
    main()
