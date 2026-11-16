"""Build BIS-only currency context features, forecast screens and offline findings."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fx_forecasting.config import INGESTED_DIR, PROJECT_ROOT
from fx_forecasting.evaluation.context_workflow import build_context_prototype, DEFAULT_OUTPUT, DEFAULT_SETTINGS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--settings-path", type=Path, default=DEFAULT_SETTINGS)
    parser.add_argument("--ingested-dir", type=Path, default=INGESTED_DIR)
    parser.add_argument("--sample-path", type=Path, default=PROJECT_ROOT / "config/research_sample.json")
    args = vars(parser.parse_args())
    manifest = build_context_prototype(**args)
    print(f"Saved {len(manifest['outputs'])} hashed outputs plus manifest to {args['output_dir']}")


if __name__ == "__main__":
    main()
