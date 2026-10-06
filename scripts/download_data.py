#!/usr/bin/env python3
"""Download BIS FX and macro data if local copies are missing.

This script is intentionally robust and intentionally conservative:
- it does not assume a single public BIS archive URL;
- it checks for existing local data before downloading;
- it supports environment-variable-based URLs for the exact BIS files used in the project.
- if no local data are present and no URLs are configured, it exits with a clear message.

The project expects the following raw files to appear under `data/raw/`:
- WS_XRU.csv or equivalent
- WS_EER.csv or equivalent
- WS_CBPOL.csv or equivalent
- WS_LONG_CPI.csv or equivalent
- WS_GLI.csv or equivalent

If your institution or course has a preferred BIS mirror, simply set the corresponding
environment variables before running this script.
"""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

DATASET_URLS = {
    "WS_XRU": os.getenv("BIS_WS_XRU_URL"),
    "WS_EER": os.getenv("BIS_WS_EER_URL"),
    "WS_CBPOL": os.getenv("BIS_WS_CBPOL_URL"),
    "WS_LONG_CPI": os.getenv("BIS_WS_LONG_CPI_URL"),
    "WS_GLI": os.getenv("BIS_WS_GLI_URL"),
}

EXPECTED_FILES = {
    "WS_XRU": ["WS_XRU.csv", "WS_XRU.xls", "WS_XRU.xlsx", "WS_XRU.txt"],
    "WS_EER": ["WS_EER.csv", "WS_EER.xls", "WS_EER.xlsx", "WS_EER.txt"],
    "WS_CBPOL": ["WS_CBPOL.csv", "WS_CBPOL.xls", "WS_CBPOL.xlsx", "WS_CBPOL.txt"],
    "WS_LONG_CPI": ["WS_LONG_CPI.csv", "WS_LONG_CPI.xls", "WS_LONG_CPI.xlsx", "WS_LONG_CPI.txt"],
    "WS_GLI": ["WS_GLI.csv", "WS_GLI.xls", "WS_GLI.xlsx", "WS_GLI.txt"],
}


def ensure_directories() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def has_local_file(dataset_name: str) -> Path | None:
    for filename in EXPECTED_FILES.get(dataset_name, []):
        candidate = RAW_DIR / filename
        if candidate.exists():
            return candidate
    return None


def safe_download(url: str, destination: Path) -> None:
    if not url:
        raise ValueError("Empty URL provided")

    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"Invalid URL: {url}")

    print(f"Downloading {url} -> {destination}")
    urllib.request.urlretrieve(url, destination)


def try_download_missing_data() -> None:
    missing = []

    for dataset_name, url in DATASET_URLS.items():
        if has_local_file(dataset_name):
            print(f"Found local copy for {dataset_name}: {has_local_file(dataset_name)}")
            continue

        if not url:
            missing.append(dataset_name)
            continue

        extension = Path(urlparse(url).path).suffix.lower()
        if extension not in {".csv", ".xls", ".xlsx", ".txt"}:
            print(f"Unsupported or unknown file format for {dataset_name}: {url}")
            missing.append(dataset_name)
            continue
        destination = RAW_DIR / (dataset_name + extension)
        try:
            safe_download(url, destination)
            print(f"Saved downloaded file to {destination}")
        except Exception as exc:
            print(f"Failed to download {dataset_name} from configured URL: {exc}")
            missing.append(dataset_name)

    if missing:
        print("\nMissing data files: " + ", ".join(missing))
        print("\nSome datasets are missing, have no configured URL, or could not be downloaded.")
        print("Set one or more of the following environment variables before running the script:")
        print("  BIS_WS_XRU_URL")
        print("  BIS_WS_EER_URL")
        print("  BIS_WS_CBPOL_URL")
        print("  BIS_WS_LONG_CPI_URL")
        print("  BIS_WS_GLI_URL")
        print("\nExample:")
        print("  export BIS_WS_XRU_URL='https://example.com/WS_XRU.csv'")
        print("  python scripts/download_data.py")
        print("\nThe script will then place the downloaded files under data/raw/ automatically.")
    else:
        print("\nAll configured datasets are present or successfully downloaded.")


def main() -> None:
    ensure_directories()
    print(f"Checking for project data in: {RAW_DIR}")
    try_download_missing_data()


if __name__ == "__main__":
    main()

