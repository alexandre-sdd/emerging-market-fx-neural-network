"""Acquire official BIS bulk exports, preserving source CSVs and provenance."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests

from fx_forecasting.config import RAW_DIR

DATASET_URLS = {
    name: f"https://data.bis.org/static/bulk/{name}_csv_col.zip"
    for name in ("WS_XRU", "WS_EER", "WS_CBPOL", "WS_LONG_CPI", "WS_GLI")
}
EXPECTED_FILES = {
    name: [f"{name}.csv"]
    for name in DATASET_URLS
}


def ensure_directories(raw_dir: Path = RAW_DIR) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)


def has_local_file(dataset_name: str, raw_dir: Path = RAW_DIR) -> Path | None:
    for filename in EXPECTED_FILES[dataset_name]:
        candidate = raw_dir / filename
        if candidate.is_file() and candidate.stat().st_size:
            return candidate
    return None


def safe_download(url: str, destination: Path) -> dict:
    """Stream to a temporary file, validate, then atomically install the export.

    ZIPs must contain exactly one CSV. The source archive is retained alongside
    its unchanged CSV. Failed transfers never become apparently complete files.
    """
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"Invalid HTTP(S) URL: {url}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        payload = Path(temporary) / "download"
        with requests.get(url, stream=True, timeout=(15, 120)) as response:
            response.raise_for_status()
            with payload.open("wb") as handle:
                for chunk in response.iter_content(1024 * 1024):
                    handle.write(chunk)
            headers = {k: response.headers.get(k) for k in ("Last-Modified", "ETag")}
        source_member = None
        export = payload
        is_zip = zipfile.is_zipfile(payload)
        if is_zip:
            with zipfile.ZipFile(payload) as archive:
                members = [n for n in archive.namelist() if n.lower().endswith(".csv")]
                if len(members) != 1:
                    raise ValueError(f"Expected one CSV in archive, found {members}")
                source_member = members[0]
                export = Path(temporary) / "export.csv"
                with archive.open(source_member) as source, export.open("wb") as target:
                    shutil.copyfileobj(source, target)
        if destination.suffix == ".csv":
            with export.open(encoding="utf-8-sig", newline="") as handle:
                header = next(csv.reader(handle), [])
            if not {"FREQ", "Series"}.issubset(header):
                raise ValueError("Expected BIS wide CSV with FREQ and Series columns")
        if not export.stat().st_size:
            raise ValueError("Downloaded an empty file")
        digest = hashlib.sha256(export.read_bytes()).hexdigest()
        if is_zip:
            payload.replace(destination.with_suffix(".zip"))
        export.replace(destination)
    return {
        "source_url": url,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "file": destination.name,
        "source_member": source_member,
        "bytes": destination.stat().st_size,
        "sha256": digest,
        "http_headers": headers,
    }


def try_download_missing_data(*, raw_dir: Path = RAW_DIR, force: bool = False) -> list[str]:
    ensure_directories(raw_dir)
    manifest_path = raw_dir / "download_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    missing = []
    for name, default_url in DATASET_URLS.items():
        local = has_local_file(name, raw_dir)
        if local and not force:
            print(f"Found {name}: {local}", flush=True)
            continue
        url = os.getenv(f"BIS_{name}_URL") or default_url
        destination = raw_dir / f"{name}.csv"
        try:
            print(f"Downloading {name} from {url}", flush=True)
            manifest[name] = safe_download(url, destination)
            temporary_manifest = manifest_path.with_suffix(".json.tmp")
            temporary_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
            temporary_manifest.replace(manifest_path)
            print(f"Saved {destination} ({destination.stat().st_size:,} bytes)", flush=True)
        except (requests.RequestException, OSError, ValueError, zipfile.BadZipFile) as exc:
            print(f"Failed {name}: {exc}", flush=True)
            missing.append(name)
    return missing


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download the five official BIS wide CSV exports without processing them.")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR, help="Destination for source CSVs, ZIPs and manifest")
    parser.add_argument("--force", action="store_true", help="Refresh existing exports from BIS")
    args = parser.parse_args(argv)
    missing = try_download_missing_data(raw_dir=args.raw_dir, force=args.force)
    if missing:
        raise SystemExit("Missing datasets: " + ", ".join(missing))
    print("All five BIS datasets are available.")


if __name__ == "__main__":
    main()
