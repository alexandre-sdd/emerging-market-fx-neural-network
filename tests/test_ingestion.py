"""Check calendar, source selection, and failure behavior of the ingestion stage."""

import csv
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from fx_forecasting.data.download import safe_download, try_download_missing_data
from fx_forecasting.data.ingest import expected_periods, ingest_data, sha256_file


def write_sources(directory: Path, *, invalid_value: bool = False, duplicate: bool = False) -> None:
    definitions = {
        "WS_XRU": [
            {"FREQ": "M", "Series": "M:BR:BRL:E", "REF_AREA": "BR", "Reference area": "Brazil", "CURRENCY": "BRL", "COLLECTION": "E", "2026-06": "5", "2026-07": "NaN", "2026-08": "7"},
            {"FREQ": "M", "Series": "M:BR:BRL:A", "REF_AREA": "BR", "Reference area": "Brazil", "CURRENCY": "BRL", "COLLECTION": "A", "2026-06": "999"},
            {"FREQ": "D", "Series": "D:BR:BRL:A", "REF_AREA": "BR", "Reference area": "Brazil", "CURRENCY": "BRL", "COLLECTION": "A", "2026-08-01": "999"},
        ],
        "WS_EER": [
            {"FREQ": "M", "Series": f"M:{kind}:B:US", "REF_AREA": "US", "Reference area": "United States", "EER_TYPE": kind, "EER_BASKET": "B", "2026-06": "100"}
            for kind in ("R", "N")
        ],
        "WS_CBPOL": [{"FREQ": "M", "Series": "M:US", "REF_AREA": "US", "Reference area": "United States", "2026-06": "-0.1"}],
        "WS_LONG_CPI": [
            {"FREQ": "M", "Series": "M:US:628", "REF_AREA": "US", "Reference area": "United States", "UNIT_MEASURE": "628", "2026-06": "bad" if invalid_value else "120"},
            {"FREQ": "M", "Series": "M:US:771", "REF_AREA": "US", "Reference area": "United States", "UNIT_MEASURE": "771", "2026-06": "999"},
        ],
        "WS_GLI": [
            {"FREQ": "Q", "Series": f"Q:USD:{area}:N:A:I:B:USD", "BORROWERS_CTY": area, "Borrowers' country": label, "CURR_DENOM": "USD", "BORROWERS_SECTOR": "N", "LENDERS_SECTOR": "A", "L_POS_TYPE": "I", "L_INSTR": "B", "UNIT_MEASURE": "USD", "UNIT_MULT": "6", "2026-Q1": "1000000"}
            for area, label in (("3P", "All countries excluding residents"), ("BR", "Brazil"), ("4T", "Emerging economies"))
        ],
    }
    if duplicate:
        definitions["WS_XRU"].append(definitions["WS_XRU"][0].copy())
    directory.mkdir(parents=True, exist_ok=True)
    for name, rows in definitions.items():
        columns = list(dict.fromkeys(k for row in rows for k in row))
        with (directory / f"{name}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)


class IngestionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.raw = Path(self.temporary.name) / "raw"
        self.output = Path(self.temporary.name) / "ingested"
        write_sources(self.raw)

    def ingest(self):
        return ingest_data(raw_dir=self.raw, output_dir=self.output, start="2026-01", end="2026-08")

    def test_tidy_roundtrip_retains_missing_native_values_and_metadata(self):
        before = {p.name: p.read_bytes() for p in self.raw.glob("*.csv")}
        manifest = self.ingest()
        observations = pd.read_parquet(self.output / "observations.parquet")
        series = pd.read_parquet(self.output / "series.parquet")
        self.assertEqual(len(observations), 44)
        self.assertEqual(len(series), 7)
        self.assertFalse(observations.duplicated(["dataset", "series_id", "period"]).any())
        fx = observations[observations.dataset.eq("WS_XRU")].set_index("period")
        self.assertEqual(fx.loc["2026-06", "value"], 5)
        self.assertTrue(pd.isna(fx.loc["2026-07", "value"]))
        self.assertEqual(fx.loc["2026-08", "value"], 7)
        self.assertNotIn(999, observations.value.dropna().tolist())
        credit = observations[observations.dataset.eq("WS_GLI")]
        self.assertEqual(set(credit.frequency), {"Q"})
        self.assertEqual(set(credit.period), {"2026-Q1", "2026-Q2"})
        self.assertTrue(credit.loc[credit.period.eq("2026-Q2"), "value"].isna().all())
        self.assertEqual(set(credit.loc[credit.period.eq("2026-Q1"), "value"]), {1000000})
        self.assertEqual(set(series.loc[series.dataset.eq("WS_GLI"), "unit_multiplier"]), {6})
        self.assertIn(-0.1, observations.value.dropna().tolist())
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.raw.glob("*.csv")})
        for name, detail in manifest["outputs"].items():
            self.assertEqual(sha256_file(self.output / name), detail["sha256"])
        for metadata in series.source_metadata_json:
            self.assertIn("FREQ", json.loads(metadata))

    def test_incomplete_quarter_excluded_and_complete_quarter_included(self):
        self.assertEqual(str(expected_periods("usd_credit", "2026-01", "2026-08")[-1]), "2026Q2")
        self.assertEqual(str(expected_periods("usd_credit", "2026-01", "2026-09")[-1]), "2026Q3")
        self.assertEqual(str(expected_periods("usd_credit", "1994-01", "2026-08")[0]), "2000Q1")

    def test_invalid_token_preserves_previously_ingested_files(self):
        self.ingest()
        before = {p.name: p.read_bytes() for p in self.output.iterdir()}
        write_sources(self.raw, invalid_value=True)
        with self.assertRaisesRegex(ValueError, "invalid_numeric_tokens"):
            self.ingest()
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.output.iterdir()})

    def test_duplicate_series_is_rejected_without_deduplication(self):
        write_sources(self.raw, duplicate=True)
        with self.assertRaisesRegex(ValueError, "Duplicate series"):
            self.ingest()
        self.assertFalse(self.output.exists())

    def test_source_hash_mismatch_is_rejected(self):
        (self.raw / "download_manifest.json").write_text(json.dumps({"WS_XRU": {"sha256": "incorrect"}}))
        with self.assertRaisesRegex(ValueError, "source differs"):
            self.ingest()
        self.assertFalse(self.output.exists())


class DownloadTests(unittest.TestCase):
    def response(self, payload):
        from unittest.mock import MagicMock
        response = MagicMock()
        response.__enter__.return_value = response
        response.headers = {}
        response.iter_content.return_value = [payload]
        return response

    def test_zip_download_retains_original_csv_and_archive(self):
        csv_data = b"FREQ,Series,2026-08\nM,M:US,1\n"
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("source.csv", csv_data)
        payload = buffer.getvalue()
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "WS_XRU.csv"
            with patch("fx_forecasting.data.download.requests.get", return_value=self.response(payload)):
                metadata = safe_download("https://example.com/data.zip", destination)
            self.assertEqual(destination.read_bytes(), csv_data)
            self.assertEqual(destination.with_suffix(".zip").read_bytes(), payload)
            self.assertEqual(metadata["sha256"], sha256_file(destination))

    def test_invalid_response_does_not_replace_existing_source(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "WS_XRU.csv"
            destination.write_bytes(b"original source")
            with patch("fx_forecasting.data.download.requests.get", return_value=self.response(b"<html>error</html>")):
                with self.assertRaisesRegex(ValueError, "Expected BIS wide CSV"):
                    safe_download("https://example.com/data.csv", destination)
            self.assertEqual(destination.read_bytes(), b"original source")

    def test_existing_csvs_skip_network_without_creating_processed_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "raw"
            write_sources(raw)
            with patch("fx_forecasting.data.download.requests.get") as get:
                self.assertEqual(try_download_missing_data(raw_dir=raw), [])
            get.assert_not_called()
            self.assertEqual([p.name for p in Path(directory).iterdir()], ["raw"])

    def test_force_refreshes_existing_files_and_records_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "raw"
            write_sources(raw)
            payload = b"FREQ,Series,2026-08\nM,M:US,1\n"
            with patch("fx_forecasting.data.download.requests.get", side_effect=lambda *args, **kwargs: self.response(payload)) as get:
                self.assertEqual(try_download_missing_data(raw_dir=raw, force=True), [])
            self.assertEqual(get.call_count, 5)
            manifest = json.loads((raw / "download_manifest.json").read_text())
            self.assertEqual(len(manifest), 5)
            for name, metadata in manifest.items():
                self.assertEqual(metadata["sha256"], sha256_file(raw / f"{name}.csv"))


if __name__ == "__main__":
    unittest.main()
