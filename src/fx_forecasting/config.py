"""Repository paths shared by scripts and research modules.

Paths target this repository checkout, including an editable installation.
Importing this module does not create directories.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
