"""Shared project paths; importing this module does not create directories."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
MODELS_DIR = OUTPUT_DIR / "models"
MEDIA_DIR = OUTPUT_DIR / "media"
METRICS_DIR = OUTPUT_DIR / "metrics"
SUBMISSIONS_DIR = OUTPUT_DIR / "submissions"
