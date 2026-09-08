"""Application paths, constants and persisted settings."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path

APP_NAME = "Aether"
APP_VERSION = "0.1.0"

# The project was called RTLBaseline before it grew beyond one receiver.
# Existing installs keep working: their data directory and stored API keys
# are still found under the old name.
LEGACY_APP_NAME = "RTLBaseline"

# --- Paths ---------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LIB_DIR = PROJECT_ROOT / "lib"


def data_root() -> Path:
    """Where scan sessions live. Override with AETHER_DATA.

    Nothing is ever moved automatically. Saved sessions store absolute
    paths to their .npz files, so relocating the directory behind the
    user's back would break every one of them. If a pre-rename directory
    exists and the new one does not, we simply keep using the old one --
    ``tools/migrate_data.py`` moves it properly when the user chooses to.
    """
    env = os.environ.get("AETHER_DATA") or os.environ.get("RTLBASELINE_DATA")
    if env:
        root = Path(env)
    else:
        root = Path.home() / "Documents" / APP_NAME
        legacy = Path.home() / "Documents" / LEGACY_APP_NAME
        if not root.exists() and legacy.is_dir():
            root = legacy
    root.mkdir(parents=True, exist_ok=True)
    return root


def db_path() -> Path:
    return data_root() / "sessions.db"


def spectra_dir() -> Path:
    d = data_root() / "spectra"
    d.mkdir(parents=True, exist_ok=True)
    return d


# --- Hardware constants (RTL-SDR Blog V3 / RTL2832U + R820T2) -----------

# Tuner range. The R820T2 is usually specced 24 MHz - 1766 MHz.
TUNER_MIN_HZ = 24_000_000
TUNER_MAX_HZ = 1_766_000_000

# Direct sampling (Q-branch) uses the 28.8 MHz ADC clock directly.
# Below 14.4 MHz is the first Nyquist zone and is clean; above that,
# images fold back and results need careful interpretation.
DIRECT_MIN_HZ = 500_000
DIRECT_NYQUIST_HZ = 14_400_000
DIRECT_MAX_HZ = 28_800_000

ADC_CLOCK_HZ = 28_800_000

# Sample rates the RTL2832U handles without dropping samples.
# 3.2 MS/s drops samples on most dongles; offered but not default.
SAMPLE_RATES = [2_400_000, 2_048_000, 1_800_000, 1_024_000, 2_560_000, 3_200_000]
DEFAULT_SAMPLE_RATE = 2_400_000

# Fraction of each capture we keep. The outer edges are filter rolloff
# and are not trustworthy, so we only stitch the middle.
DEFAULT_CROP = 0.80

# Reads discarded after a retune, to let the R820T2 PLL settle.
SETTLE_READS = 2

# Bins either side of the tuned center replaced by interpolation,
# to remove the RTL2832U's fixed DC offset artifact.
DC_SPIKE_BINS = 2

# Guardrail: refuse absurd bin counts that would wedge the UI.
MAX_TOTAL_BINS = 2_000_000


# --- Settings ------------------------------------------------------------

@dataclass
class Settings:
    """User settings, persisted as JSON next to the database."""

    # Measurement
    cal_offset_db: float = 0.0        # added to every dB value; 0 => raw dBFS
    ppm: int = 0

    # AI
    ai_provider: str = "anthropic"    # "anthropic" | "openai"
    anthropic_model: str = "claude-opus-5"
    openai_model: str = "gpt-4o"

    # GPS
    gps_port: str = ""                # "" => auto-detect
    gps_baud: int = 0                 # 0 => auto-detect

    # Analysis
    peak_prominence_db: float = 6.0
    delta_threshold_db: float = 6.0
    max_peaks_for_ai: int = 60

    @classmethod
    def path(cls) -> Path:
        return data_root() / "settings.json"

    @classmethod
    def load(cls) -> "Settings":
        p = cls.path()
        if not p.exists():
            return cls()
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return cls()
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in raw.items() if k in known})

    def save(self) -> None:
        self.path().write_text(
            json.dumps(asdict(self), indent=2), encoding="utf-8"
        )
