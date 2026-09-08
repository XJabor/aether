"""Session metadata and the on-disk spectrum arrays."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .. import config

ARRAY_KEYS = ("freqs_hz", "avg_db", "min_db", "max_db", "peak_hold_db", "count")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class SessionMeta:
    """Everything needed to interpret a saved spectrum, and to decide
    whether two of them are legitimately comparable.

    Gain, bin width, sample rate and crop all change what the numbers mean,
    so they are stored per session rather than assumed. :mod:`analysis.compare`
    refuses or warns on mismatches instead of silently differencing scans
    that were measured differently.
    """

    name: str = ""
    started_utc: str = field(default_factory=utc_now)
    ended_utc: str = ""

    # Measurement parameters
    f_start_hz: float = 0.0
    f_stop_hz: float = 0.0
    bin_hz: float = 0.0
    sample_rate: int = 0
    n_fft: int = 0
    crop: float = config.DEFAULT_CROP
    gain_db: float | None = None          # None means AGC / auto
    agc: bool = False
    ppm: int = 0
    direct_sampling: bool = False
    cal_offset_db: float = 0.0
    clip_fraction: float = 0.0     # fraction of samples that hit the ADC rails

    # Results
    n_bins: int = 0
    sweep_count: int = 0
    duration_s: float = 0.0

    # Location (all nullable: scanning without GPS is normal)
    lat: float | None = None
    lon: float | None = None
    alt_m: float | None = None
    fix_quality: int | None = None
    sats: int | None = None

    # Provenance
    device: str = ""
    notes: str = ""
    npz_path: str = ""
    app_version: str = config.APP_VERSION
    id: int | None = None

    # -- helpers ---------------------------------------------------------

    @property
    def has_location(self) -> bool:
        return self.lat is not None and self.lon is not None

    @property
    def was_overloaded(self) -> bool:
        from ..sdr.engine import CLIP_WARN_FRACTION
        return self.clip_fraction > CLIP_WARN_FRACTION

    @property
    def gain_label(self) -> str:
        return "AGC" if self.gain_db is None else "%.1f dB" % self.gain_db

    def default_name(self) -> str:
        return "%.3f-%.3f MHz @ %s" % (
            self.f_start_hz / 1e6, self.f_stop_hz / 1e6,
            self.started_utc.replace("T", " ")[:16],
        )

    def slug(self) -> str:
        stamp = re.sub(r"[^0-9]", "", self.started_utc)[:14]
        return "%s_%.0f-%.0fMHz" % (stamp, self.f_start_hz / 1e6, self.f_stop_hz / 1e6)

    def summary(self) -> str:
        loc = ("%.5f, %.5f" % (self.lat, self.lon)) if self.has_location else "no GPS"
        text = (
            "%s | %.3f-%.3f MHz | %.2f kHz bins | %d sweeps | %.0fs | "
            "gain %s | %s"
            % (self.name or self.default_name(), self.f_start_hz / 1e6,
               self.f_stop_hz / 1e6, self.bin_hz / 1e3, self.sweep_count,
               self.duration_s, self.gain_label, loc)
        )
        if self.was_overloaded:
            text += " | OVERLOADED (%.2f%% clipped)" % (100 * self.clip_fraction)
        return text

    def to_row(self) -> dict:
        return asdict(self)

    @classmethod
    def from_row(cls, row) -> "SessionMeta":
        known = {f.name for f in fields(cls)}
        data = {k: row[k] for k in row.keys() if k in known}
        for b in ("agc", "direct_sampling"):
            if b in data and data[b] is not None:
                data[b] = bool(data[b])
        return cls(**data)


@dataclass
class SpectrumData:
    """The arrays themselves, kept out of SQLite and stored as .npz."""

    freqs_hz: np.ndarray
    avg_db: np.ndarray
    min_db: np.ndarray
    max_db: np.ndarray
    peak_hold_db: np.ndarray
    count: np.ndarray

    @classmethod
    def from_accumulator(cls, freqs_hz: np.ndarray, acc) -> "SpectrumData":
        d = acc.as_dict()
        return cls(freqs_hz=np.asarray(freqs_hz, dtype=np.float64), **d)

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            freqs_hz=self.freqs_hz.astype(np.float64),
            avg_db=self.avg_db.astype(np.float32),
            min_db=self.min_db.astype(np.float32),
            max_db=self.max_db.astype(np.float32),
            peak_hold_db=self.peak_hold_db.astype(np.float32),
            count=self.count.astype(np.int32),
        )
        return path

    @classmethod
    def load(cls, path: Path) -> "SpectrumData":
        with np.load(path) as z:
            missing = [k for k in ARRAY_KEYS if k not in z]
            if missing:
                raise ValueError("%s is missing arrays: %s" % (path, ", ".join(missing)))
            return cls(**{k: z[k] for k in ARRAY_KEYS})

    def trace(self, name: str) -> np.ndarray:
        try:
            return getattr(self, name)
        except AttributeError:
            raise KeyError("no such trace: %s" % name) from None

    def with_offset(self, cal_offset_db: float) -> "SpectrumData":
        """Apply a calibration offset. Frequencies and counts are untouched."""
        if not cal_offset_db:
            return self
        return SpectrumData(
            freqs_hz=self.freqs_hz,
            avg_db=self.avg_db + cal_offset_db,
            min_db=self.min_db + cal_offset_db,
            max_db=self.max_db + cal_offset_db,
            peak_hold_db=self.peak_hold_db + cal_offset_db,
            count=self.count,
        )

    def to_csv(self, path: Path) -> Path:
        """Export for Excel / external tooling."""
        arr = np.column_stack([
            self.freqs_hz, self.avg_db, self.min_db,
            self.max_db, self.peak_hold_db, self.count,
        ])
        np.savetxt(
            path, arr, delimiter=",",
            header="frequency_hz,avg_db,min_db,max_db,peak_hold_db,sweep_count",
            comments="", fmt=["%.1f", "%.3f", "%.3f", "%.3f", "%.3f", "%d"],
        )
        return path


@dataclass
class GpsFixRecord:
    t_utc: str
    lat: float
    lon: float
    alt_m: float | None = None
    speed_kt: float | None = None
    fix_quality: int | None = None
    sats: int | None = None
