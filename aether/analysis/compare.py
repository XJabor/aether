"""Comparing a scan against a saved baseline.

This is what turns a pile of spectra into something actionable: not "what
is in my spectrum" but "what is here now that was not here last time".
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import bands
from ..storage.session import SessionMeta, SpectrumData


def align_and_diff(
    current: SpectrumData,
    reference: SpectrumData,
    trace: str = "avg_db",
) -> tuple[np.ndarray, np.ndarray]:
    """Difference two spectra on the current scan's frequency grid.

    Returns (freqs_hz, delta_db) where delta is current minus reference.
    Bins outside the overlap, or unmeasured in either scan, come back NaN so
    the plot breaks the line rather than inventing a value.
    """
    f_cur = np.asarray(current.freqs_hz, dtype=np.float64)
    f_ref = np.asarray(reference.freqs_hz, dtype=np.float64)
    cur = np.asarray(current.trace(trace), dtype=np.float64)
    ref = np.asarray(reference.trace(trace), dtype=np.float64)

    if f_cur.size == 0 or f_ref.size == 0:
        return f_cur, np.full(f_cur.shape, np.nan)

    same_grid = (
        f_cur.size == f_ref.size
        and abs(f_cur[0] - f_ref[0]) < 1e-6
        and abs(f_cur[-1] - f_ref[-1]) < 1e-6
    )
    if same_grid:
        ref_on_grid = ref
    else:
        # Interpolate the reference onto our grid. NaNs are dropped first so
        # they cannot smear across neighbouring bins during interpolation.
        good = np.isfinite(ref)
        if good.sum() < 2:
            return f_cur, np.full(f_cur.shape, np.nan)
        ref_on_grid = np.interp(
            f_cur, f_ref[good], ref[good], left=np.nan, right=np.nan
        )

    delta = cur - ref_on_grid
    delta[~np.isfinite(cur) | ~np.isfinite(ref_on_grid)] = np.nan
    return f_cur, delta


def comparability_warnings(a: SessionMeta, b: SessionMeta) -> list[str]:
    """Reasons two sessions may not be fairly comparable.

    Nothing here blocks a comparison -- but a delta between scans taken at
    different gains is measuring the receiver, not the environment, and the
    user needs to be told that rather than shown a confident graph.
    """
    out: list[str] = []

    if a.gain_db != b.gain_db:
        out.append(
            "Different gain (%s vs %s). The delta will mostly reflect the "
            "gain change, not the RF environment." % (a.gain_label, b.gain_label)
        )
    if a.agc or b.agc:
        which = "both scans" if (a.agc and b.agc) else ("this scan" if a.agc else "the reference")
        out.append(
            "AGC was enabled for %s, so its gain drifted during capture. "
            "Absolute levels are not comparable." % which
        )
    if abs(a.bin_hz - b.bin_hz) > 1e-6:
        out.append(
            "Different resolution (%.2f vs %.2f kHz). Narrow signals will "
            "measure differently in each." % (a.bin_hz / 1e3, b.bin_hz / 1e3)
        )
    if a.sample_rate != b.sample_rate:
        out.append("Different sample rate (%.3f vs %.3f MS/s)."
                   % (a.sample_rate / 1e6, b.sample_rate / 1e6))
    if a.direct_sampling != b.direct_sampling:
        out.append("One scan used HF direct sampling and the other did not.")
    if abs(a.cal_offset_db - b.cal_offset_db) > 1e-9:
        out.append("Different calibration offsets (%.1f vs %.1f dB)."
                   % (a.cal_offset_db, b.cal_offset_db))

    lo = max(a.f_start_hz, b.f_start_hz)
    hi = min(a.f_stop_hz, b.f_stop_hz)
    if hi <= lo:
        out.append("The two scans do not overlap in frequency at all.")
    else:
        span_a = a.f_stop_hz - a.f_start_hz
        if span_a > 0 and (hi - lo) / span_a < 0.9:
            out.append(
                "Only %.1f%% of this scan's range overlaps the reference "
                "(%.3f-%.3f MHz)."
                % (100.0 * (hi - lo) / span_a, lo / 1e6, hi / 1e6)
            )

    if a.has_location and b.has_location:
        from ..storage.db import haversine_km

        km = haversine_km(a.lat, a.lon, b.lat, b.lon)
        if km > 0.25:
            out.append(
                "Recorded %.2f km apart, so differences may simply be "
                "different locations." % km
            )
    return out


@dataclass
class Change:
    """A contiguous run of bins that moved by more than the threshold."""

    lo_hz: float
    hi_hz: float
    peak_hz: float
    peak_delta_db: float
    mean_delta_db: float
    current_db: float
    reference_db: float
    band: str | None

    @property
    def width_hz(self) -> float:
        return self.hi_hz - self.lo_hz

    @property
    def direction(self) -> str:
        return "appeared/stronger" if self.peak_delta_db > 0 else "gone/weaker"

    def describe(self) -> str:
        return (
            "%.4f MHz  %+.1f dB  (%.1f -> %.1f dBFS)  width %.1f kHz  %s  [%s]"
            % (self.peak_hz / 1e6, self.peak_delta_db, self.reference_db,
               self.current_db, self.width_hz / 1e3, self.direction,
               self.band or "unallocated")
        )


def find_changes(
    current: SpectrumData,
    reference: SpectrumData,
    threshold_db: float = 6.0,
    trace: str = "avg_db",
    min_bins: int = 1,
    max_changes: int = 100,
) -> list[Change]:
    """Contiguous regions where the scans differ by more than threshold_db.

    Grouping adjacent bins matters: a single 200 kHz FM carrier appearing
    would otherwise report as ~20 separate "changes" at 10 kHz resolution
    and swamp the list.
    """
    freqs, delta = align_and_diff(current, reference, trace)
    if freqs.size == 0:
        return []

    cur = np.asarray(current.trace(trace), dtype=np.float64)
    ref_vals = cur - delta                            # reference on our grid

    flags = np.isfinite(delta) & (np.abs(delta) >= threshold_db)
    if not flags.any():
        return []

    changes: list[Change] = []
    i = 0
    n = flags.size
    while i < n:
        if not flags[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and flags[j + 1]:
            j += 1
        if (j - i + 1) >= min_bins:
            seg = delta[i:j + 1]
            k = int(np.nanargmax(np.abs(seg))) + i
            changes.append(Change(
                lo_hz=float(freqs[i]),
                hi_hz=float(freqs[j]),
                peak_hz=float(freqs[k]),
                peak_delta_db=float(delta[k]),
                mean_delta_db=float(np.nanmean(seg)),
                current_db=float(cur[k]),
                reference_db=float(ref_vals[k]),
                band=bands.lookup(float(freqs[k])),
            ))
        i = j + 1

    changes.sort(key=lambda c: abs(c.peak_delta_db), reverse=True)
    return changes[:max_changes]


def summarise(changes: list[Change]) -> str:
    if not changes:
        return "No differences above the threshold."
    up = sum(1 for c in changes if c.peak_delta_db > 0)
    down = len(changes) - up
    return "%d change(s): %d stronger/new, %d weaker/gone." % (len(changes), up, down)
