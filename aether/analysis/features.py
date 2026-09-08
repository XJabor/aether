"""Turning a spectrum into a short list of things worth talking about.

This is what makes the AI feature possible: a full-range scan is ~186,000
bins, far too many to send to a model. Reducing it to a noise floor, a
handful of measured signals, and a band-level rollup keeps the payload
small while preserving what actually matters.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks

from . import bands
from ..storage.session import SpectrumData


@dataclass
class Signal:
    """One detected peak, with everything measurable about it."""

    freq_hz: float
    bin_index: int
    avg_db: float
    min_db: float
    max_db: float
    peak_hold_db: float
    prominence_db: float
    local_floor_db: float
    bw_3db_hz: float
    bw_10db_hz: float
    band: str | None

    @property
    def snr_db(self) -> float:
        return self.avg_db - self.local_floor_db

    @property
    def burstiness_db(self) -> float:
        """How much stronger a single frame was than the long-run average.

        Near zero means a continuous carrier (a broadcast station). Large
        means intermittent -- a handheld, a data burst, a transponder.
        """
        return float(self.peak_hold_db - self.avg_db)

    @property
    def character(self) -> str:
        b = self.burstiness_db
        if b < 3:
            return "continuous"
        if b < 10:
            return "intermittent"
        return "bursty"

    def describe(self) -> str:
        return (
            "%.4f MHz  %.1f dBFS  SNR %.1f dB  BW %.1f kHz  %s  %s"
            % (self.freq_hz / 1e6, self.avg_db, self.snr_db,
               self.bw_3db_hz / 1e3, self.character, self.band or "unallocated")
        )


def noise_floor_db(trace: np.ndarray, percentile: float = 10.0) -> float:
    """Global floor estimate. A low percentile rather than the minimum, so
    one dead bin cannot define the floor for the whole scan."""
    finite = trace[np.isfinite(trace)]
    if finite.size == 0:
        return float("nan")
    return float(np.percentile(finite, percentile))


def rolling_floor_db(trace: np.ndarray, window_bins: int = 501,
                     percentile: float = 25.0) -> np.ndarray:
    """Per-bin local floor.

    The floor is not flat across 1.7 GHz -- it tilts with tuner gain and
    steps at band edges. Comparing a peak to a *local* floor rather than a
    global one is what stops a strong band from hiding weak signals
    elsewhere.

    Implemented by decimating, taking percentiles over coarse blocks, then
    interpolating back. Exact per-bin percentiles would be far slower with
    no meaningful benefit at this scale.
    """
    n = trace.size
    filled = np.where(np.isfinite(trace), trace, np.nan)
    step = max(1, window_bins // 4)
    centers, values = [], []
    for start in range(0, n, step):
        block = filled[start:start + window_bins]
        block = block[np.isfinite(block)]
        if block.size:
            centers.append(min(start + window_bins // 2, n - 1))
            values.append(np.percentile(block, percentile))
    if not centers:
        return np.full(n, np.nan)
    return np.interp(np.arange(n), np.array(centers), np.array(values))


def _width_at_drop(trace: np.ndarray, idx: int, drop_db: float,
                   bin_hz: float, max_bins: int = 20000) -> float:
    """Width where the trace falls `drop_db` below the peak."""
    target = trace[idx] - drop_db
    lo = idx
    while lo > 0 and idx - lo < max_bins and np.isfinite(trace[lo]) and trace[lo] > target:
        lo -= 1
    hi = idx
    n = trace.size
    while hi < n - 1 and hi - idx < max_bins and np.isfinite(trace[hi]) and trace[hi] > target:
        hi += 1
    return max(hi - lo, 1) * bin_hz


def find_signals(
    data: SpectrumData,
    prominence_db: float = 6.0,
    trace_name: str = "avg_db",
    max_signals: int = 200,
    min_separation_hz: float = 0.0,
) -> list[Signal]:
    """Detect peaks and measure each one. Sorted strongest first."""
    trace = np.asarray(data.trace(trace_name), dtype=np.float64)
    freqs = data.freqs_hz
    if trace.size < 3:
        return []

    bin_hz = float(freqs[1] - freqs[0]) if freqs.size > 1 else 1.0
    floor = rolling_floor_db(trace)

    # Peak finding needs a finite series; unmeasured bins become the floor
    # so they cannot masquerade as peaks or as valleys next to one.
    filled = np.where(np.isfinite(trace), trace, np.nan)
    filled = np.where(np.isnan(filled), floor, filled)

    distance = None
    if min_separation_hz > 0 and bin_hz > 0:
        distance = max(1, int(min_separation_hz / bin_hz))

    idx, props = find_peaks(filled, prominence=prominence_db, distance=distance)
    if idx.size == 0:
        return []

    order = np.argsort(filled[idx])[::-1][:max_signals]
    out: list[Signal] = []
    for j in order:
        i = int(idx[j])
        out.append(Signal(
            freq_hz=float(freqs[i]),
            bin_index=i,
            avg_db=_at(data.avg_db, i),
            min_db=_at(data.min_db, i),
            max_db=_at(data.max_db, i),
            peak_hold_db=_at(data.peak_hold_db, i),
            prominence_db=float(props["prominences"][j]),
            local_floor_db=float(floor[i]),
            bw_3db_hz=_width_at_drop(filled, i, 3.0, bin_hz),
            bw_10db_hz=_width_at_drop(filled, i, 10.0, bin_hz),
            band=bands.lookup(float(freqs[i])),
        ))
    return out


def _at(arr: np.ndarray, i: int) -> float:
    v = float(arr[i])
    return v if np.isfinite(v) else float("nan")


@dataclass
class BandOccupancy:
    name: str
    lo_hz: float
    hi_hz: float
    floor_db: float
    median_db: float
    peak_db: float
    n_signals: int

    def describe(self) -> str:
        return ("%-38s %8.3f-%8.3f MHz  floor %6.1f  peak %6.1f  %d signals"
                % (self.name, self.lo_hz / 1e6, self.hi_hz / 1e6,
                   self.floor_db, self.peak_db, self.n_signals))


def band_occupancy(data: SpectrumData, signals: list[Signal] | None = None,
                   min_bins: int = 8) -> list[BandOccupancy]:
    """Per-allocation rollup across the scanned span."""
    freqs = data.freqs_hz
    if freqs.size == 0:
        return []
    avg = np.asarray(data.avg_db, dtype=np.float64)
    peak = np.asarray(data.peak_hold_db, dtype=np.float64)
    sigs = signals if signals is not None else []

    out: list[BandOccupancy] = []
    for b in bands.bands_overlapping(float(freqs[0]), float(freqs[-1])):
        if b.width_hz <= 0:
            continue
        lo = int(np.searchsorted(freqs, b.lo_hz, "left"))
        hi = int(np.searchsorted(freqs, b.hi_hz, "right"))
        if hi - lo < min_bins:
            continue
        seg_avg = avg[lo:hi][np.isfinite(avg[lo:hi])]
        seg_peak = peak[lo:hi][np.isfinite(peak[lo:hi])]
        if seg_avg.size == 0:
            continue
        out.append(BandOccupancy(
            name=b.name,
            lo_hz=max(b.lo_hz, float(freqs[0])),
            hi_hz=min(b.hi_hz, float(freqs[-1])),
            floor_db=float(np.percentile(seg_avg, 10)),
            median_db=float(np.median(seg_avg)),
            peak_db=float(seg_peak.max()) if seg_peak.size else float(seg_avg.max()),
            n_signals=sum(1 for s in sigs if b.contains(s.freq_hz)),
        ))
    return out
