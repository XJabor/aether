"""Turning a scan into a prompt the model can actually reason about.

A full-range sweep at 10 kHz resolution is ~186,000 bins. That cannot be
sent, and would not help if it could -- the model does not need every noise
sample, it needs the floor, the signals, and what changed.

This module compresses a scan to a few thousand tokens: measurement
conditions, noise floor, a ranked table of detected signals with their
measured properties, a band-level rollup, and (when a reference is loaded)
the differences.
"""
from __future__ import annotations

import numpy as np

from .. import config
from ..analysis import compare
from ..analysis.features import band_occupancy, find_signals, noise_floor_db
from ..storage.session import SessionMeta, SpectrumData

SYSTEM_PROMPT = """\
You are helping someone interpret RF spectrum measurements captured with an \
RTL-SDR (RTL2832U + R820T2), a low-cost software-defined radio.

What the numbers are:
- Power is in dBFS -- decibels relative to the receiver's full scale. It is \
NOT dBm and NOT calibrated to any absolute reference. Comparisons between \
signals in the same scan, or between scans taken at the same gain, are \
meaningful. Absolute power claims are not.
- "avg" is the mean over the whole capture. "peak hold" is the strongest any \
single FFT frame saw. Their difference is a burstiness measure: near zero \
means a continuous carrier, large means an intermittent transmitter. Note \
that wideband FM broadcast normally shows 3-10 dB of this purely from \
modulation, not from intermittency.
- Bandwidths are measured at -3 dB and -10 dB from each peak.

Critical caveat -- this receiver manufactures signals that are not there:
- Images and mixing products from the tuner, which appear at predictable \
offsets from strong nearby stations.
- Harmonics and spurs from the 28.8 MHz sample clock and its multiples.
- Intermodulation when a strong signal (an FM station, a nearby transmitter) \
overloads the front end, scattering false peaks across the band.
- With no antenna filtering, strong FM broadcast frequently appears as ghosts \
elsewhere in the spectrum.

So: treat every band label as a hint about what normally occupies that \
frequency, never as an identification. When something looks unusual, \
consider a receiver artifact before concluding it is a real transmitter, and \
say which you think it is and why. Suggest what would confirm it -- \
retuning, changing gain, adding attenuation, or checking whether it moves \
with a nearby strong signal.

Be concrete and quantitative. Cite specific frequencies and levels from the \
data. If the data does not support an answer, say so plainly rather than \
speculating.\
"""


def _fmt_hz(hz: float) -> str:
    if hz >= 1e9:
        return "%.4f GHz" % (hz / 1e9)
    if hz >= 1e6:
        return "%.4f MHz" % (hz / 1e6)
    if hz >= 1e3:
        return "%.2f kHz" % (hz / 1e3)
    return "%.0f Hz" % hz


def build_context(
    data: SpectrumData,
    meta: SessionMeta | None = None,
    reference: SpectrumData | None = None,
    reference_meta: SessionMeta | None = None,
    freq_range: tuple[float, float] | None = None,
    settings: config.Settings | None = None,
) -> str:
    """Compose the scan summary sent alongside the user's question."""
    settings = settings or config.Settings()

    if freq_range is not None:
        data = _slice(data, *freq_range)
        if reference is not None:
            reference = _slice(reference, *freq_range)

    lines: list[str] = []
    lines.append("=== MEASUREMENT ===")
    lines.extend(_describe_meta(data, meta, freq_range))

    avg = np.asarray(data.avg_db, dtype=np.float64)
    finite = avg[np.isfinite(avg)]
    if finite.size == 0:
        lines.append("\nNo valid data in this range.")
        return "\n".join(lines)

    floor = noise_floor_db(avg)
    lines.append("")
    lines.append("=== NOISE FLOOR ===")
    lines.append("10th percentile: %.1f dBFS   median: %.1f dBFS   max: %.1f dBFS"
                 % (floor, float(np.median(finite)), float(finite.max())))

    signals = find_signals(data, prominence_db=settings.peak_prominence_db)
    lines.append("")
    lines.append("=== DETECTED SIGNALS (%d found, strongest %d shown) ==="
                 % (len(signals), min(len(signals), settings.max_peaks_for_ai)))
    if not signals:
        lines.append("Nothing rose more than %.0f dB above the local floor."
                     % settings.peak_prominence_db)
    else:
        lines.append("%-14s %8s %8s %8s %9s %9s  %s" % (
            "freq", "avg", "peak", "SNR", "BW-3dB", "burst", "band (hint only)"))
        for s in signals[:settings.max_peaks_for_ai]:
            lines.append("%-14s %8.1f %8.1f %8.1f %9s %9.1f  %s" % (
                _fmt_hz(s.freq_hz), s.avg_db, s.peak_hold_db, s.snr_db,
                _fmt_hz(s.bw_3db_hz), s.burstiness_db, s.band or "-"))
        lines.append("(avg/peak/SNR/burst in dB; burst = peak hold minus avg)")

    occupancy = band_occupancy(data, signals)
    if occupancy:
        lines.append("")
        lines.append("=== BAND OCCUPANCY ===")
        for b in occupancy:
            lines.append("%-40s %10s-%-10s floor %6.1f  peak %6.1f  %d signals" % (
                b.name[:40], _fmt_hz(b.lo_hz), _fmt_hz(b.hi_hz),
                b.floor_db, b.peak_db, b.n_signals))

    if reference is not None:
        lines.append("")
        lines.extend(_describe_changes(data, reference, meta, reference_meta, settings))

    return "\n".join(lines)


def _describe_meta(data: SpectrumData, meta: SessionMeta | None,
                   freq_range: tuple[float, float] | None) -> list[str]:
    f = data.freqs_hz
    out = []
    if f.size:
        out.append("Range analysed: %s to %s (%d bins)"
                   % (_fmt_hz(float(f[0])), _fmt_hz(float(f[-1])), f.size))
        if f.size > 1:
            out.append("Resolution: %s per bin" % _fmt_hz(float(f[1] - f[0])))
    if freq_range is not None:
        out.append("(user selected this sub-range of a wider scan)")

    if meta is None:
        out.append("Gain, duration and location were not recorded for this capture.")
        return out

    out.append("Captured: %s to %s (%.0f s, %d complete sweeps)"
               % (meta.started_utc, meta.ended_utc or "?", meta.duration_s,
                  meta.sweep_count))
    out.append("Receiver: gain %s%s, %.3f MS/s, %d ppm%s"
               % (meta.gain_label,
                  " -- AGC ON, levels drift between sweeps" if meta.agc else "",
                  meta.sample_rate / 1e6, meta.ppm,
                  ", HF direct sampling" if meta.direct_sampling else ""))
    if meta.cal_offset_db:
        out.append("A calibration offset of %+.1f dB has been applied."
                   % meta.cal_offset_db)
    if meta.has_location:
        loc = "Location: %.6f, %.6f" % (meta.lat, meta.lon)
        if meta.alt_m is not None:
            loc += " at %.0f m" % meta.alt_m
        if meta.sats:
            loc += " (%d satellites)" % meta.sats
        out.append(loc)
    else:
        out.append("No GPS location was recorded.")
    if meta.notes:
        out.append("User notes: %s" % meta.notes)
    return out


def _describe_changes(data, reference, meta, reference_meta, settings) -> list[str]:
    out = ["=== COMPARISON WITH BASELINE ==="]
    if meta is not None and reference_meta is not None:
        warnings = compare.comparability_warnings(meta, reference_meta)
        if warnings:
            out.append("CAUTION -- these scans are not cleanly comparable:")
            out.extend("  - " + w.replace("\n", " ") for w in warnings)
        out.append("Baseline: %s" % reference_meta.summary())

    changes = compare.find_changes(
        data, reference, threshold_db=settings.delta_threshold_db
    )
    out.append(compare.summarise(changes))
    if changes:
        out.append("Differences above %.0f dB (current minus baseline):"
                   % settings.delta_threshold_db)
        out.append("%-14s %9s %10s %10s  %s"
                   % ("freq", "delta", "now", "baseline", "band"))
        for c in changes[:settings.max_peaks_for_ai]:
            out.append("%-14s %+9.1f %10.1f %10.1f  %s" % (
                _fmt_hz(c.peak_hz), c.peak_delta_db, c.current_db,
                c.reference_db, c.band or "-"))
    return out


def _slice(data: SpectrumData, lo_hz: float, hi_hz: float) -> SpectrumData:
    f = data.freqs_hz
    i = int(np.searchsorted(f, lo_hz, "left"))
    j = int(np.searchsorted(f, hi_hz, "right"))
    i, j = max(0, i), min(f.size, max(j, i + 1))
    return SpectrumData(
        freqs_hz=f[i:j],
        avg_db=data.avg_db[i:j],
        min_db=data.min_db[i:j],
        max_db=data.max_db[i:j],
        peak_hold_db=data.peak_hold_db[i:j],
        count=data.count[i:j],
    )


def estimate_tokens(text: str) -> int:
    """Rough token estimate for the UI. Deliberately approximate -- it exists
    to warn before an oversized request, not to bill anyone."""
    return max(1, len(text) // 4)
