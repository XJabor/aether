"""Sweep geometry: how a frequency span is broken into tuner steps.

Pure arithmetic, no hardware. This is the piece that decides where every
bin in the final spectrum comes from, so it is kept separate and testable.

The core trick is that all step centres are *snapped to the global bin
grid*. If the offset from ``f_start`` to a step centre is an exact integer
number of bins, then every FFT bin of that step lands exactly on a global
bin and stitching is a plain array copy -- no resampling, no interpolation,
no accumulated drift across a 900-step sweep.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .. import config


def next_pow2(x: float) -> int:
    return 1 << max(0, math.ceil(math.log2(max(x, 1))))


@dataclass(frozen=True)
class SweepStep:
    """One tuner position and where its samples land in the global array."""

    index: int
    center_hz: float
    fft_lo: int      # slice into the fftshifted spectrum
    fft_hi: int
    glob_lo: int     # matching slice into the global bin array
    glob_hi: int

    @property
    def width(self) -> int:
        return self.fft_hi - self.fft_lo


@dataclass(frozen=True)
class SweepPlan:
    f_start_hz: float
    f_stop_hz: float
    sample_rate: int
    n_fft: int
    bin_hz: float
    crop: float
    keep_bins: int
    frames_per_step: int
    dwell_s: float
    direct_sampling: bool
    freqs_hz: np.ndarray = field(repr=False)
    steps: list[SweepStep] = field(repr=False)

    # -- construction ----------------------------------------------------

    @classmethod
    def create(
        cls,
        f_start_hz: float,
        f_stop_hz: float,
        bin_hz: float,
        sample_rate: int = config.DEFAULT_SAMPLE_RATE,
        crop: float = config.DEFAULT_CROP,
        dwell_s: float = 0.05,
        direct_sampling: bool = False,
    ) -> "SweepPlan":
        if f_stop_hz <= f_start_hz:
            raise ValueError("Stop frequency must be above start frequency.")
        if not 0.1 < crop <= 1.0:
            raise ValueError("Crop must be in (0.1, 1.0].")
        if bin_hz <= 0:
            raise ValueError("Bin width must be positive.")

        cls._validate_range(f_start_hz, f_stop_hz, direct_sampling)

        # FFT size from the requested resolution; the achieved bin width is
        # sample_rate / n_fft, which is what we actually report.
        n_fft = next_pow2(sample_rate / bin_hz)
        n_fft = max(16, min(n_fft, 1 << 20))
        actual_bin_hz = sample_rate / n_fft

        span = f_stop_hz - f_start_hz
        n_bins = int(math.ceil(span / actual_bin_hz))
        if n_bins > config.MAX_TOTAL_BINS:
            raise ValueError(
                f"That range at {actual_bin_hz/1e3:.2f} kHz resolution needs "
                f"{n_bins:,} bins (limit {config.MAX_TOTAL_BINS:,}). "
                "Widen the bin width or narrow the range."
            )

        # Bins kept from each capture. Even, so the kept window is centred.
        keep = int(round(crop * n_fft)) & ~1
        keep = max(2, min(keep, n_fft))
        fft_lo = (n_fft - keep) // 2

        n_steps = int(math.ceil(n_bins / keep))
        half = n_fft // 2

        steps: list[SweepStep] = []
        for k in range(n_steps):
            # Place this step so its kept window starts exactly at global
            # bin k*keep. center_idx is an integer => perfect grid alignment.
            center_idx = k * keep - fft_lo + half
            center_hz = f_start_hz + center_idx * actual_bin_hz

            lo_f, hi_f = fft_lo, fft_lo + keep
            lo_g, hi_g = k * keep, k * keep + keep

            # The last step usually overhangs the end of the global array.
            if hi_g > n_bins:
                over = hi_g - n_bins
                hi_g -= over
                hi_f -= over
            if lo_g < 0:                       # defensive; k >= 0 so unreachable
                lo_f -= lo_g
                lo_g = 0
            if hi_f <= lo_f:
                continue

            steps.append(
                SweepStep(k, center_hz, lo_f, hi_f, lo_g, hi_g)
            )

        if not steps:
            raise ValueError("Sweep produced no steps; check the range.")

        cls._validate_centers(steps, direct_sampling)

        frames = max(1, int(round(dwell_s * sample_rate / n_fft)))
        freqs = f_start_hz + np.arange(n_bins, dtype=np.float64) * actual_bin_hz

        return cls(
            f_start_hz=f_start_hz,
            f_stop_hz=f_stop_hz,
            sample_rate=sample_rate,
            n_fft=n_fft,
            bin_hz=actual_bin_hz,
            crop=crop,
            keep_bins=keep,
            frames_per_step=frames,
            dwell_s=dwell_s,
            direct_sampling=direct_sampling,
            freqs_hz=freqs,
            steps=steps,
        )

    # -- validation ------------------------------------------------------

    @staticmethod
    def _validate_range(f_start: float, f_stop: float, direct: bool) -> None:
        if direct:
            lo, hi, name = config.DIRECT_MIN_HZ, config.DIRECT_MAX_HZ, "direct sampling"
        else:
            lo, hi, name = config.TUNER_MIN_HZ, config.TUNER_MAX_HZ, "the tuner"
        if f_start < lo or f_stop > hi:
            raise ValueError(
                f"{f_start/1e6:.3f}-{f_stop/1e6:.3f} MHz is outside the range "
                f"{name} supports ({lo/1e6:.3f}-{hi/1e6:.3f} MHz)."
            )

    @staticmethod
    def _validate_centers(steps: list[SweepStep], direct: bool) -> None:
        """Step centres sit outside the requested span by up to half a
        capture, so they need checking separately from the span itself."""
        if direct:
            return  # direct sampling has no tuner PLL to drive out of range
        first, last = steps[0].center_hz, steps[-1].center_hz
        if first < config.TUNER_MIN_HZ or last > config.TUNER_MAX_HZ:
            raise ValueError(
                f"Sweep needs tuner centres from {first/1e6:.3f} to "
                f"{last/1e6:.3f} MHz, which falls outside "
                f"{config.TUNER_MIN_HZ/1e6:.0f}-{config.TUNER_MAX_HZ/1e6:.0f} MHz. "
                "Move the range in slightly."
            )

    # -- derived numbers for the UI --------------------------------------

    @property
    def n_bins(self) -> int:
        return len(self.freqs_hz)

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    @property
    def samples_per_step(self) -> int:
        return self.frames_per_step * self.n_fft

    def estimated_sweep_seconds(self) -> float:
        """Rough wall-clock per sweep. Retune cost dominates at fine
        resolution over wide spans, so it is modelled explicitly."""
        settle = config.SETTLE_READS * self.n_fft / self.sample_rate
        capture = self.samples_per_step / self.sample_rate
        retune_overhead = 0.005
        return self.n_steps * (settle + capture + retune_overhead)

    def describe(self) -> str:
        return (
            f"{self.f_start_hz/1e6:.3f}-{self.f_stop_hz/1e6:.3f} MHz | "
            f"{self.bin_hz/1e3:.2f} kHz bins | {self.n_bins:,} bins | "
            f"{self.n_steps} steps | FFT {self.n_fft} x {self.frames_per_step} | "
            f"~{self.estimated_sweep_seconds():.1f}s/sweep"
        )
