"""The sweep loop.

Deliberately free of Qt so it can be driven from the CLI and from tests.
:mod:`aether.sdr.worker` wraps this in a QThread for the GUI.

Power scaling: the FFT is normalised by the window's coherent gain, so a
full-scale complex sinusoid reads 0 dBFS. Everything downstream is
therefore in **dBFS** -- honest, uncalibrated, and comparable between
scans taken at the same gain. It is not dBm and must not be labelled as
such without a real calibration.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .. import config
from .accumulate import SpectrumAccumulator
from .device import SdrDevice
from .sweep import SweepPlan

# A normalised sample at or beyond this magnitude came from a railed ADC code.
CLIP_LEVEL = 0.99
# Above this fraction of clipped samples the capture is compressed enough
# that the numbers should not be trusted.
CLIP_WARN_FRACTION = 0.001


@dataclass
class StopPolicy:
    """When to stop scanning. ``indefinite`` runs until stop is signalled."""

    mode: str = "indefinite"          # "duration" | "sweeps" | "indefinite"
    duration_s: float | None = None
    max_sweeps: int | None = None

    def describe(self) -> str:
        if self.mode == "duration":
            return "for %.0f s" % (self.duration_s or 0)
        if self.mode == "sweeps":
            return "for %d sweeps" % (self.max_sweeps or 0)
        return "until stopped"


@dataclass
class SweepProgress:
    sweep_index: int
    step_index: int
    n_steps: int
    elapsed_s: float
    center_hz: float


class SweepEngine:
    """Drives a device through a :class:`SweepPlan`, accumulating statistics.

    Note on partial sweeps: when a scan is stopped mid-sweep the segments
    already captured are *kept*. The accumulator counts samples per bin, so
    the average stays correct; the bins covered by the partial sweep simply
    have one extra observation. The per-bin ``count`` array records this
    honestly rather than silently discarding good measurements.
    """

    def __init__(
        self,
        device: SdrDevice,
        plan: SweepPlan,
        gain: float | str = "auto",
        ppm: int = 0,
    ) -> None:
        self.device = device
        self.plan = plan
        self.gain = gain
        self.ppm = ppm
        self.acc = SpectrumAccumulator(plan.n_bins)
        self.stop_event = threading.Event()

        # Hann window, normalised so a full-scale tone reads 0 dBFS.
        w = np.hanning(plan.n_fft).astype(np.float64)
        self._window = w
        self._window_gain = w.sum()

        # Read in chunks of at least ~16k samples: big enough that USB
        # transfers stay efficient, small enough (~7 ms at 2.4 MS/s) that
        # Stop is checked often and responds immediately.
        frames_per_chunk = max(1, int(np.ceil(16384 / plan.n_fft)))
        self._chunk_frames = min(frames_per_chunk, plan.frames_per_step)
        self._chunk_samples = self._chunk_frames * plan.n_fft

        self.started_at: float | None = None
        self.finished_at: float | None = None

        # Overload tracking. A clipped capture is not just noisy, it is
        # wrong: the front end compresses, strong signals read low, and
        # intermodulation scatters false peaks across the band. Silently
        # recording a clipped baseline is the worst failure this app can
        # have, so every sample is checked.
        self.clipped_samples = 0
        self.total_samples = 0

    # -- public ----------------------------------------------------------

    def stop(self) -> None:
        self.stop_event.set()

    def run(
        self,
        policy: StopPolicy,
        on_segment: Callable[[int, int], None] | None = None,
        on_sweep: Callable[[int], None] | None = None,
        on_progress: Callable[[SweepProgress], None] | None = None,
    ) -> SpectrumAccumulator:
        plan = self.plan
        self.device.configure(
            plan.sample_rate, self.gain, self.ppm, plan.direct_sampling
        )

        # The plan's bin width, step centres and frequency axis are all
        # derived from the requested sample rate. If the hardware quietly
        # snapped to a different one, every frequency we report would be
        # wrong by that ratio -- so refuse rather than mislabel the data.
        actual = int(getattr(self.device, "sample_rate", plan.sample_rate))
        if abs(actual - plan.sample_rate) > max(1, plan.sample_rate * 1e-4):
            raise RuntimeError(
                "Device set %.6f MS/s but the sweep was planned for %.6f MS/s.\n"
                "Every reported frequency would be off by %.3f%%. Choose a "
                "sample rate the RTL2832U can divide its 28.8 MHz clock to "
                "(2.4 MS/s is exact)."
                % (actual / 1e6, plan.sample_rate / 1e6,
                   100.0 * abs(actual - plan.sample_rate) / plan.sample_rate)
            )

        self.started_at = time.monotonic()
        sweep_i = 0
        try:
            while not self._should_stop(policy, sweep_i):
                for step_i, step in enumerate(plan.steps):
                    if self._should_stop(policy, sweep_i, mid_sweep=True):
                        break

                    avg_lin, frame_max_lin = self._capture_step(step.center_hz)
                    if avg_lin is None:
                        break  # stopped during capture

                    avg_seg = avg_lin[step.fft_lo:step.fft_hi]
                    max_seg = frame_max_lin[step.fft_lo:step.fft_hi]
                    self.acc.add_segment(step.glob_lo, step.glob_hi, avg_seg, max_seg)

                    if on_segment is not None:
                        on_segment(step.glob_lo, step.glob_hi)
                    if on_progress is not None:
                        on_progress(SweepProgress(
                            sweep_index=sweep_i,
                            step_index=step_i,
                            n_steps=plan.n_steps,
                            elapsed_s=self.elapsed(),
                            center_hz=step.center_hz,
                        ))
                else:
                    # Only counts as a completed sweep if no break happened.
                    self.acc.mark_sweep_complete()
                    sweep_i += 1
                    if on_sweep is not None:
                        on_sweep(sweep_i)
                    continue
                break
        finally:
            self.finished_at = time.monotonic()
        return self.acc

    @property
    def clip_fraction(self) -> float:
        if self.total_samples == 0:
            return 0.0
        return self.clipped_samples / self.total_samples

    @property
    def is_overloaded(self) -> bool:
        return self.clip_fraction > CLIP_WARN_FRACTION

    def overload_warning(self) -> str | None:
        """A plain explanation of an overloaded capture, or None."""
        if not self.is_overloaded:
            return None
        return (
            "The receiver was overloaded: %.2f%% of samples hit the ADC "
            "limit. Strong signals will read low, and intermodulation may "
            "have created peaks that are not real transmitters. Lower the "
            "gain and scan again." % (100.0 * self.clip_fraction)
        )

    def elapsed(self) -> float:
        if self.started_at is None:
            return 0.0
        end = self.finished_at if self.finished_at is not None else time.monotonic()
        return end - self.started_at

    # -- internals -------------------------------------------------------

    def _should_stop(self, policy: StopPolicy, sweeps_done: int,
                     mid_sweep: bool = False) -> bool:
        if self.stop_event.is_set():
            return True
        if policy.mode == "duration" and policy.duration_s is not None:
            if self.elapsed() >= policy.duration_s:
                return True
        if policy.mode == "sweeps" and policy.max_sweeps is not None:
            # Checked only between sweeps; mid-sweep this would truncate the
            # final sweep and leave it half-populated.
            if not mid_sweep and sweeps_done >= policy.max_sweeps:
                return True
        return False

    def _capture_step(self, center_hz: float):
        """Tune, settle, capture. Returns (mean power, per-frame max power),
        both fftshifted and DC-corrected, or (None, None) if stopped."""
        plan = self.plan
        self.device.set_center_freq(center_hz)

        # The R820T2 PLL needs a moment after a retune. Reading and throwing
        # away the first couple of chunks is the difference between clean
        # step boundaries and garbage at every seam.
        for _ in range(config.SETTLE_READS):
            if self.stop_event.is_set():
                return None, None
            self.device.read(self._chunk_samples)

        n_fft = plan.n_fft
        power_sum = np.zeros(n_fft, dtype=np.float64)
        power_max = np.zeros(n_fft, dtype=np.float64)
        frames_done = 0

        while frames_done < plan.frames_per_step:
            if self.stop_event.is_set():
                return None, None
            want = min(self._chunk_frames, plan.frames_per_step - frames_done)
            samples = self.device.read(want * n_fft)
            got = samples.size // n_fft
            if got == 0:
                continue

            # The 8-bit ADC rails at +/-1.0 after normalisation.
            self.clipped_samples += int(
                np.count_nonzero(np.abs(samples.real) >= CLIP_LEVEL)
                + np.count_nonzero(np.abs(samples.imag) >= CLIP_LEVEL)
            )
            self.total_samples += 2 * samples.size

            block = samples[: got * n_fft].reshape(got, n_fft)
            spec = np.fft.fft(block * self._window, axis=1) / self._window_gain
            power = (spec.real ** 2 + spec.imag ** 2)

            power_sum += power.sum(axis=0)
            np.maximum(power_max, power.max(axis=0), out=power_max)
            frames_done += got

        mean_power = power_sum / max(frames_done, 1)
        mean_power = np.fft.fftshift(mean_power)
        power_max = np.fft.fftshift(power_max)

        self._remove_dc_spike(mean_power)
        self._remove_dc_spike(power_max)
        return mean_power, power_max

    @staticmethod
    def _remove_dc_spike(shifted: np.ndarray) -> None:
        """Interpolate across the fixed DC artifact at the tuned centre.

        The RTL2832U puts a large fixed spike at exactly the tuned centre.
        Left alone it appears once per step -- a comb of ~900 fake carriers
        across a full-range sweep, evenly spaced, which is exactly what a
        real signal is not.
        """
        n = shifted.size
        c = n // 2
        k = config.DC_SPIKE_BINS
        lo, hi = c - k, c + k
        if lo - 1 < 0 or hi + 1 >= n:
            return
        left, right = shifted[lo - 1], shifted[hi + 1]
        shifted[lo:hi + 1] = np.linspace(left, right, hi - lo + 1)
