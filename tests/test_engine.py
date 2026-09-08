"""Capture engine and per-bin statistics, against the mock device's known truth."""
from __future__ import annotations

import numpy as np
import pytest

from aether.sdr.accumulate import SpectrumAccumulator, to_db
from aether.sdr.device import MockDevice
from aether.sdr.engine import StopPolicy, SweepEngine
from aether.sdr.sweep import SweepPlan


def run_scan(tones, f_start, f_stop, sweeps=4, bin_hz=10e3, dwell=0.03, seed=0):
    dev = MockDevice(tones=tones, seed=seed)
    plan = SweepPlan.create(f_start, f_stop, bin_hz, dwell_s=dwell)
    engine = SweepEngine(dev, plan, gain=30.0)
    acc = engine.run(StopPolicy("sweeps", max_sweeps=sweeps))
    return plan, acc


def bin_of(plan, hz):
    return int(np.argmin(np.abs(plan.freqs_hz - hz)))


def test_tones_are_recovered_at_the_right_frequency():
    """Every tone must land within half a bin of where it was injected."""
    tones = [(89_700_000, 0.30, 1.0), (101_100_000, 0.25, 1.0),
             (107_900_000, 0.10, 1.0)]
    plan, acc = run_scan(tones, 88e6, 108e6)
    avg = acc.avg_db
    for hz, _amp, _duty in tones:
        i = bin_of(plan, hz)
        window = avg[max(0, i - 3):i + 4]
        peak_offset = int(np.argmax(window)) - min(i, 3)
        assert abs(peak_offset) <= 1, "%0.3f MHz peak is off by %d bins" % (hz / 1e6, peak_offset)


def test_tone_amplitude_matches_dbfs_within_scalloping_loss():
    """A full-scale sinusoid should read 0 dBFS. A Hann window costs up to
    ~1.4 dB when the tone sits between bins, so allow that much."""
    amp = 0.25
    plan, acc = run_scan([(100_000_000, amp, 1.0)], 99e6, 101e6)
    i = bin_of(plan, 100_000_000)
    measured = np.nanmax(acc.avg_db[i - 3:i + 4])
    expected = 20 * np.log10(amp)
    assert expected - 1.5 <= measured <= expected + 0.3


def test_peak_hold_catches_a_bursty_signal_the_average_misses():
    """The central claim of the four-trace design."""
    plan, acc = run_scan([(462_562_500, 0.25, 0.10)], 462e6, 463e6,
                         sweeps=8, dwell=0.05, seed=7)
    i = bin_of(plan, 462_562_500)
    floor = np.nanpercentile(acc.avg_db, 10)

    peak, avg, mn = acc.peak_hold_db[i], acc.avg_db[i], acc.min_db[i]
    assert peak > avg + 5, "peak hold should far exceed avg for a 10% duty tone"
    assert mn < floor + 3, "min should sit near the floor when the tone is absent"
    assert peak == pytest.approx(20 * np.log10(0.25), abs=2.0), \
        "peak hold should recover the true tone amplitude"


def test_trace_ordering_holds_everywhere():
    plan, acc = run_scan(MockDevice.DEFAULT_TONES, 88e6, 120e6, sweeps=3)
    valid = acc.valid
    assert np.all(acc.min_db[valid] <= acc.avg_db[valid] + 1e-6)
    assert np.all(acc.avg_db[valid] <= acc.max_db[valid] + 1e-6)
    assert np.all(acc.max_db[valid] <= acc.peak_hold_db[valid] + 1e-6)


def test_all_bins_receive_the_same_number_of_sweeps():
    plan, acc = run_scan([], 88e6, 108e6, sweeps=3)
    assert acc.count.min() == 3
    assert acc.count.max() == 3
    assert acc.sweeps_completed == 3


def test_dc_spike_is_removed():
    """The mock injects a DC offset at every step centre, as real hardware
    does. Without removal it shows up as a comb of fake carriers."""
    plan, acc = run_scan([], 88e6, 108e6, sweeps=2)
    avg = acc.avg_db[acc.valid]
    floor = np.percentile(avg, 50)
    # With the spike left in, each step centre would sit far above the floor.
    assert avg.max() < floor + 20, "something spike-like survived DC removal"


def test_stop_event_halts_promptly():
    dev = MockDevice()
    plan = SweepPlan.create(24e6, 1766e6, 20e3, dwell_s=0.05)
    engine = SweepEngine(dev, plan, gain=30.0)
    engine.stop()                       # pre-set: run should exit immediately
    acc = engine.run(StopPolicy("indefinite"))
    assert acc.sweeps_completed == 0
    assert engine.elapsed() < 5.0


class TestAccumulator:
    def test_averaging_happens_in_the_power_domain(self):
        """Averaging in dB instead would give -15 dB here, not -13."""
        acc = SpectrumAccumulator(1)
        acc.add_segment(0, 1, np.array([1.0]))       # 0 dB
        acc.add_segment(0, 1, np.array([0.001]))     # -30 dB
        expected = 10 * np.log10((1.0 + 0.001) / 2)
        assert acc.avg_db[0] == pytest.approx(expected, abs=1e-6)
        assert acc.avg_db[0] > -3.5                   # not the -15 dB mean

    def test_unmeasured_bins_are_nan_not_zero(self):
        acc = SpectrumAccumulator(4)
        acc.add_segment(0, 2, np.array([1.0, 1.0]))
        assert np.isfinite(acc.avg_db[0])
        assert np.isnan(acc.avg_db[3]), "an unscanned bin must not read as 0 dB"
        assert np.isnan(acc.min_db[3])

    def test_segment_length_mismatch_is_rejected(self):
        acc = SpectrumAccumulator(10)
        with pytest.raises(ValueError, match="bins"):
            acc.add_segment(0, 5, np.ones(3))

    def test_to_db_floors_zero_instead_of_returning_inf(self):
        assert np.isfinite(to_db(np.array([0.0]))[0])


class TestOverloadDetection:
    """A clipped capture is silently wrong: the front end compresses, strong
    signals read low, and intermod invents peaks. It must be reported."""

    def test_clean_capture_reports_no_overload(self):
        dev = MockDevice(tones=[(100e6, 0.2, 1.0)], noise_amp=0.01)
        plan = SweepPlan.create(99e6, 101e6, 10e3, dwell_s=0.02)
        engine = SweepEngine(dev, plan, gain=30.0)
        engine.run(StopPolicy("sweeps", max_sweeps=2))
        assert engine.total_samples > 0
        assert not engine.is_overloaded
        assert engine.overload_warning() is None

    def test_railed_capture_is_detected(self):
        """A tone well past full scale rails the simulated ADC.

        The fraction stays well below 1.0 because the tone only falls inside
        some of the sweep's steps -- the rest see nothing but noise. What
        matters is that it lands far above the warning threshold.
        """
        dev = MockDevice(tones=[(100e6, 3.0, 1.0)], noise_amp=0.01)
        plan = SweepPlan.create(99e6, 101e6, 10e3, dwell_s=0.02)
        engine = SweepEngine(dev, plan, gain=30.0)
        engine.run(StopPolicy("sweeps", max_sweeps=2))
        assert engine.is_overloaded
        assert engine.clip_fraction > 0.2
        warning = engine.overload_warning()
        assert warning and "overloaded" in warning.lower()

    def test_clip_fraction_is_zero_before_any_capture(self):
        dev = MockDevice()
        plan = SweepPlan.create(99e6, 101e6, 10e3)
        assert SweepEngine(dev, plan).clip_fraction == 0.0
