"""Sweep geometry. If this is wrong, every frequency the app reports is wrong."""
from __future__ import annotations

import numpy as np
import pytest

from aether import config
from aether.sdr.sweep import SweepPlan, next_pow2

RANGES = [
    (88e6, 108e6, 10e3),          # FM broadcast
    (24e6, 1766e6, 20e3),         # full tuner range
    (462e6, 468e6, 1e3),          # narrow, fine resolution
    (144e6, 148e6, 5e3),          # 2 m
]


@pytest.mark.parametrize("f_start,f_stop,bin_hz", RANGES)
def test_every_bin_covered_exactly_once(f_start, f_stop, bin_hz):
    """No gaps and no overlaps -- an overlap would double-count a bin into
    the average, a gap would leave a hole the plot draws straight through."""
    plan = SweepPlan.create(f_start, f_stop, bin_hz)
    coverage = np.zeros(plan.n_bins, dtype=np.int32)
    for step in plan.steps:
        coverage[step.glob_lo:step.glob_hi] += 1
    assert coverage.min() == 1, "gap in coverage"
    assert coverage.max() == 1, "overlapping steps"


@pytest.mark.parametrize("f_start,f_stop,bin_hz", RANGES)
def test_step_centres_land_on_the_bin_grid(f_start, f_stop, bin_hz):
    """The whole stitching scheme depends on this: if a centre is off-grid,
    that step's bins do not line up and the error accumulates."""
    plan = SweepPlan.create(f_start, f_stop, bin_hz)
    for step in plan.steps:
        offset_bins = (step.center_hz - plan.f_start_hz) / plan.bin_hz
        assert abs(offset_bins - round(offset_bins)) < 1e-6


@pytest.mark.parametrize("f_start,f_stop,bin_hz", RANGES)
def test_slice_widths_match(f_start, f_stop, bin_hz):
    plan = SweepPlan.create(f_start, f_stop, bin_hz)
    for step in plan.steps:
        assert step.fft_hi - step.fft_lo == step.glob_hi - step.glob_lo
        assert 0 <= step.fft_lo < step.fft_hi <= plan.n_fft
        assert 0 <= step.glob_lo < step.glob_hi <= plan.n_bins


def test_frequency_axis_is_uniform_and_spans_the_request():
    plan = SweepPlan.create(88e6, 108e6, 10e3)
    f = plan.freqs_hz
    assert f[0] == pytest.approx(88e6)
    assert f[-1] <= 108e6
    assert np.allclose(np.diff(f), plan.bin_hz)


def test_fft_size_is_a_power_of_two():
    plan = SweepPlan.create(88e6, 108e6, 10e3)
    assert plan.n_fft & (plan.n_fft - 1) == 0
    assert plan.bin_hz == pytest.approx(plan.sample_rate / plan.n_fft)


def test_next_pow2():
    assert next_pow2(1) == 1
    assert next_pow2(100) == 128
    assert next_pow2(256) == 256
    assert next_pow2(257) == 512


class TestValidation:
    def test_rejects_inverted_range(self):
        with pytest.raises(ValueError, match="above"):
            SweepPlan.create(108e6, 88e6, 10e3)

    def test_rejects_range_outside_the_tuner(self):
        with pytest.raises(ValueError, match="outside"):
            SweepPlan.create(1e6, 10e6, 10e3)

    def test_direct_sampling_allows_hf(self):
        plan = SweepPlan.create(1e6, 10e6, 5e3, direct_sampling=True)
        assert plan.direct_sampling
        assert plan.n_steps > 0

    def test_direct_sampling_still_has_an_upper_limit(self):
        with pytest.raises(ValueError, match="outside"):
            SweepPlan.create(1e6, 40e6, 5e3, direct_sampling=True)

    def test_rejects_absurd_bin_counts(self):
        """A 1 Hz sweep of the whole tuner range would be 1.7 billion bins."""
        with pytest.raises(ValueError, match="bins"):
            SweepPlan.create(config.TUNER_MIN_HZ, config.TUNER_MAX_HZ, 1.0)


def test_sweep_time_estimate_scales_with_steps():
    narrow = SweepPlan.create(88e6, 108e6, 10e3)
    wide = SweepPlan.create(24e6, 1766e6, 10e3)
    assert wide.n_steps > narrow.n_steps
    assert wide.estimated_sweep_seconds() > narrow.estimated_sweep_seconds()
