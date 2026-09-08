"""Running per-bin statistics across sweeps.

Four traces, each answering a different question:

  avg        the baseline -- what is normally here
  min        an estimate of the true noise floor
  max        strongest value any single sweep averaged to
  peak_hold  strongest value any single FFT *frame* saw

The avg/peak_hold split is the one that matters in practice. A handheld
keying up for two seconds inside a ten minute scan barely moves ``avg``,
shows weakly in ``max`` (it is diluted by that sweep's own averaging) and
stands out clearly in ``peak_hold``.

Averaging happens in the linear power domain and is converted to dB only
on read. Averaging dB values directly is a different (and wrong) operation
that biases the noise floor low.
"""
from __future__ import annotations

import threading

import numpy as np

# Floor for the log conversion, so an empty or zeroed bin cannot produce
# -inf and poison the plot autoscale.
_TINY = 1e-30


def to_db(power_linear: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(power_linear, _TINY))


class SpectrumAccumulator:
    """Accumulates sweep segments into per-bin statistics."""

    def __init__(self, n_bins: int) -> None:
        self.n_bins = n_bins
        self._power_sum = np.zeros(n_bins, dtype=np.float64)
        self.count = np.zeros(n_bins, dtype=np.int32)
        self._min_db = np.full(n_bins, np.inf, dtype=np.float32)
        self._max_db = np.full(n_bins, -np.inf, dtype=np.float32)
        self._peak_db = np.full(n_bins, -np.inf, dtype=np.float32)
        self.sweeps_completed = 0

        # The capture thread writes while the GUI thread reads for plotting.
        # Without this, a snapshot taken mid-write can divide a bin's power
        # sum by a count that has already been incremented, putting a single
        # visibly wrong point on the graph.
        self.lock = threading.Lock()

    # -- writing ---------------------------------------------------------

    def add_segment(
        self,
        lo: int,
        hi: int,
        avg_power_lin: np.ndarray,
        frame_max_lin: np.ndarray | None = None,
    ) -> None:
        """Fold one step's worth of a sweep into the running statistics.

        ``avg_power_lin`` is that step's Welch-averaged linear power.
        ``frame_max_lin`` is the per-bin max over individual FFT frames.
        """
        if hi - lo != avg_power_lin.size:
            raise ValueError(
                f"segment [{lo}:{hi}] is {hi-lo} bins but got "
                f"{avg_power_lin.size} values"
            )

        seg_db = to_db(avg_power_lin).astype(np.float32)
        peak_src = seg_db if frame_max_lin is None else to_db(frame_max_lin).astype(np.float32)

        with self.lock:
            self._power_sum[lo:hi] += avg_power_lin
            self.count[lo:hi] += 1
            np.minimum(self._min_db[lo:hi], seg_db, out=self._min_db[lo:hi])
            np.maximum(self._max_db[lo:hi], seg_db, out=self._max_db[lo:hi])
            np.maximum(self._peak_db[lo:hi], peak_src, out=self._peak_db[lo:hi])

    def mark_sweep_complete(self) -> None:
        self.sweeps_completed += 1

    # -- reading ---------------------------------------------------------

    @property
    def valid(self) -> np.ndarray:
        """Bins that have received at least one sample."""
        return self.count > 0

    @property
    def avg_db(self) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = self._power_sum / np.maximum(self.count, 1)
        out = to_db(mean).astype(np.float32)
        out[self.count == 0] = np.nan
        return out

    @property
    def min_db(self) -> np.ndarray:
        return self._finite(self._min_db)

    @property
    def max_db(self) -> np.ndarray:
        return self._finite(self._max_db)

    @property
    def peak_hold_db(self) -> np.ndarray:
        return self._finite(self._peak_db)

    def _finite(self, arr: np.ndarray) -> np.ndarray:
        out = arr.copy()
        out[~np.isfinite(out)] = np.nan
        return out

    def as_dict(self) -> dict[str, np.ndarray]:
        """Consistent snapshot of all traces, safe to take mid-scan."""
        with self.lock:
            return {
                "avg_db": self.avg_db,
                "min_db": self.min_db,
                "max_db": self.max_db,
                "peak_hold_db": self.peak_hold_db,
                "count": self.count.copy(),
            }

    def reset(self) -> None:
        self._power_sum.fill(0.0)
        self.count.fill(0)
        self._min_db.fill(np.inf)
        self._max_db.fill(-np.inf)
        self._peak_db.fill(-np.inf)
        self.sweeps_completed = 0
