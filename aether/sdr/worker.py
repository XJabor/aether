"""Qt wrapper around :class:`SweepEngine`.

The engine owns the device handle and runs entirely on its own thread; the
GUI thread never touches librtlsdr. Communication is one-way via signals.

Progress is deliberately *not* signalled per step. A full-range sweep is
~900 steps, and emitting a queued signal for each one floods the event loop
faster than it can drain. The worker instead throttles progress to ~15 Hz
and the window repaints on its own timer.
"""
from __future__ import annotations

import time

from PySide6.QtCore import QObject, Signal, Slot

from .accumulate import SpectrumAccumulator
from .device import SdrDevice, SdrError
from .engine import StopPolicy, SweepEngine, SweepProgress
from .sweep import SweepPlan

PROGRESS_INTERVAL_S = 1.0 / 15.0


class ScanWorker(QObject):
    """Runs one scan. Create it, move it to a QThread, then call start()."""

    started = Signal()
    progress = Signal(object)        # SweepProgress
    sweepCompleted = Signal(int)     # number of completed sweeps
    dataChanged = Signal()           # traces updated; repaint when convenient
    failed = Signal(str)
    finished = Signal()

    def __init__(
        self,
        device: SdrDevice,
        plan: SweepPlan,
        policy: StopPolicy,
        gain: float | str = "auto",
        ppm: int = 0,
    ) -> None:
        super().__init__()
        self.device = device
        self.plan = plan
        self.policy = policy
        self.engine = SweepEngine(device, plan, gain=gain, ppm=ppm)
        self._last_progress = 0.0
        self._dirty = False

    @property
    def accumulator(self) -> SpectrumAccumulator:
        return self.engine.acc

    def elapsed(self) -> float:
        return self.engine.elapsed()

    # -- control ---------------------------------------------------------

    @Slot()
    def stop(self) -> None:
        """Safe to call from the GUI thread: only sets an Event."""
        self.engine.stop()

    @Slot()
    def run(self) -> None:
        self.started.emit()
        try:
            self.engine.run(
                self.policy,
                on_segment=self._on_segment,
                on_sweep=self._on_sweep,
                on_progress=self._on_progress,
            )
        except (SdrError, RuntimeError, ValueError) as exc:
            self.failed.emit(str(exc))
        except Exception as exc:                      # never kill the thread silently
            self.failed.emit("Unexpected error: %r" % (exc,))
        finally:
            try:
                self.device.close()
            except Exception:
                pass
            if self._dirty:
                self.dataChanged.emit()
            self.finished.emit()

    # -- engine callbacks (all on the worker thread) ---------------------

    def _on_segment(self, lo: int, hi: int) -> None:
        self._dirty = True

    def _on_sweep(self, n: int) -> None:
        self.dataChanged.emit()
        self._dirty = False
        self.sweepCompleted.emit(n)

    def _on_progress(self, p: SweepProgress) -> None:
        now = time.monotonic()
        if now - self._last_progress < PROGRESS_INTERVAL_S:
            return
        self._last_progress = now
        self.progress.emit(p)
        if self._dirty:
            self.dataChanged.emit()
            self._dirty = False
