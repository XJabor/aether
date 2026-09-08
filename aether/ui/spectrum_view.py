"""The spectrum plot: power (dB) against frequency (MHz), SDR# style."""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QSizePolicy, QVBoxLayout, QWidget

from ..storage.session import SpectrumData

# Trace name -> (label, colour, width, z-order)
TRACES: dict[str, tuple[str, str, float, int]] = {
    "min_db":       ("Min",       "#5c6b7a", 1.0, 0),
    "avg_db":       ("Average",   "#4ec9f0", 1.6, 3),
    "max_db":       ("Max",       "#e8734a", 1.0, 1),
    "peak_hold_db": ("Peak hold", "#f2c94c", 1.0, 2),
}
REFERENCE_COLOR = "#b47cff"
DELTA_COLOR = "#7ee081"

BACKGROUND = "#0f1419"
FOREGROUND = "#c8d0d8"


class SpectrumView(QWidget):
    """Live spectrum with optional reference overlay and delta panel."""

    regionChanged = Signal(float, float)   # lo_hz, hi_hz

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        pg.setConfigOptions(antialias=False, background=BACKGROUND, foreground=FOREGROUND)

        # A plain QWidget defaults to a Preferred size policy, which makes
        # QMainWindow hand it only its size hint and give every spare pixel
        # to the docks -- the plot ends up a narrow column no matter how wide
        # the window is. The spectrum is the point of the app; it expands.
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumWidth(420)

        self._freqs_mhz: np.ndarray | None = None
        self._data: SpectrumData | None = None
        self._reference: SpectrumData | None = None
        self._cal_offset_db = 0.0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._layout_widget = pg.GraphicsLayoutWidget()
        layout.addWidget(self._layout_widget)

        self._build_main_plot()
        self._build_delta_plot()
        self._build_crosshair()
        self._build_region()

    # -- construction ----------------------------------------------------

    def _build_main_plot(self) -> None:
        self.plot = self._layout_widget.addPlot(row=0, col=0)
        self.plot.setLabel("left", "Power", units="dBFS")
        self.plot.setLabel("bottom", "Frequency", units="Hz")
        self.plot.showGrid(x=True, y=True, alpha=0.18)
        self.plot.setClipToView(True)
        self.plot.setMouseEnabled(x=True, y=True)
        self.legend = self.plot.addLegend(offset=(-12, 12), labelTextColor=FOREGROUND)

        self.curves: dict[str, pg.PlotDataItem] = {}
        for name, (label, color, width, z) in TRACES.items():
            pen = pg.mkPen(QColor(color), width=width)
            curve = self.plot.plot([], [], pen=pen, name=label)
            # 'peak' downsampling keeps narrow carriers visible when zoomed
            # out. With the default mean mode a 10 kHz-wide signal inside a
            # 1.7 GHz view averages into the noise and disappears entirely.
            curve.setDownsampling(auto=True, method="peak")
            curve.setClipToView(True)
            curve.setZValue(z)
            self.curves[name] = curve

        ref_pen = pg.mkPen(QColor(REFERENCE_COLOR), width=1.2, style=Qt.DashLine)
        self.reference_curve = self.plot.plot([], [], pen=ref_pen, name="Reference")
        self.reference_curve.setDownsampling(auto=True, method="peak")
        self.reference_curve.setClipToView(True)
        self.reference_curve.setZValue(4)
        self.reference_curve.setVisible(False)

    def _build_delta_plot(self) -> None:
        self.delta_plot = self._layout_widget.addPlot(row=1, col=0)
        self.delta_plot.setLabel("left", "Delta", units="dB")
        self.delta_plot.setLabel("bottom", "Frequency", units="Hz")
        self.delta_plot.showGrid(x=True, y=True, alpha=0.18)
        self.delta_plot.setXLink(self.plot)          # pan/zoom together
        self.delta_plot.setClipToView(True)
        self.delta_plot.setMaximumHeight(180)

        self.delta_curve = self.delta_plot.plot(
            [], [], pen=pg.mkPen(QColor(DELTA_COLOR), width=1.2)
        )
        self.delta_curve.setDownsampling(auto=True, method="peak")
        self.delta_curve.setClipToView(True)

        zero = pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen("#55606b", width=1))
        self.delta_plot.addItem(zero)
        self._layout_widget.ci.layout.setRowStretchFactor(0, 3)
        self._layout_widget.ci.layout.setRowStretchFactor(1, 1)
        self._delta_visible = True
        self.set_delta_visible(False)

    def _build_crosshair(self) -> None:
        pen = pg.mkPen("#6b7785", width=1, style=Qt.DotLine)
        self._vline = pg.InfiniteLine(angle=90, movable=False, pen=pen)
        self._hline = pg.InfiniteLine(angle=0, movable=False, pen=pen)
        for item in (self._vline, self._hline):
            item.setZValue(10)
            self.plot.addItem(item, ignoreBounds=True)

        self._readout = pg.TextItem(color=FOREGROUND, anchor=(0, 1))
        self._readout.setZValue(11)
        self.plot.addItem(self._readout, ignoreBounds=True)

        self.plot.scene().sigMouseMoved.connect(self._on_mouse_moved)

    def _build_region(self) -> None:
        """Draggable band selector, used to scope AI questions and zoom."""
        self.region = pg.LinearRegionItem(
            brush=pg.mkBrush(78, 201, 240, 28),
            hoverBrush=pg.mkBrush(78, 201, 240, 45),
            pen=pg.mkPen("#4ec9f0", width=1),
        )
        self.region.setZValue(-10)
        self.region.sigRegionChangeFinished.connect(self._on_region_changed)
        # ignoreBounds keeps the selector out of autoRange(): otherwise an
        # unpositioned region at 0 Hz drags the x-axis down to DC and
        # squashes the actual scan into a corner of the plot.
        self.plot.addItem(self.region, ignoreBounds=True)
        self.region.setVisible(False)
        self._region_active = False
        self._region_placed = False

    # -- data ------------------------------------------------------------

    def set_data(self, data: SpectrumData | None, autorange: bool = False) -> None:
        was_empty = self._data is None or self._data.freqs_hz.size == 0
        self._data = data
        if data is None or data.freqs_hz.size == 0:
            for c in self.curves.values():
                c.setData([], [])
            self.delta_curve.setData([], [])
            self._region_placed = False
            return

        freqs = data.freqs_hz
        for name, curve in self.curves.items():
            trace = np.asarray(data.trace(name), dtype=np.float64)
            if self._cal_offset_db:
                trace = trace + self._cal_offset_db
            # connect='finite' breaks the line at unmeasured bins instead of
            # drawing a straight segment across a gap that was never sampled.
            curve.setData(freqs, trace, connect="finite")

        self._update_delta()
        if was_empty:
            self._place_region()
        if autorange:
            self.autoscale()

    def set_reference(self, data: SpectrumData | None) -> None:
        self._reference = data
        if data is None:
            self.reference_curve.setVisible(False)
            self.set_delta_visible(False)
            return
        trace = np.asarray(data.avg_db, dtype=np.float64)
        self.reference_curve.setData(data.freqs_hz, trace, connect="finite")
        self.reference_curve.setVisible(True)
        self._update_delta()
        self.set_delta_visible(True)

    def _update_delta(self) -> None:
        if self._data is None or self._reference is None:
            self.delta_curve.setData([], [])
            return
        from ..analysis.compare import align_and_diff

        freqs, delta = align_and_diff(self._data, self._reference)
        self.delta_curve.setData(freqs, delta, connect="finite")

    def set_cal_offset(self, offset_db: float) -> None:
        self._cal_offset_db = float(offset_db)
        label = "Power" if not offset_db else "Power (offset %+.1f dB)" % offset_db
        self.plot.setLabel("left", label, units="dBFS")
        self.set_data(self._data)

    # -- appearance ------------------------------------------------------

    def set_trace_visible(self, name: str, visible: bool) -> None:
        if name in self.curves:
            self.curves[name].setVisible(visible)

    def set_delta_visible(self, visible: bool) -> None:
        """Add or remove the delta panel from the layout.

        Hiding a PlotItem with setVisible() leaves its row occupying space,
        so the main plot would keep only two thirds of the window with a
        dead band underneath. It has to leave the layout entirely.
        """
        if visible == self._delta_visible:
            return
        self._delta_visible = visible
        if visible:
            self._layout_widget.addItem(self.delta_plot, row=1, col=0)
            self.delta_plot.setXLink(self.plot)
        else:
            self._layout_widget.removeItem(self.delta_plot)

    def autoscale(self) -> None:
        self.plot.enableAutoRange()
        self.plot.autoRange()

    def set_x_range_hz(self, lo: float, hi: float) -> None:
        self.plot.setXRange(lo, hi, padding=0.01)

    # -- region ----------------------------------------------------------

    def enable_region(self, enabled: bool) -> None:
        self._region_active = enabled
        self.region.setVisible(enabled)
        if enabled:
            self._place_region()

    def _place_region(self) -> None:
        """Centre the selector on the current span.

        Deferred until data exists: enabling the selector before a scan has
        started would otherwise leave it parked at 0 Hz.
        """
        if not self._region_active or self._region_placed:
            return
        if self._data is None or self._data.freqs_hz.size == 0:
            return
        f = self._data.freqs_hz
        span = float(f[-1] - f[0])
        self.region.setRegion((f[0] + span * 0.4, f[0] + span * 0.6))
        self._region_placed = True
        self._on_region_changed()

    def selected_range_hz(self) -> tuple[float, float] | None:
        if not self._region_active:
            return None
        lo, hi = self.region.getRegion()
        return (float(lo), float(hi)) if hi > lo else None

    def _on_region_changed(self) -> None:
        rng = self.selected_range_hz()
        if rng:
            self.regionChanged.emit(*rng)

    # -- crosshair -------------------------------------------------------

    def _on_mouse_moved(self, pos) -> None:
        vb = self.plot.vb
        if not self.plot.sceneBoundingRect().contains(pos):
            self._readout.setText("")
            return
        pt = vb.mapSceneToView(pos)
        self._vline.setPos(pt.x())
        self._hline.setPos(pt.y())

        text = "%.4f MHz   %.1f dB" % (pt.x() / 1e6, pt.y())
        if self._data is not None and self._data.freqs_hz.size:
            f = self._data.freqs_hz
            i = int(np.clip(np.searchsorted(f, pt.x()), 0, f.size - 1))
            v = self._data.avg_db[i] + self._cal_offset_db
            if np.isfinite(v):
                text += "\navg %.1f dBFS @ %.4f MHz" % (v, f[i] / 1e6)
        self._readout.setText(text)
        self._readout.setPos(pt.x(), pt.y())
