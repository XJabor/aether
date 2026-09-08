"""Scan setup: range, resolution, gain, and stop conditions."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout,
    QLabel, QPushButton, QRadioButton, QSizePolicy, QSpinBox, QVBoxLayout, QWidget,
)

from .. import config
from ..sdr.engine import StopPolicy
from ..sdr.sweep import SweepPlan

def shrinkable(combo: QComboBox) -> QComboBox:
    """Let a combo shrink below its longest item.

    A QComboBox reports a minimum width wide enough for its widest entry.
    With long entries ("Full VHF/UHF sweep (24-1766 MHz)") that minimum
    propagates all the way up and forces the dock -- and therefore the whole
    window -- wider than the spectrum plot deserves.
    """
    combo.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
    combo.setMinimumContentsLength(10)
    combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
    return combo


PRESETS: list[tuple[str, float, float, float]] = [
    ("FM broadcast",        88.0,   108.0,  10.0),
    ("Airband",             118.0,  137.0,  5.0),
    ("2 m amateur",         144.0,  148.0,  5.0),
    ("VHF public safety",   150.0,  174.0,  5.0),
    ("70 cm amateur",       420.0,  450.0,  5.0),
    ("UHF business/GMRS",   450.0,  470.0,  2.5),
    ("ISM 900 MHz",         902.0,  928.0,  10.0),
    ("ADS-B",               1085.0, 1095.0, 5.0),
    ("Full VHF/UHF sweep",  24.0,   1766.0, 20.0),
    ("HF (direct sampling)", 0.5,   28.8,   1.0),
]


class ControlPanel(QWidget):
    startRequested = Signal()
    stopRequested = Signal()
    planChanged = Signal(object)      # SweepPlan or None

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._gains: list[float] = []
        self._scanning = False
        self._build()
        self._wire()
        self._refresh_plan()

    # -- construction ----------------------------------------------------

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)

        # Range -----------------------------------------------------------
        range_box = QGroupBox("Frequency range")
        form = QFormLayout(range_box)

        self.preset = shrinkable(QComboBox())
        self.preset.addItem("Custom")
        for name, lo, hi, _ in PRESETS:
            self.preset.addItem("%s  (%g-%g MHz)" % (name, lo, hi))
        form.addRow("Preset", self.preset)

        self.f_start = self._mhz_spin(88.0)
        self.f_stop = self._mhz_spin(108.0)
        form.addRow("Start (MHz)", self.f_start)
        form.addRow("Stop (MHz)", self.f_stop)

        self.bin_khz = QDoubleSpinBox()
        self.bin_khz.setRange(0.05, 1000.0)
        self.bin_khz.setDecimals(2)
        self.bin_khz.setValue(10.0)
        self.bin_khz.setSuffix(" kHz")
        form.addRow("Resolution", self.bin_khz)

        self.direct_sampling = QCheckBox("HF direct sampling")
        self.direct_sampling.setToolTip(
            "Bypasses the tuner and samples the ADC directly, which is how "
            "the V3 receives HF.\nCleanest below 14.4 MHz; above that, "
            "images fold back and need interpretation."
        )
        form.addRow("", self.direct_sampling)
        root.addWidget(range_box)

        # Receiver --------------------------------------------------------
        rx_box = QGroupBox("Receiver")
        rx_form = QFormLayout(rx_box)

        self.gain = shrinkable(QComboBox())
        self.gain.addItem("Auto (AGC)", "auto")
        rx_form.addRow("Gain", self.gain)

        self.agc_warning = QLabel(
            "AGC drifts between sweeps, so this baseline will not be "
            "comparable to other scans. Pick a fixed gain for survey work."
        )
        self.agc_warning.setWordWrap(True)
        self.agc_warning.setMinimumWidth(1)
        self.agc_warning.setStyleSheet("color: #e8b34a; font-size: 11px;")
        rx_form.addRow("", self.agc_warning)

        self.sample_rate = shrinkable(QComboBox())
        for rate in config.SAMPLE_RATES:
            label = "%.3f MS/s" % (rate / 1e6)
            if rate == config.DEFAULT_SAMPLE_RATE:
                label += "  (recommended)"
            elif rate > 2_500_000:
                label += "  (may drop samples)"
            self.sample_rate.addItem(label, rate)
        rx_form.addRow("Sample rate", self.sample_rate)

        self.ppm = QSpinBox()
        self.ppm.setRange(-200, 200)
        self.ppm.setSuffix(" ppm")
        rx_form.addRow("Correction", self.ppm)

        self.dwell = QDoubleSpinBox()
        self.dwell.setRange(0.005, 5.0)
        self.dwell.setDecimals(3)
        self.dwell.setSingleStep(0.01)
        self.dwell.setValue(0.05)
        self.dwell.setSuffix(" s")
        self.dwell.setToolTip(
            "Capture time at each tuner step. Longer means a smoother "
            "average and better weak-signal sensitivity, but slower sweeps."
        )
        rx_form.addRow("Dwell/step", self.dwell)

        self.bias_tee = QCheckBox("Bias tee (4.5 V)")
        self.bias_tee.setToolTip(
            "Powers an external LNA through the coax.\n"
            "Leave off unless you have one connected."
        )
        rx_form.addRow("", self.bias_tee)
        root.addWidget(rx_box)

        # Duration --------------------------------------------------------
        dur_box = QGroupBox("Record for")
        dur = QVBoxLayout(dur_box)

        self.mode_indefinite = QRadioButton("Until I stop it")
        self.mode_indefinite.setChecked(True)
        dur.addWidget(self.mode_indefinite)

        row = QHBoxLayout()
        self.mode_duration = QRadioButton("Time")
        self.duration_min = QDoubleSpinBox()
        self.duration_min.setRange(0.1, 1440.0)
        self.duration_min.setDecimals(1)
        self.duration_min.setValue(5.0)
        self.duration_min.setSuffix(" min")
        row.addWidget(self.mode_duration)
        row.addWidget(self.duration_min)
        dur.addLayout(row)

        row2 = QHBoxLayout()
        self.mode_sweeps = QRadioButton("Sweeps")
        self.sweep_count = QSpinBox()
        self.sweep_count.setRange(1, 100000)
        self.sweep_count.setValue(10)
        row2.addWidget(self.mode_sweeps)
        row2.addWidget(self.sweep_count)
        dur.addLayout(row2)
        root.addWidget(dur_box)

        # Estimate + actions ----------------------------------------------
        self.estimate = QLabel()
        self.estimate.setWordWrap(True)
        self.estimate.setMinimumWidth(1)
        self.estimate.setTextFormat(Qt.RichText)
        self.estimate.setStyleSheet("font-size: 11px; padding: 4px;")
        root.addWidget(self.estimate)

        buttons = QHBoxLayout()
        self.start_button = QPushButton("Start")
        self.start_button.setMinimumHeight(34)
        self.stop_button = QPushButton("Stop")
        self.stop_button.setMinimumHeight(34)
        self.stop_button.setEnabled(False)
        buttons.addWidget(self.start_button)
        buttons.addWidget(self.stop_button)
        root.addLayout(buttons)
        root.addStretch(1)

    @staticmethod
    def _mhz_spin(value: float) -> QDoubleSpinBox:
        s = QDoubleSpinBox()
        s.setRange(0.1, 2000.0)
        s.setDecimals(4)
        s.setSingleStep(1.0)
        s.setValue(value)
        return s

    def _wire(self) -> None:
        self.preset.currentIndexChanged.connect(self._apply_preset)
        for w in (self.f_start, self.f_stop, self.bin_khz, self.dwell):
            w.valueChanged.connect(self._refresh_plan)
        self.sample_rate.currentIndexChanged.connect(self._refresh_plan)
        self.direct_sampling.toggled.connect(self._refresh_plan)
        self.gain.currentIndexChanged.connect(self._on_gain_changed)
        self.start_button.clicked.connect(self.startRequested)
        self.stop_button.clicked.connect(self.stopRequested)
        self._on_gain_changed()

    # -- state -----------------------------------------------------------

    def set_available_gains(self, gains: list[float]) -> None:
        self._gains = list(gains)
        current = self.gain.currentData()
        self.gain.blockSignals(True)
        self.gain.clear()
        self.gain.addItem("Auto (AGC)", "auto")
        for g in gains:
            self.gain.addItem("%.1f dB" % g, g)
        # Default to a mid-scale gain: high enough for weak signals, well
        # short of the clipping that maximum gain causes on strong ones.
        if gains:
            idx = self.gain.findData(gains[len(gains) // 2])
            self.gain.setCurrentIndex(max(idx, 0))
        if current is not None:
            found = self.gain.findData(current)
            if found >= 0:
                self.gain.setCurrentIndex(found)
        self.gain.blockSignals(False)
        self._on_gain_changed()

    def set_scanning(self, scanning: bool) -> None:
        self._scanning = scanning
        self.start_button.setEnabled(not scanning)
        self.stop_button.setEnabled(scanning)
        for w in (self.f_start, self.f_stop, self.bin_khz, self.gain,
                  self.sample_rate, self.ppm, self.dwell, self.preset,
                  self.direct_sampling, self.bias_tee):
            w.setEnabled(not scanning)

    def _on_gain_changed(self) -> None:
        self.agc_warning.setVisible(self.gain.currentData() == "auto")

    def _apply_preset(self, index: int) -> None:
        if index <= 0:
            return
        name, lo, hi, rbw = PRESETS[index - 1]
        for w in (self.f_start, self.f_stop, self.bin_khz):
            w.blockSignals(True)
        self.f_start.setValue(lo)
        self.f_stop.setValue(hi)
        self.bin_khz.setValue(rbw)
        for w in (self.f_start, self.f_stop, self.bin_khz):
            w.blockSignals(False)
        self.direct_sampling.setChecked(hi <= config.DIRECT_MAX_HZ / 1e6)
        self._refresh_plan()

    # -- plan ------------------------------------------------------------

    def current_plan(self) -> SweepPlan | None:
        try:
            return SweepPlan.create(
                f_start_hz=self.f_start.value() * 1e6,
                f_stop_hz=self.f_stop.value() * 1e6,
                bin_hz=self.bin_khz.value() * 1e3,
                sample_rate=int(self.sample_rate.currentData()),
                dwell_s=self.dwell.value(),
                direct_sampling=self.direct_sampling.isChecked(),
            )
        except ValueError:
            return None

    def current_policy(self) -> StopPolicy:
        if self.mode_duration.isChecked():
            return StopPolicy("duration", duration_s=self.duration_min.value() * 60.0)
        if self.mode_sweeps.isChecked():
            return StopPolicy("sweeps", max_sweeps=self.sweep_count.value())
        return StopPolicy("indefinite")

    def current_gain(self):
        return self.gain.currentData()

    def _refresh_plan(self) -> None:
        try:
            plan = SweepPlan.create(
                f_start_hz=self.f_start.value() * 1e6,
                f_stop_hz=self.f_stop.value() * 1e6,
                bin_hz=self.bin_khz.value() * 1e3,
                sample_rate=int(self.sample_rate.currentData()),
                dwell_s=self.dwell.value(),
                direct_sampling=self.direct_sampling.isChecked(),
            )
        except ValueError as exc:
            self.estimate.setText(
                "<span style='color:#e06c6c;'>%s</span>" % str(exc).replace("\n", "<br>")
            )
            self.start_button.setEnabled(False)
            self.planChanged.emit(None)
            return

        secs = plan.estimated_sweep_seconds()
        self.estimate.setText(
            "<b>%s bins</b> &middot; %d steps &middot; %.2f kHz actual<br>"
            "<b>~%s per sweep</b>"
            % ("{:,}".format(plan.n_bins), plan.n_steps, plan.bin_hz / 1e3,
               _fmt_duration(secs))
        )
        self.start_button.setEnabled(not self._scanning)
        self.planChanged.emit(plan)


def _fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return "%.1f s" % seconds
    if seconds < 3600:
        return "%d m %02d s" % (seconds // 60, seconds % 60)
    return "%d h %02d m" % (seconds // 3600, (seconds % 3600) // 60)
