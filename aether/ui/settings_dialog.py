"""Settings: API keys, calibration, and analysis thresholds."""
from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton, QSpinBox,
    QVBoxLayout, QWidget,
)

from .. import config
from ..ai import keys


class SettingsDialog(QDialog):
    def __init__(self, settings: config.Settings, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(560)
        self.settings = settings
        self._build()

    def _build(self) -> None:
        root = QVBoxLayout(self)

        # --- AI ----------------------------------------------------------
        ai_box = QGroupBox("AI")
        ai_form = QFormLayout(ai_box)

        note = QLabel(
            keys.storage_problem()
            or "Keys are stored in %s, never in project files, the session "
            "database or exported scans." % keys.STORE_DESCRIPTION
        )
        note.setWordWrap(True)
        note.setStyleSheet("font-size: 11px; color: #8a949e;")
        ai_form.addRow(note)

        self.anthropic_key = self._key_row(ai_form, "Anthropic key", "anthropic")
        self.openai_key = self._key_row(ai_form, "OpenAI key", "openai")
        root.addWidget(ai_box)

        # --- Measurement --------------------------------------------------
        meas_box = QGroupBox("Measurement")
        meas = QFormLayout(meas_box)

        self.cal_offset = QDoubleSpinBox()
        self.cal_offset.setRange(-200.0, 200.0)
        self.cal_offset.setDecimals(1)
        self.cal_offset.setSuffix(" dB")
        self.cal_offset.setValue(self.settings.cal_offset_db)
        self.cal_offset.setToolTip(
            "Added to every power reading. Leave at 0 for raw dBFS.\n"
            "Only set this if you have actually characterised your antenna, "
            "feedline and gain against a known reference -- otherwise it makes "
            "the numbers look authoritative without making them true."
        )
        meas.addRow("Calibration offset", self.cal_offset)

        cal_note = QLabel(
            "An offset does not calibrate the receiver. It shifts the axis so "
            "readings line up with a reference you have measured yourself."
        )
        cal_note.setWordWrap(True)
        cal_note.setStyleSheet("font-size: 11px; color: #8a949e;")
        meas.addRow("", cal_note)
        root.addWidget(meas_box)

        # --- Analysis -----------------------------------------------------
        an_box = QGroupBox("Analysis")
        an = QFormLayout(an_box)

        self.prominence = QDoubleSpinBox()
        self.prominence.setRange(1.0, 60.0)
        self.prominence.setSuffix(" dB")
        self.prominence.setValue(self.settings.peak_prominence_db)
        self.prominence.setToolTip(
            "How far above the local noise floor a peak must rise before it "
            "counts as a signal. Lower finds more, including more noise."
        )
        an.addRow("Peak threshold", self.prominence)

        self.delta_threshold = QDoubleSpinBox()
        self.delta_threshold.setRange(1.0, 60.0)
        self.delta_threshold.setSuffix(" dB")
        self.delta_threshold.setValue(self.settings.delta_threshold_db)
        an.addRow("Change threshold", self.delta_threshold)

        self.max_peaks = QSpinBox()
        self.max_peaks.setRange(5, 500)
        self.max_peaks.setValue(self.settings.max_peaks_for_ai)
        self.max_peaks.setToolTip(
            "How many signals are listed in the summary sent to the AI. "
            "More detail costs more tokens."
        )
        an.addRow("Signals sent to AI", self.max_peaks)
        root.addWidget(an_box)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def _key_row(self, form: QFormLayout, label: str, provider: str) -> QLineEdit:
        field = QLineEdit()
        field.setEchoMode(QLineEdit.Password)
        field.setPlaceholderText(keys.masked(provider))

        save = QPushButton("Save")
        clear = QPushButton("Remove")

        def do_save() -> None:
            value = field.text().strip()
            if not value:
                return
            try:
                keys.set_key(provider, value)
            except keys.KeyStoreError as exc:
                QMessageBox.critical(self, "Could not save key", str(exc))
                return
            field.clear()
            field.setPlaceholderText(keys.masked(provider))
            QMessageBox.information(self, "Saved", "%s stored." % label)

        def do_clear() -> None:
            keys.delete_key(provider)
            field.clear()
            field.setPlaceholderText(keys.masked(provider))

        save.clicked.connect(do_save)
        clear.clicked.connect(do_clear)

        row = QHBoxLayout()
        row.addWidget(field, 1)
        row.addWidget(save)
        row.addWidget(clear)
        holder = QWidget()
        holder.setLayout(row)
        form.addRow(label, holder)
        return field

    def apply_to(self, settings: config.Settings) -> None:
        settings.cal_offset_db = self.cal_offset.value()
        settings.peak_prominence_db = self.prominence.value()
        settings.delta_threshold_db = self.delta_threshold.value()
        settings.max_peaks_for_ai = self.max_peaks.value()
        settings.save()
