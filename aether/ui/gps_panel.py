"""GPS status and control.

Degrades cleanly: with no receiver attached the panel says so, scanning
proceeds normally, and sessions are simply stored without a location.
"""
from __future__ import annotations

import threading

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QComboBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QPushButton,
    QVBoxLayout, QWidget,
)

from ..gps import nmea
from ..storage.session import GpsFixRecord

STALE_AFTER_S = 10.0


class GpsPanel(QWidget):
    fixReceived = Signal(object)      # nmea.Fix

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._reader: nmea.NmeaReader | None = None
        self._track: list[GpsFixRecord] = []
        self._recording = False
        self._track_lock = threading.Lock()
        self._build()
        self.refresh_ports()

        # Poll rather than signalling per sentence: fixes arrive at 1 Hz and
        # the reader runs on a plain thread, so a timer on the GUI thread is
        # simpler and avoids cross-thread signal plumbing for no benefit.
        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._refresh_status)
        self._timer.start()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)

        box = QGroupBox("GPS receiver")
        form = QFormLayout(box)

        self.port = QComboBox()
        self.port.setMinimumWidth(200)
        form.addRow("Port", self.port)

        self.baud = QComboBox()
        for b in nmea.COMMON_BAUDS:
            self.baud.addItem(str(b), b)
        form.addRow("Baud", self.baud)

        buttons = QHBoxLayout()
        self.refresh_button = QPushButton("Rescan")
        self.detect_button = QPushButton("Auto-detect")
        self.connect_button = QPushButton("Connect")
        buttons.addWidget(self.refresh_button)
        buttons.addWidget(self.detect_button)
        buttons.addWidget(self.connect_button)
        form.addRow("", self._wrap(buttons))
        root.addWidget(box)

        status_box = QGroupBox("Position")
        status = QVBoxLayout(status_box)
        self.status = QLabel("Not connected")
        self.status.setWordWrap(True)
        self.status.setMinimumWidth(1)
        self.position = QLabel("--")
        self.position.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.position.setWordWrap(True)
        self.position.setMinimumWidth(1)
        self.position.setStyleSheet("font-family: Consolas, monospace;")
        self.track_label = QLabel("")
        self.track_label.setStyleSheet("color: #8a949e; font-size: 11px;")
        for w in (self.status, self.position, self.track_label):
            status.addWidget(w)
        root.addWidget(status_box)
        root.addStretch(1)

        self.refresh_button.clicked.connect(self.refresh_ports)
        self.detect_button.clicked.connect(self.autodetect)
        self.connect_button.clicked.connect(self.toggle_connection)

    @staticmethod
    def _wrap(layout) -> QWidget:
        w = QWidget()
        w.setLayout(layout)
        return w

    # -- ports -----------------------------------------------------------

    def refresh_ports(self) -> None:
        current = self.port.currentData()
        self.port.clear()
        ports = nmea.list_ports()
        if not ports:
            self.port.addItem("No serial ports found", None)
            self.connect_button.setEnabled(False)
            self.detect_button.setEnabled(False)
            self.status.setText(
                "No serial ports detected. Plug in a USB GPS receiver and "
                "press Rescan. Scanning works fine without one -- sessions "
                "are just saved without a location."
            )
            return
        self.connect_button.setEnabled(True)
        self.detect_button.setEnabled(True)
        for p in ports:
            self.port.addItem(str(p), p.device)
        if current:
            i = self.port.findData(current)
            if i >= 0:
                self.port.setCurrentIndex(i)

    def autodetect(self) -> None:
        self.status.setText("Probing serial ports for NMEA...")
        self.detect_button.setEnabled(False)
        self.repaint()
        try:
            found = nmea.autodetect(timeout_s=2.0)
        finally:
            self.detect_button.setEnabled(True)
        if found is None:
            self.status.setText(
                "No NMEA data found on any port. Check the receiver is "
                "powered and not held open by another program."
            )
            return
        device, baud = found
        i = self.port.findData(device)
        if i >= 0:
            self.port.setCurrentIndex(i)
        j = self.baud.findData(baud)
        if j >= 0:
            self.baud.setCurrentIndex(j)
        self.status.setText("Found NMEA on %s at %d baud." % (device, baud))
        self.connect()

    # -- connection ------------------------------------------------------

    def toggle_connection(self) -> None:
        self.disconnect() if self.is_connected else self.connect()

    @property
    def is_connected(self) -> bool:
        return self._reader is not None and self._reader.running

    def connect(self) -> None:
        device = self.port.currentData()
        if not device:
            return
        self.disconnect()
        self._reader = nmea.NmeaReader(
            device, int(self.baud.currentData()),
            on_fix=self._on_fix, on_error=self._on_error,
        )
        self._reader.start()
        self.connect_button.setText("Disconnect")
        self.status.setText("Listening on %s..." % device)

    def disconnect(self) -> None:
        if self._reader is not None:
            self._reader.stop()
            self._reader = None
        self.connect_button.setText("Connect")
        self.position.setText("--")
        self.status.setText("Not connected")

    # -- fixes -----------------------------------------------------------

    def _on_fix(self, fix) -> None:
        """Called on the reader thread. Only touches lock-protected state."""
        if not self._recording:
            return
        with self._track_lock:
            self._track.append(GpsFixRecord(
                t_utc=fix.utc, lat=fix.lat, lon=fix.lon, alt_m=fix.alt_m,
                speed_kt=fix.speed_kt, fix_quality=fix.quality, sats=fix.sats,
            ))

    def _on_error(self, message: str) -> None:
        self.status.setText(message)

    def _refresh_status(self) -> None:
        if self._reader is None:
            return
        fix = self._reader.last_fix
        if fix is None:
            self.status.setText(
                "Connected, waiting for a fix. A cold start can take "
                "several minutes with a clear view of the sky."
            )
            return
        age = self._reader.fix_age_s() or 0.0
        self.position.setText(fix.describe())
        if age > STALE_AFTER_S:
            self.status.setText("Last fix %.0f s ago (signal lost?)" % age)
        else:
            self.status.setText("Fix good (%.0f s ago)" % age)

    # -- recording -------------------------------------------------------

    def start_track(self) -> None:
        with self._track_lock:
            self._track.clear()
        self._recording = True

    def stop_track(self) -> list[GpsFixRecord]:
        self._recording = False
        with self._track_lock:
            return list(self._track)

    def current_fix(self):
        return self._reader.last_fix if self._reader is not None else None

    def update_track_label(self) -> None:
        with self._track_lock:
            n = len(self._track)
        self.track_label.setText("%d fixes recorded this scan" % n if n else "")
