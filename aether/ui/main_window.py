"""Main application window."""
from __future__ import annotations

from PySide6.QtCore import Qt, QThread, QTimer, Slot
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QCheckBox, QDockWidget, QFileDialog, QHBoxLayout, QInputDialog,
    QLabel, QMainWindow, QMessageBox, QProgressBar, QToolBar, QWidget,
)

from .. import config
from ..sdr.device import MockDevice, RtlDevice, SdrError, enumerate_devices, librtlsdr_status
from ..sdr.worker import ScanWorker
from ..storage.db import Database
from ..storage.session import SessionMeta, SpectrumData, utc_now
from .chat_panel import ChatPanel
from .control_panel import ControlPanel
from .gps_panel import GpsPanel
from .session_browser import SessionBrowser
from .settings_dialog import SettingsDialog
from .spectrum_view import TRACES, SpectrumView

REFRESH_HZ = 10


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("%s %s" % (config.APP_NAME, config.APP_VERSION))
        self.resize(1500, 900)

        self.settings = config.Settings.load()
        self.db = Database()

        self._thread: QThread | None = None
        self._worker: ScanWorker | None = None
        self._device = None
        self._plan = None
        self._live_data: SpectrumData | None = None
        self._loaded_meta: SessionMeta | None = None
        self._reference_meta: SessionMeta | None = None
        self._track: list = []
        self._elapsed = 0.0
        self._sweeps = 0
        self._saved = True
        self._use_mock = False

        self._build_ui()
        self._build_toolbar()
        self._build_menu()
        self._detect_device()
        self.spectrum.set_cal_offset(self.settings.cal_offset_db)

        self._refresh_timer = QTimer(self)
        self._refresh_timer.setInterval(int(1000 / REFRESH_HZ))
        self._refresh_timer.timeout.connect(self._repaint_traces)
        self._dirty = False

    # -- construction ----------------------------------------------------

    def _build_ui(self) -> None:
        self.spectrum = SpectrumView()
        self.setCentralWidget(self.spectrum)

        self.controls = ControlPanel()
        self.controls.startRequested.connect(self.start_scan)
        self.controls.stopRequested.connect(self.stop_scan)
        self.controls.planChanged.connect(self._on_plan_changed)

        self.setup_dock = self._dock("Scan setup", self.controls,
                                     Qt.LeftDockWidgetArea, 300)

        self.gps = GpsPanel()
        self.gps_dock = self._dock("GPS", self.gps, Qt.LeftDockWidgetArea, 300)
        self.tabifyDockWidget(self.setup_dock, self.gps_dock)
        self.setup_dock.raise_()

        self.sessions = SessionBrowser(self.db)
        self.sessions.sessionLoaded.connect(self._on_session_loaded)
        self.sessions.referenceLoaded.connect(self._on_reference_loaded)
        self.sessions.referenceCleared.connect(self._on_reference_cleared)
        self.sessions_dock = self._dock("Saved scans", self.sessions,
                                        Qt.BottomDockWidgetArea, 0)

        self.chat = ChatPanel()
        self.chat_dock = self._dock("Ask AI", self.chat,
                                    Qt.RightDockWidgetArea, 320)
        self.spectrum.regionChanged.connect(self.chat.set_region)

        self.status_label = QLabel("Ready")
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(240)
        self.progress.setVisible(False)
        self.statusBar().addWidget(self.status_label, 1)
        self.statusBar().addPermanentWidget(self.progress)

        self.device_label = QLabel("")
        self.statusBar().addPermanentWidget(self.device_label)

        self._arrange_docks()

    def _arrange_docks(self) -> None:
        """Give the spectrum the space. It is the reason the app exists.

        Left and right docks own the bottom corners, so the saved-scans dock
        spans only the centre column instead of the full window width.
        """
        self.setCorner(Qt.BottomLeftCorner, Qt.LeftDockWidgetArea)
        self.setCorner(Qt.BottomRightCorner, Qt.RightDockWidgetArea)
        self.setCorner(Qt.TopLeftCorner, Qt.LeftDockWidgetArea)
        self.setCorner(Qt.TopRightCorner, Qt.RightDockWidgetArea)

        self.resizeDocks([self.setup_dock, self.chat_dock], [340, 380], Qt.Horizontal)
        self.resizeDocks([self.sessions_dock], [190], Qt.Vertical)
        self.sessions_dock.setMaximumHeight(320)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        # Dock sizes set before the window is shown do not stick -- Qt has
        # not laid anything out yet, so the docks come up at their size
        # hints and the plot gets whatever is left. Re-apply once the real
        # geometry exists.
        if not getattr(self, "_docks_sized", False):
            self._docks_sized = True
            QTimer.singleShot(0, self._arrange_docks)

    def _dock(self, title: str, widget, area, min_width: int) -> QDockWidget:
        dock = QDockWidget(title, self)
        dock.setWidget(widget)
        dock.setFeatures(QDockWidget.DockWidgetMovable
                         | QDockWidget.DockWidgetFloatable
                         | QDockWidget.DockWidgetClosable)
        if min_width:
            dock.setMinimumWidth(min_width)
        self.addDockWidget(area, dock)
        return dock

    def _build_toolbar(self) -> None:
        bar = QToolBar("Traces")
        bar.setMovable(False)
        self.addToolBar(bar)

        holder = QWidget()
        row = QHBoxLayout(holder)
        row.setContentsMargins(6, 0, 6, 0)
        row.addWidget(QLabel("Show:"))

        self._trace_boxes: dict[str, QCheckBox] = {}
        for name, (label, color, _w, _z) in TRACES.items():
            box = QCheckBox(label)
            box.setChecked(name in ("avg_db", "peak_hold_db"))
            box.setStyleSheet("color: %s; font-weight: 600;" % color)
            box.toggled.connect(
                lambda checked, n=name: self.spectrum.set_trace_visible(n, checked)
            )
            row.addWidget(box)
            self._trace_boxes[name] = box
            self.spectrum.set_trace_visible(name, box.isChecked())

        row.addSpacing(16)
        self.region_box = QCheckBox("Band selector")
        self.region_box.setToolTip(
            "Drag a region on the plot to zoom it or scope an AI question to it."
        )
        self.region_box.toggled.connect(self.spectrum.enable_region)
        row.addWidget(self.region_box)
        row.addStretch(1)
        bar.addWidget(holder)

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("&File")

        self.save_action = QAction("&Save session...", self)
        self.save_action.setShortcut(QKeySequence.Save)
        self.save_action.setEnabled(False)
        self.save_action.triggered.connect(self.save_session)
        file_menu.addAction(self.save_action)

        self.csv_action = QAction("Export &CSV...", self)
        self.csv_action.setEnabled(False)
        self.csv_action.triggered.connect(self.export_csv)
        file_menu.addAction(self.csv_action)
        file_menu.addSeparator()

        quit_action = QAction("&Quit", self)
        quit_action.setShortcut(QKeySequence.Quit)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        settings_action = QAction("Se&ttings...", self)
        settings_action.triggered.connect(self.open_settings)
        file_menu.insertAction(quit_action, settings_action)
        file_menu.insertSeparator(quit_action)

        view_menu = self.menuBar().addMenu("&View")
        autoscale = QAction("&Autoscale", self)
        autoscale.setShortcut("Ctrl+0")
        autoscale.triggered.connect(self.spectrum.autoscale)
        view_menu.addAction(autoscale)
        view_menu.addSeparator()
        for dock in (self.setup_dock, self.gps_dock, self.sessions_dock,
                     self.chat_dock):
            view_menu.addAction(dock.toggleViewAction())

        dev_menu = self.menuBar().addMenu("&Device")
        self.mock_action = QAction("Use &mock device (no hardware)", self)
        self.mock_action.setCheckable(True)
        self.mock_action.toggled.connect(self._on_mock_toggled)
        dev_menu.addAction(self.mock_action)

        redetect = QAction("&Re-detect hardware", self)
        redetect.triggered.connect(self._detect_device)
        dev_menu.addAction(redetect)

    # -- device ----------------------------------------------------------

    def _detect_device(self) -> None:
        ok, msg = librtlsdr_status()
        if not ok:
            self.device_label.setText("No driver")
            self.device_label.setToolTip(msg)
            self._set_status(msg.split("\n")[0] + "  (Device > Use mock device to continue)")
            self.mock_action.setChecked(True)
            return

        names = enumerate_devices()
        if not names:
            self.device_label.setText("No device")
            self.device_label.setToolTip("librtlsdr loaded but no dongle found.")
            self._set_status("No RTL-SDR detected. Plug one in, or use the mock device.")
            self.mock_action.setChecked(True)
            return

        self.device_label.setText(names[0])
        self.device_label.setToolTip(msg)
        self._set_status("Found %s" % names[0])
        self._load_gains()

    def _load_gains(self) -> None:
        """Read the tuner's discrete gain steps so the UI offers real values."""
        if self._use_mock:
            self.controls.set_available_gains(MockDevice().valid_gains_db)
            return
        try:
            dev = RtlDevice()
        except SdrError as exc:
            self._set_status(str(exc).split("\n")[0])
            return
        try:
            self.controls.set_available_gains(dev.valid_gains_db)
        finally:
            dev.close()

    def _on_mock_toggled(self, checked: bool) -> None:
        self._use_mock = checked
        if checked:
            self.device_label.setText("Mock device")
            self.controls.set_available_gains(MockDevice().valid_gains_db)
            self._set_status("Using the synthetic device. No hardware involved.")
        else:
            self._detect_device()

    # -- scanning --------------------------------------------------------

    @Slot()
    def start_scan(self) -> None:
        if self._thread is not None:
            return
        plan = self.controls.current_plan()
        if plan is None:
            QMessageBox.warning(self, "Cannot start", "The current settings do not "
                                                      "describe a valid sweep.")
            return
        if not self._saved and self._live_data is not None:
            answer = QMessageBox.question(
                self, "Discard previous scan?",
                "The last scan has not been saved. Starting a new one will "
                "discard it.\n\nStart anyway?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return

        try:
            device = MockDevice() if self._use_mock else RtlDevice(
                bias_tee=self.controls.bias_tee.isChecked()
            )
        except SdrError as exc:
            QMessageBox.critical(self, "Cannot open device", str(exc))
            return

        self._device = device
        self._plan = plan
        policy = self.controls.current_policy()

        worker = ScanWorker(
            device, plan, policy,
            gain=self.controls.current_gain(),
            ppm=self.controls.ppm.value(),
        )
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_progress)
        worker.sweepCompleted.connect(self._on_sweep_completed)
        worker.dataChanged.connect(self._on_data_changed)
        worker.failed.connect(self._on_failed)
        worker.finished.connect(self._on_finished)

        self._worker, self._thread = worker, thread
        self._saved = False
        self._live_data = None
        self._loaded_meta = None
        self.gps.start_track()

        self.controls.set_scanning(True)
        self.progress.setVisible(True)
        self.progress.setRange(0, plan.n_steps)
        self.spectrum.set_data(None)
        self._set_status("Scanning %s, %s" % (plan.describe(), policy.describe()))

        thread.start()
        self._refresh_timer.start()

    @Slot()
    def stop_scan(self) -> None:
        if self._worker is not None:
            self._set_status("Stopping...")
            self._worker.stop()

    # -- worker signals --------------------------------------------------

    @Slot(object)
    def _on_progress(self, p) -> None:
        self.progress.setValue(p.step_index + 1)
        self._set_status(
            "Sweep %d  |  step %d/%d  |  %.3f MHz  |  %.0fs elapsed"
            % (p.sweep_index + 1, p.step_index + 1, p.n_steps,
               p.center_hz / 1e6, p.elapsed_s)
        )

    @Slot(int)
    def _on_sweep_completed(self, n: int) -> None:
        self._dirty = True

    @Slot()
    def _on_data_changed(self) -> None:
        self._dirty = True

    def _repaint_traces(self) -> None:
        """Driven by a timer rather than by the worker.

        The worker can finish a step every few milliseconds; repainting on
        each one would spend all its time in the renderer and make the UI
        unresponsive. Ten redraws a second looks continuous and costs almost
        nothing.
        """
        if not self._dirty or self._worker is None or self._plan is None:
            return
        self._dirty = False
        acc = self._worker.accumulator
        data = SpectrumData.from_accumulator(self._plan.freqs_hz, acc)
        first = self._live_data is None
        self._live_data = data
        self.spectrum.set_data(data, autorange=first)

    @Slot(str)
    def _on_failed(self, message: str) -> None:
        QMessageBox.critical(self, "Scan failed", message)
        self._set_status("Scan failed: %s" % message.split("\n")[0])

    @Slot()
    def _on_finished(self) -> None:
        self._refresh_timer.stop()
        self._track = self.gps.stop_track()
        self.gps.update_track_label()
        if self._worker is not None and self._plan is not None:
            acc = self._worker.accumulator
            if acc.count.max() > 0:
                self._live_data = SpectrumData.from_accumulator(self._plan.freqs_hz, acc)
                self.spectrum.set_data(self._live_data)
                self.chat.set_scan(self._live_data, self._current_meta())
                self._elapsed = self._worker.elapsed()
                self._sweeps = acc.sweeps_completed
                self.save_action.setEnabled(True)
                self.csv_action.setEnabled(True)
                self._clip_fraction = self._worker.engine.clip_fraction
                overload = self._worker.engine.overload_warning()
                if overload:
                    QMessageBox.warning(self, "Receiver overloaded", overload)
                    self._set_status("Finished, but OVERLOADED (%.2f%% clipped) "
                                     "-- lower the gain and rescan."
                                     % (100 * self._clip_fraction))
                else:
                    self._set_status(
                        "Finished: %d sweeps in %.0fs. Ctrl+S to save."
                        % (acc.sweeps_completed, self._worker.elapsed())
                    )
            else:
                self._set_status("Stopped before any data was captured.")

        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(3000)
        self._thread = None
        self._worker = None
        self._device = None
        self.controls.set_scanning(False)
        self.progress.setVisible(False)

    # -- saving ----------------------------------------------------------

    @Slot()
    def save_session(self) -> None:
        if self._live_data is None or self._plan is None:
            return
        plan = self._plan
        default = "%.3f-%.3f MHz  %s" % (
            plan.f_start_hz / 1e6, plan.f_stop_hz / 1e6, utc_now()[:16].replace("T", " ")
        )
        name, ok = QInputDialog.getText(self, "Save session", "Name:", text=default)
        if not ok:
            return

        gain = self.controls.current_gain()
        # Prefer the last live fix; fall back to the first point of the
        # recorded track if the receiver dropped out before the scan ended.
        track = getattr(self, "_track", []) or []
        fix = self.gps.current_fix()
        lat = fix.lat if fix else (track[0].lat if track else None)
        lon = fix.lon if fix else (track[0].lon if track else None)
        alt = fix.alt_m if fix else (track[0].alt_m if track else None)

        meta = SessionMeta(
            name=name.strip() or default,
            ended_utc=utc_now(),
            lat=lat,
            lon=lon,
            alt_m=alt,
            fix_quality=fix.quality if fix else None,
            sats=fix.sats if fix else None,
            f_start_hz=plan.f_start_hz,
            f_stop_hz=plan.f_stop_hz,
            bin_hz=plan.bin_hz,
            sample_rate=plan.sample_rate,
            n_fft=plan.n_fft,
            crop=plan.crop,
            gain_db=None if gain == "auto" else float(gain),
            agc=(gain == "auto"),
            ppm=self.controls.ppm.value(),
            direct_sampling=plan.direct_sampling,
            cal_offset_db=self.settings.cal_offset_db,
            clip_fraction=getattr(self, "_clip_fraction", 0.0),
            sweep_count=getattr(self, "_sweeps", 0),
            duration_s=getattr(self, "_elapsed", 0.0),
            device="Mock SDR" if self._use_mock else self.device_label.text(),
        )
        try:
            meta = self.db.save_session(meta, self._live_data, track)
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc))
            return
        self._saved = True
        self._loaded_meta = meta
        self.chat.set_scan(self._live_data, meta)
        self.sessions.refresh()
        self._set_status(
            "Saved session #%d (%d GPS fixes) to %s"
            % (meta.id, len(track), meta.npz_path)
        )

    @Slot()
    def export_csv(self) -> None:
        if self._live_data is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export CSV", str(config.data_root() / "spectrum.csv"), "CSV (*.csv)"
        )
        if not path:
            return
        try:
            self._live_data.to_csv(path)
        except OSError as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
            return
        self._set_status("Exported %s" % path)

    # -- misc ------------------------------------------------------------

    @Slot(object, object)
    def _on_session_loaded(self, meta, data) -> None:
        """Show a saved scan. Its own calibration offset applies, not the
        current setting -- the numbers belong to that capture."""
        self._live_data = data
        self._loaded_meta = meta
        self._saved = True
        self.spectrum.set_cal_offset(meta.cal_offset_db)
        self.spectrum.set_data(data, autorange=True)
        self.chat.set_scan(data, meta)
        self.csv_action.setEnabled(True)
        self.save_action.setEnabled(False)
        self._set_status("Loaded #%d: %s" % (meta.id, meta.summary()))

    @Slot(object, object)
    def _on_reference_loaded(self, meta, data) -> None:
        self._reference_meta = meta
        self.spectrum.set_reference(data)
        self.chat.set_reference(data, meta)
        current = getattr(self, "_loaded_meta", None) or self._current_meta()
        self.sessions.warn_if_incomparable(current, meta)
        self._set_status("Baseline: %s" % meta.summary())

    @Slot()
    def _on_reference_cleared(self) -> None:
        self._reference_meta = None
        self.spectrum.set_reference(None)
        self.chat.set_reference(None, None)
        self._set_status("Baseline cleared.")

    @Slot()
    def open_settings(self) -> None:
        dialog = SettingsDialog(self.settings, self)
        if dialog.exec() != SettingsDialog.Accepted:
            return
        dialog.apply_to(self.settings)
        self.spectrum.set_cal_offset(self.settings.cal_offset_db)
        self.chat.settings = self.settings
        self.chat._refresh_provider_state()
        self._set_status("Settings saved.")

    def _current_meta(self) -> SessionMeta | None:
        """Metadata describing the in-memory scan, for comparability checks
        before it has been saved."""
        if self._plan is None:
            return None
        gain = self.controls.current_gain()
        return SessionMeta(
            ended_utc=utc_now(),
            duration_s=self._elapsed,
            f_start_hz=self._plan.f_start_hz,
            f_stop_hz=self._plan.f_stop_hz,
            bin_hz=self._plan.bin_hz,
            sample_rate=self._plan.sample_rate,
            gain_db=None if gain == "auto" else float(gain),
            agc=(gain == "auto"),
            ppm=self.controls.ppm.value(),
            direct_sampling=self._plan.direct_sampling,
            cal_offset_db=self.settings.cal_offset_db,
            clip_fraction=getattr(self, "_clip_fraction", 0.0),
            sweep_count=getattr(self, "_sweeps", 0),
        )

    def _on_plan_changed(self, plan) -> None:
        if plan is not None and self._thread is None:
            self.spectrum.set_x_range_hz(plan.f_start_hz, plan.f_stop_hz)

    def _set_status(self, text: str) -> None:
        self.status_label.setText(text)

    def closeEvent(self, event) -> None:
        if self._thread is not None:
            self.stop_scan()
            self._thread.quit()
            self._thread.wait(3000)
        if not self._saved and self._live_data is not None:
            answer = QMessageBox.question(
                self, "Unsaved scan",
                "This scan has not been saved. Quit anyway?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                event.ignore()
                return
        self.gps.disconnect()
        self.db.close()
        event.accept()
