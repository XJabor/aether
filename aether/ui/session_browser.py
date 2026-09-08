"""Browse, load and compare saved scans."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView, QFileDialog, QHBoxLayout, QHeaderView, QLabel,
    QMessageBox, QPushButton, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
)

from ..analysis import compare
from ..storage.db import Database
from ..storage.session import SessionMeta

COLUMNS = ["Name", "When", "Range (MHz)", "RBW", "Sweeps", "Gain", "Location"]


class SessionBrowser(QWidget):
    sessionLoaded = Signal(object, object)      # SessionMeta, SpectrumData
    referenceLoaded = Signal(object, object)
    referenceCleared = Signal()

    def __init__(self, db: Database, parent=None) -> None:
        super().__init__(parent)
        self.db = db
        self._build()
        self.refresh()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)

        self.tree = QTreeWidget()
        self.tree.setColumnCount(len(COLUMNS))
        self.tree.setHeaderLabels(COLUMNS)
        self.tree.setRootIsDecorated(False)
        self.tree.setAlternatingRowColors(True)
        self.tree.setSelectionMode(QAbstractItemView.SingleSelection)
        self.tree.header().setSectionResizeMode(0, QHeaderView.Stretch)
        self.tree.setMinimumWidth(200)
        self.tree.itemDoubleClicked.connect(lambda *_: self.load_selected())
        root.addWidget(self.tree, 1)

        self.detail = QLabel("")
        self.detail.setWordWrap(True)
        self.detail.setStyleSheet("font-size: 11px; color: #9aa4ae; padding: 4px;")
        root.addWidget(self.detail)
        self.tree.itemSelectionChanged.connect(self._update_detail)

        row1 = QHBoxLayout()
        self.load_button = QPushButton("Load")
        self.ref_button = QPushButton("Load as baseline")
        self.clear_ref_button = QPushButton("Clear baseline")
        for b in (self.load_button, self.ref_button, self.clear_ref_button):
            row1.addWidget(b)
        root.addLayout(row1)

        row2 = QHBoxLayout()
        self.refresh_button = QPushButton("Refresh")
        self.csv_button = QPushButton("Export CSV")
        self.delete_button = QPushButton("Delete")
        for b in (self.refresh_button, self.csv_button, self.delete_button):
            row2.addWidget(b)
        root.addLayout(row2)

        self.load_button.clicked.connect(self.load_selected)
        self.ref_button.clicked.connect(self.load_reference)
        self.clear_ref_button.clicked.connect(self.referenceCleared.emit)
        self.refresh_button.clicked.connect(self.refresh)
        self.csv_button.clicked.connect(self.export_selected)
        self.delete_button.clicked.connect(self.delete_selected)

    # -- listing ---------------------------------------------------------

    def refresh(self) -> None:
        selected = self.selected_meta()
        self.tree.clear()
        for meta in self.db.list_sessions():
            item = QTreeWidgetItem([
                meta.name or meta.default_name(),
                meta.started_utc.replace("T", " ")[:16],
                "%.3f - %.3f" % (meta.f_start_hz / 1e6, meta.f_stop_hz / 1e6),
                "%.2f kHz" % (meta.bin_hz / 1e3),
                str(meta.sweep_count),
                meta.gain_label,
                ("%.5f, %.5f" % (meta.lat, meta.lon)) if meta.has_location else "-",
            ])
            item.setData(0, Qt.UserRole, meta)
            self.tree.addTopLevelItem(item)
        for i in range(1, len(COLUMNS)):
            self.tree.resizeColumnToContents(i)
        if selected is not None:
            self._reselect(selected.id)

    def _reselect(self, session_id) -> None:
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            meta = item.data(0, Qt.UserRole)
            if meta is not None and meta.id == session_id:
                self.tree.setCurrentItem(item)
                return

    def selected_meta(self) -> SessionMeta | None:
        item = self.tree.currentItem()
        return item.data(0, Qt.UserRole) if item is not None else None

    def _update_detail(self) -> None:
        meta = self.selected_meta()
        if meta is None:
            self.detail.setText("")
            return
        bits = [meta.summary()]
        if meta.device:
            bits.append("Device: %s" % meta.device)
        if meta.notes:
            bits.append("Notes: %s" % meta.notes)
        self.detail.setText("\n".join(bits))

    # -- actions ---------------------------------------------------------

    def _load_data(self, meta: SessionMeta):
        try:
            return self.db.load_data(meta)
        except (OSError, ValueError, FileNotFoundError) as exc:
            QMessageBox.critical(
                self, "Cannot load session",
                "The spectrum file for this session could not be read.\n\n%s" % exc,
            )
            return None

    def load_selected(self) -> None:
        meta = self.selected_meta()
        if meta is None:
            return
        data = self._load_data(meta)
        if data is not None:
            self.sessionLoaded.emit(meta, data)

    def load_reference(self) -> None:
        meta = self.selected_meta()
        if meta is None:
            return
        data = self._load_data(meta)
        if data is not None:
            self.referenceLoaded.emit(meta, data)

    def warn_if_incomparable(self, current: SessionMeta | None,
                             reference: SessionMeta | None) -> None:
        """Say plainly when a delta would be measuring the receiver."""
        if current is None or reference is None:
            return
        warnings = compare.comparability_warnings(current, reference)
        if not warnings:
            return
        QMessageBox.warning(
            self, "These scans may not be comparable",
            "The delta will still be drawn, but read it carefully:\n\n"
            + "\n\n".join("- " + w for w in warnings),
        )

    def export_selected(self) -> None:
        meta = self.selected_meta()
        if meta is None:
            return
        data = self._load_data(meta)
        if data is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export CSV", "%s.csv" % meta.slug(), "CSV (*.csv)"
        )
        if not path:
            return
        try:
            data.to_csv(path)
        except OSError as exc:
            QMessageBox.critical(self, "Export failed", str(exc))

    def delete_selected(self) -> None:
        meta = self.selected_meta()
        if meta is None:
            return
        answer = QMessageBox.question(
            self, "Delete session",
            "Permanently delete \"%s\" and its spectrum file?\n\n"
            "This cannot be undone." % (meta.name or meta.default_name()),
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        self.db.delete_session(meta.id)
        self.refresh()
