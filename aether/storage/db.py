"""SQLite index over saved scan sessions.

Metadata lives here so sessions can be browsed and queried ("everything
within 1 km of this spot"); the bulky float arrays live beside it in .npz.
"""
from __future__ import annotations

import math
import sqlite3
from dataclasses import fields
from pathlib import Path

from .. import config
from .session import GpsFixRecord, SessionMeta, SpectrumData

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT    NOT NULL DEFAULT '',
    started_utc     TEXT    NOT NULL,
    ended_utc       TEXT    NOT NULL DEFAULT '',
    f_start_hz      REAL    NOT NULL,
    f_stop_hz       REAL    NOT NULL,
    bin_hz          REAL    NOT NULL,
    sample_rate     INTEGER NOT NULL,
    n_fft           INTEGER NOT NULL DEFAULT 0,
    crop            REAL    NOT NULL DEFAULT 0.8,
    gain_db         REAL,
    agc             INTEGER NOT NULL DEFAULT 0,
    ppm             INTEGER NOT NULL DEFAULT 0,
    direct_sampling INTEGER NOT NULL DEFAULT 0,
    cal_offset_db   REAL    NOT NULL DEFAULT 0,
    clip_fraction   REAL    NOT NULL DEFAULT 0,
    n_bins          INTEGER NOT NULL DEFAULT 0,
    sweep_count     INTEGER NOT NULL DEFAULT 0,
    duration_s      REAL    NOT NULL DEFAULT 0,
    lat             REAL,
    lon             REAL,
    alt_m           REAL,
    fix_quality     INTEGER,
    sats            INTEGER,
    device          TEXT    NOT NULL DEFAULT '',
    notes           TEXT    NOT NULL DEFAULT '',
    npz_path        TEXT    NOT NULL DEFAULT '',
    app_version     TEXT    NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS gps_track (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    t_utc       TEXT    NOT NULL,
    lat         REAL    NOT NULL,
    lon         REAL    NOT NULL,
    alt_m       REAL,
    speed_kt    REAL,
    fix_quality INTEGER,
    sats        INTEGER
);

CREATE INDEX IF NOT EXISTS idx_sessions_started ON sessions(started_utc DESC);
CREATE INDEX IF NOT EXISTS idx_sessions_range   ON sessions(f_start_hz, f_stop_hz);
CREATE INDEX IF NOT EXISTS idx_track_session    ON gps_track(session_id);
"""

_COLUMNS = [f.name for f in fields(SessionMeta) if f.name != "id"]


class Database:
    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path else config.db_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(SCHEMA)
        self._migrate()
        self._conn.commit()

    def _migrate(self) -> None:
        """Add columns this version knows about but an older database lacks.

        CREATE TABLE IF NOT EXISTS silently does nothing on an existing
        table, so a database written by an earlier version would be missing
        newer columns and every INSERT would fail.
        """
        have = {r["name"] for r in self._conn.execute("PRAGMA table_info(sessions)")}
        for column, ddl in [
            ("clip_fraction", "REAL NOT NULL DEFAULT 0"),
        ]:
            if column not in have:
                self._conn.execute(
                    "ALTER TABLE sessions ADD COLUMN %s %s" % (column, ddl)
                )

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- writing ---------------------------------------------------------

    def save_session(
        self,
        meta: SessionMeta,
        data: SpectrumData,
        track: list[GpsFixRecord] | None = None,
    ) -> SessionMeta:
        """Persist arrays then metadata, and return meta with id/npz_path set."""
        if not meta.name:
            meta.name = meta.default_name()

        npz = config.spectra_dir() / (meta.slug() + ".npz")
        n = 1
        while npz.exists():                      # never clobber an existing scan
            npz = config.spectra_dir() / ("%s_%d.npz" % (meta.slug(), n))
            n += 1
        data.save(npz)
        meta.npz_path = str(npz)
        meta.n_bins = int(data.freqs_hz.size)

        row = meta.to_row()
        cols = ", ".join(_COLUMNS)
        marks = ", ".join("?" for _ in _COLUMNS)
        cur = self._conn.execute(
            "INSERT INTO sessions (%s) VALUES (%s)" % (cols, marks),
            [self._encode(row[c]) for c in _COLUMNS],
        )
        meta.id = int(cur.lastrowid)

        if track:
            self.add_track(meta.id, track)
        self._conn.commit()
        return meta

    def add_track(self, session_id: int, fixes: list[GpsFixRecord]) -> None:
        self._conn.executemany(
            "INSERT INTO gps_track "
            "(session_id, t_utc, lat, lon, alt_m, speed_kt, fix_quality, sats) "
            "VALUES (?,?,?,?,?,?,?,?)",
            [(session_id, f.t_utc, f.lat, f.lon, f.alt_m, f.speed_kt,
              f.fix_quality, f.sats) for f in fixes],
        )
        self._conn.commit()

    def update_fields(self, session_id: int, **changes) -> None:
        allowed = {c: v for c, v in changes.items() if c in _COLUMNS}
        if not allowed:
            return
        sets = ", ".join("%s = ?" % c for c in allowed)
        self._conn.execute(
            "UPDATE sessions SET %s WHERE id = ?" % sets,
            [self._encode(v) for v in allowed.values()] + [session_id],
        )
        self._conn.commit()

    def delete_session(self, session_id: int, remove_file: bool = True) -> None:
        row = self._conn.execute(
            "SELECT npz_path FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        self._conn.execute("DELETE FROM gps_track WHERE session_id = ?", (session_id,))
        self._conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        self._conn.commit()
        if remove_file and row and row["npz_path"]:
            Path(row["npz_path"]).unlink(missing_ok=True)

    # -- reading ---------------------------------------------------------

    def list_sessions(self, limit: int = 500) -> list[SessionMeta]:
        rows = self._conn.execute(
            "SELECT * FROM sessions ORDER BY started_utc DESC LIMIT ?", (limit,)
        ).fetchall()
        return [SessionMeta.from_row(r) for r in rows]

    def get_session(self, session_id: int) -> SessionMeta | None:
        row = self._conn.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        return SessionMeta.from_row(row) if row else None

    def load_data(self, meta: SessionMeta) -> SpectrumData:
        if not meta.npz_path:
            raise FileNotFoundError("session %s has no spectrum file" % meta.id)
        return SpectrumData.load(Path(meta.npz_path))

    def get_track(self, session_id: int) -> list[GpsFixRecord]:
        rows = self._conn.execute(
            "SELECT t_utc, lat, lon, alt_m, speed_kt, fix_quality, sats "
            "FROM gps_track WHERE session_id = ? ORDER BY t_utc", (session_id,)
        ).fetchall()
        return [GpsFixRecord(**dict(r)) for r in rows]

    def sessions_near(self, lat: float, lon: float, radius_km: float,
                      limit: int = 100) -> list[SessionMeta]:
        """Sessions within radius_km. Pre-filtered with a bounding box in SQL,
        then refined with great-circle distance in Python."""
        dlat = radius_km / 111.0
        coslat = max(math.cos(math.radians(lat)), 1e-6)
        dlon = radius_km / (111.0 * coslat)
        rows = self._conn.execute(
            "SELECT * FROM sessions WHERE lat IS NOT NULL AND lon IS NOT NULL "
            "AND lat BETWEEN ? AND ? AND lon BETWEEN ? AND ? "
            "ORDER BY started_utc DESC LIMIT ?",
            (lat - dlat, lat + dlat, lon - dlon, lon + dlon, limit * 4),
        ).fetchall()
        out = []
        for r in rows:
            if haversine_km(lat, lon, r["lat"], r["lon"]) <= radius_km:
                out.append(SessionMeta.from_row(r))
        return out[:limit]

    # -- internals -------------------------------------------------------

    @staticmethod
    def _encode(value):
        return int(value) if isinstance(value, bool) else value


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))
