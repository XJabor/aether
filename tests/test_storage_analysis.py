"""Storage round-trips, band lookup, signal detection and comparison."""
from __future__ import annotations

import numpy as np
import pytest

from aether.analysis import bands, compare
from aether.analysis.features import band_occupancy, find_signals, noise_floor_db
from aether.storage.db import Database, haversine_km
from aether.storage.session import SessionMeta, SpectrumData


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setenv("RTLBASELINE_DATA", str(tmp_path))
    database = Database(tmp_path / "sessions.db")
    yield database
    database.close()


def make_spectrum(f_start=88e6, f_stop=108e6, bin_hz=10e3, tones=()):
    freqs = np.arange(f_start, f_stop, bin_hz)
    avg = np.full(freqs.size, -60.0, dtype=np.float32)
    for hz, level in tones:
        i = int(np.argmin(np.abs(freqs - hz)))
        avg[max(0, i - 1):i + 2] = level
    return SpectrumData(
        freqs_hz=freqs, avg_db=avg, min_db=avg - 3, max_db=avg + 3,
        peak_hold_db=avg + 6, count=np.full(freqs.size, 5, dtype=np.int32),
    )


def make_meta(**kw):
    base = dict(name="t", f_start_hz=88e6, f_stop_hz=108e6, bin_hz=10e3,
                sample_rate=2_400_000, gain_db=25.4, agc=False, sweep_count=5)
    base.update(kw)
    return SessionMeta(**base)


class TestStorage:
    def test_session_round_trip(self, db, tmp_path):
        data = make_spectrum(tones=[(101.1e6, -20.0)])
        meta = db.save_session(make_meta(lat=42.36, lon=-71.06), data)

        assert meta.id is not None
        loaded_meta = db.get_session(meta.id)
        loaded = db.load_data(loaded_meta)

        assert loaded_meta.gain_db == 25.4
        assert loaded_meta.agc is False
        assert loaded_meta.has_location
        np.testing.assert_allclose(loaded.freqs_hz, data.freqs_hz)
        np.testing.assert_allclose(loaded.avg_db, data.avg_db)

    def test_saving_twice_does_not_clobber_the_first_file(self, db):
        data = make_spectrum()
        a = db.save_session(make_meta(), data)
        b = db.save_session(make_meta(), data)
        assert a.npz_path != b.npz_path
        assert db.load_data(a) is not None

    def test_geo_query(self, db):
        data = make_spectrum()
        db.save_session(make_meta(lat=42.3601, lon=-71.0589), data)   # Boston
        db.save_session(make_meta(lat=34.0522, lon=-118.2437), data)  # LA
        near = db.sessions_near(42.36, -71.06, radius_km=10)
        assert len(near) == 1

    def test_delete_removes_row_and_file(self, db):
        from pathlib import Path
        meta = db.save_session(make_meta(), make_spectrum())
        path = Path(meta.npz_path)
        assert path.exists()
        db.delete_session(meta.id)
        assert db.get_session(meta.id) is None
        assert not path.exists()

    def test_gps_track(self, db):
        from aether.storage.session import GpsFixRecord
        meta = db.save_session(make_meta(), make_spectrum(), [
            GpsFixRecord(t_utc="2026-01-01T00:00:00+00:00", lat=42.0, lon=-71.0),
            GpsFixRecord(t_utc="2026-01-01T00:00:05+00:00", lat=42.1, lon=-71.1),
        ])
        assert len(db.get_track(meta.id)) == 2

    def test_csv_export(self, tmp_path):
        path = make_spectrum().to_csv(tmp_path / "out.csv")
        header = path.read_text().splitlines()[0]
        assert header.startswith("frequency_hz,avg_db")

    def test_haversine(self):
        # Boston to LA is roughly 4170 km
        assert haversine_km(42.3601, -71.0589, 34.0522, -118.2437) == pytest.approx(4170, rel=0.02)


class TestBands:
    @pytest.mark.parametrize("mhz,expected", [
        (89.7, "FM broadcast"),
        (121.5, "Airband (VHF air traffic)"),
        (146.52, "2 m amateur"),
        (162.55, "NOAA weather radio"),
        (462.5625, "GMRS/FRS"),
        (1090.0, "ADS-B (aircraft transponders)"),
        (1575.42, "GPS L1"),
        (27.185, "CB radio"),
    ])
    def test_lookup(self, mhz, expected):
        assert bands.lookup(mhz * 1e6) == expected

    def test_narrowest_band_wins(self):
        """162.55 MHz sits inside both NOAA weather and the much wider VHF
        public safety allocation; the specific one is the useful answer."""
        assert bands.lookup(162.55e6) == "NOAA weather radio"

    def test_unallocated_returns_none(self):
        assert bands.lookup(1_000_000_000_000) is None


class TestFeatures:
    def test_finds_injected_signals(self):
        data = make_spectrum(tones=[(101.1e6, -20.0), (94.5e6, -30.0)])
        signals = find_signals(data, prominence_db=6.0)
        found = sorted(round(s.freq_hz / 1e6, 1) for s in signals)
        assert 101.1 in found and 94.5 in found

    def test_signals_are_sorted_strongest_first(self):
        data = make_spectrum(tones=[(101.1e6, -20.0), (94.5e6, -30.0)])
        signals = find_signals(data, prominence_db=6.0)
        assert signals[0].avg_db >= signals[-1].avg_db

    def test_burstiness_and_band_label(self):
        data = make_spectrum(tones=[(101.1e6, -20.0)])
        s = find_signals(data, prominence_db=6.0)[0]
        assert s.burstiness_db == pytest.approx(6.0, abs=0.1)  # peak_hold is avg+6
        assert s.band == "FM broadcast"

    def test_noise_floor_ignores_the_signals(self):
        data = make_spectrum(tones=[(101.1e6, 0.0)])
        assert noise_floor_db(data.avg_db) == pytest.approx(-60.0, abs=1.0)

    def test_band_occupancy(self):
        data = make_spectrum(tones=[(101.1e6, -20.0)])
        rollup = band_occupancy(data, find_signals(data, prominence_db=6.0))
        fm = [b for b in rollup if b.name == "FM broadcast"]
        assert fm and fm[0].n_signals == 1


class TestCompare:
    def test_identical_scans_have_zero_delta(self):
        data = make_spectrum(tones=[(101.1e6, -20.0)])
        _, delta = compare.align_and_diff(data, data)
        assert np.nanmax(np.abs(delta)) == pytest.approx(0.0, abs=1e-6)

    def test_new_signal_is_detected_as_a_change(self):
        base = make_spectrum()
        now = make_spectrum(tones=[(101.1e6, -20.0)])
        changes = compare.find_changes(now, base, threshold_db=6.0)
        assert changes
        assert changes[0].peak_hz == pytest.approx(101.1e6, abs=20e3)
        assert changes[0].peak_delta_db > 0
        assert changes[0].direction == "appeared/stronger"

    def test_adjacent_bins_group_into_one_change(self):
        """A 3-bin wide signal must report once, not three times."""
        base = make_spectrum()
        now = make_spectrum(tones=[(101.1e6, -20.0)])
        assert len(compare.find_changes(now, base, threshold_db=6.0)) == 1

    def test_differing_grids_are_interpolated(self):
        a = make_spectrum(bin_hz=10e3)
        b = make_spectrum(bin_hz=20e3)
        freqs, delta = compare.align_and_diff(a, b)
        assert freqs.size == a.freqs_hz.size
        assert np.nanmax(np.abs(delta)) == pytest.approx(0.0, abs=1e-3)

    def test_gain_mismatch_is_flagged(self):
        warnings = compare.comparability_warnings(
            make_meta(gain_db=25.4), make_meta(gain_db=40.2))
        assert any("gain" in w.lower() for w in warnings)

    def test_agc_is_flagged(self):
        warnings = compare.comparability_warnings(
            make_meta(gain_db=None, agc=True), make_meta())
        assert any("agc" in w.lower() for w in warnings)

    def test_matching_scans_produce_no_warnings(self):
        assert compare.comparability_warnings(make_meta(), make_meta()) == []


class TestMigration:
    def test_older_database_gains_new_columns(self, tmp_path):
        """A database written before clip_fraction existed must still open."""
        import sqlite3
        path = tmp_path / "old.db"
        conn = sqlite3.connect(path)
        conn.executescript(
            "CREATE TABLE sessions (id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " name TEXT NOT NULL DEFAULT '', started_utc TEXT NOT NULL,"
            " f_start_hz REAL NOT NULL, f_stop_hz REAL NOT NULL,"
            " bin_hz REAL NOT NULL, sample_rate INTEGER NOT NULL);"
        )
        conn.commit()
        conn.close()

        db = Database(path)
        cols = {r["name"] for r in db._conn.execute("PRAGMA table_info(sessions)")}
        assert "clip_fraction" in cols
        db.close()

    def test_overload_shows_in_the_summary(self):
        assert "OVERLOADED" in make_meta(clip_fraction=0.25).summary()
        assert "OVERLOADED" not in make_meta(clip_fraction=0.0).summary()
