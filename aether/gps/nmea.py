"""GPS input: NMEA over a serial port.

USB GPS receivers (u-blox, GlobalSat BU-353, most others) enumerate as a
COM port streaming NMEA sentences. Nothing here is specific to a vendor.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import serial
import serial.tools.list_ports

# Baud rates worth trying, most likely first. 9600 is the near-universal
# default; 4800 is the old NMEA 0183 standard; u-blox often ships at 38400.
COMMON_BAUDS = [9600, 4800, 38400, 115200, 57600]

# Ports whose description matches these are tried first during auto-detect.
_GPS_HINTS = ("gps", "gnss", "u-blox", "ublox", "nmea", "garmin", "globalsat",
              "prolific", "ch340", "cp210", "ftdi")


@dataclass
class Fix:
    """One position report."""

    lat: float
    lon: float
    alt_m: float | None = None
    speed_kt: float | None = None
    quality: int | None = None
    sats: int | None = None
    utc: str = ""

    @property
    def is_valid(self) -> bool:
        return (
            self.lat is not None and self.lon is not None
            and abs(self.lat) <= 90 and abs(self.lon) <= 180
            and not (self.lat == 0.0 and self.lon == 0.0)
        )

    @property
    def quality_text(self) -> str:
        return {
            0: "no fix", 1: "GPS fix", 2: "DGPS fix", 3: "PPS fix",
            4: "RTK fixed", 5: "RTK float", 6: "dead reckoning",
        }.get(self.quality, "unknown")

    def describe(self) -> str:
        parts = ["%.6f, %.6f" % (self.lat, self.lon)]
        if self.alt_m is not None:
            parts.append("%.0f m" % self.alt_m)
        if self.sats is not None:
            parts.append("%d sats" % self.sats)
        parts.append(self.quality_text)
        return "  |  ".join(parts)


@dataclass
class PortInfo:
    device: str
    description: str

    @property
    def looks_like_gps(self) -> bool:
        text = (self.description or "").lower()
        return any(h in text for h in _GPS_HINTS)

    def __str__(self) -> str:
        return "%s - %s" % (self.device, self.description or "unknown")


def list_ports() -> list[PortInfo]:
    """Serial ports, most GPS-looking first."""
    ports = [
        PortInfo(p.device, p.description or "")
        for p in serial.tools.list_ports.comports()
    ]
    return sorted(ports, key=lambda p: (not p.looks_like_gps, p.device))


def parse_sentence(line: str) -> Fix | None:
    """Parse one NMEA sentence into a Fix, or None if it carries no position.

    pynmea2 is imported lazily so that a missing or broken install degrades
    to "GPS unavailable" rather than preventing the app from starting.
    """
    try:
        import pynmea2
    except ImportError:
        return None

    line = line.strip()
    if not line.startswith("$"):
        return None
    try:
        msg = pynmea2.parse(line, check=False)
    except Exception:
        return None

    lat = getattr(msg, "latitude", None)
    lon = getattr(msg, "longitude", None)
    if lat is None or lon is None:
        return None

    kind = getattr(msg, "sentence_type", "")
    if kind == "RMC" and getattr(msg, "status", "A") != "A":
        return None      # receiver reports the fix is not valid yet

    quality = _as_int(getattr(msg, "gps_qual", None))
    if kind == "RMC" and quality is None:
        quality = 1

    fix = Fix(
        lat=float(lat),
        lon=float(lon),
        alt_m=_as_float(getattr(msg, "altitude", None)),
        speed_kt=_as_float(getattr(msg, "spd_over_grnd", None)),
        quality=quality,
        sats=_as_int(getattr(msg, "num_sats", None)),
        utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    return fix if fix.is_valid else None


def _as_float(v):
    try:
        return float(v) if v not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _as_int(v):
    try:
        return int(v) if v not in (None, "") else None
    except (TypeError, ValueError):
        return None


def probe_port(device: str, baud: int, timeout_s: float = 3.0) -> bool:
    """True if this port/baud is producing NMEA.

    Accepts any well-formed sentence, not only ones carrying a position: a
    receiver that is powered but has not yet acquired satellites is still
    the right port to listen on.
    """
    try:
        with serial.Serial(device, baud, timeout=0.5) as port:
            deadline = time.monotonic() + timeout_s
            while time.monotonic() < deadline:
                try:
                    raw = port.readline().decode("ascii", "ignore")
                except serial.SerialException:
                    return False
                if raw.startswith("$") and "," in raw:
                    return True
    except (serial.SerialException, OSError, ValueError):
        return False
    return False


def autodetect(timeout_s: float = 3.0) -> tuple[str, int] | None:
    """Find a port emitting NMEA. Returns (device, baud) or None."""
    for info in list_ports():
        for baud in COMMON_BAUDS:
            if probe_port(info.device, baud, timeout_s):
                return info.device, baud
    return None


class NmeaReader:
    """Background NMEA reader.

    Plain threading rather than QThread so the GPS layer stays usable from
    the CLI and from tests, matching how the sweep engine is structured.
    """

    def __init__(self, device: str, baud: int = 9600,
                 on_fix=None, on_error=None) -> None:
        self.device = device
        self.baud = baud
        self.on_fix = on_fix
        self.on_error = on_error

        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._last_fix: Fix | None = None
        self._last_time: float = 0.0

    # -- lifecycle -------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="nmea-reader")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # -- state -----------------------------------------------------------

    @property
    def last_fix(self) -> Fix | None:
        with self._lock:
            return self._last_fix

    def fix_age_s(self) -> float | None:
        with self._lock:
            if self._last_fix is None:
                return None
            return time.monotonic() - self._last_time

    # -- worker ----------------------------------------------------------

    def _run(self) -> None:
        try:
            port = serial.Serial(self.device, self.baud, timeout=1.0)
        except (serial.SerialException, OSError, ValueError) as exc:
            self._report_error("Could not open %s at %d baud: %s"
                               % (self.device, self.baud, exc))
            return

        try:
            while not self._stop.is_set():
                try:
                    raw = port.readline().decode("ascii", "ignore")
                except (serial.SerialException, OSError) as exc:
                    self._report_error("%s disconnected: %s" % (self.device, exc))
                    return
                if not raw:
                    continue
                fix = parse_sentence(raw)
                if fix is None:
                    continue
                with self._lock:
                    self._last_fix = fix
                    self._last_time = time.monotonic()
                if self.on_fix is not None:
                    try:
                        self.on_fix(fix)
                    except Exception:
                        pass
        finally:
            try:
                port.close()
            except Exception:
                pass

    def _report_error(self, message: str) -> None:
        if self.on_error is not None:
            try:
                self.on_error(message)
            except Exception:
                pass
