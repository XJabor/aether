"""Frequency allocation lookup, for labelling detected signals.

US-centric, deliberately coarse. These labels say "this is where such
traffic normally lives", not "this is what that signal is" -- a peak inside
the FM broadcast band is *probably* a station, but could equally be an
image or a spur from the receiver. Every label produced here should reach
the user (and the AI) as a hint, never as an identification.
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass

MHZ = 1_000_000.0


@dataclass(frozen=True)
class Band:
    lo_hz: float
    hi_hz: float
    name: str

    def contains(self, hz: float) -> bool:
        return self.lo_hz <= hz <= self.hi_hz

    @property
    def width_hz(self) -> float:
        return self.hi_hz - self.lo_hz


def _b(lo_mhz: float, hi_mhz: float, name: str) -> Band:
    return Band(lo_mhz * MHZ, hi_mhz * MHZ, name)


BANDS: list[Band] = [
    # HF (direct sampling territory)
    _b(0.530, 1.700, "AM broadcast"),
    _b(1.800, 2.000, "160 m amateur"),
    _b(2.300, 2.495, "120 m shortwave"),
    _b(3.500, 4.000, "80 m amateur"),
    _b(4.750, 5.060, "60 m shortwave"),
    _b(5.900, 6.200, "49 m shortwave"),
    _b(7.000, 7.300, "40 m amateur"),
    _b(9.400, 9.900, "31 m shortwave"),
    _b(11.600, 12.100, "25 m shortwave"),
    _b(13.570, 13.870, "22 m shortwave"),
    _b(14.000, 14.350, "20 m amateur"),
    _b(15.100, 15.800, "19 m shortwave"),
    _b(17.480, 17.900, "16 m shortwave"),
    _b(18.068, 18.168, "17 m amateur"),
    _b(21.000, 21.450, "15 m amateur"),
    _b(24.890, 24.990, "12 m amateur"),
    _b(26.965, 27.405, "CB radio"),
    _b(28.000, 29.700, "10 m amateur"),

    # VHF
    _b(50.000, 54.000, "6 m amateur"),
    _b(54.000, 88.000, "VHF TV ch 2-6"),
    _b(88.000, 108.000, "FM broadcast"),
    _b(108.000, 118.000, "Aeronautical navigation (VOR/ILS)"),
    _b(118.000, 137.000, "Airband (VHF air traffic)"),
    _b(137.000, 138.000, "Weather satellites (NOAA APT)"),
    _b(144.000, 148.000, "2 m amateur"),
    _b(148.000, 150.800, "Government / satellite"),
    _b(150.800, 156.000, "VHF business & public safety"),
    _b(156.000, 162.025, "Marine VHF"),
    _b(162.400, 162.550, "NOAA weather radio"),
    _b(162.550, 174.000, "VHF public safety / business"),
    _b(174.000, 216.000, "VHF TV ch 7-13"),
    _b(216.000, 225.000, "Maritime / land mobile"),

    # UHF
    _b(225.000, 400.000, "Military UHF air"),
    _b(300.000, 320.000, "Key fobs / remotes (315 MHz)"),
    _b(406.000, 406.100, "Emergency beacons (EPIRB/PLB)"),
    _b(420.000, 450.000, "70 cm amateur"),
    _b(433.050, 434.790, "ISM 433 MHz (fobs, sensors)"),
    _b(450.000, 470.000, "UHF business & public safety"),
    _b(462.550, 462.725, "GMRS/FRS"),
    _b(467.550, 467.725, "GMRS/FRS repeater input"),
    _b(470.000, 698.000, "UHF TV"),
    _b(698.000, 806.000, "700 MHz LTE / public safety"),
    _b(806.000, 824.000, "800 MHz public safety / SMR"),
    _b(824.000, 849.000, "Cellular 850 uplink"),
    _b(851.000, 869.000, "800 MHz public safety / SMR"),
    _b(869.000, 894.000, "Cellular 850 downlink"),
    _b(902.000, 928.000, "ISM 900 MHz (LoRa, telemetry)"),
    _b(928.000, 960.000, "Paging / fixed links"),

    # L-band and up
    _b(960.000, 1215.000, "Aeronautical radionavigation (DME/TACAN)"),
    _b(1030.000, 1030.000, "Mode S interrogation"),
    _b(1090.000, 1090.000, "ADS-B (aircraft transponders)"),
    _b(1176.000, 1176.900, "GPS L5"),
    _b(1227.000, 1228.200, "GPS L2"),
    _b(1240.000, 1300.000, "23 cm amateur"),
    _b(1525.000, 1559.000, "Inmarsat downlink"),
    _b(1574.000, 1577.000, "GPS L1"),
    _b(1610.000, 1626.500, "Iridium downlink"),
    _b(1710.000, 1755.000, "AWS-1 uplink"),
    _b(1850.000, 1910.000, "PCS uplink"),
    _b(1920.000, 1930.000, "DECT 6.0 cordless"),
    _b(1930.000, 1990.000, "PCS downlink"),
]

# Zero-width entries above are single spot frequencies; give them a
# tolerance so a detection a few kHz off still matches.
_SPOT_TOLERANCE_HZ = 2e6

_SORTED = sorted(BANDS, key=lambda b: b.lo_hz)
_LOWS = [b.lo_hz for b in _SORTED]


def lookup(hz: float) -> str | None:
    """Name of the narrowest allocation containing this frequency.

    Narrowest wins so that 462.5625 MHz reports "GMRS/FRS" rather than the
    much broader "UHF business & public safety" that also contains it.
    """
    matches = [b for b in _SORTED if b.width_hz > 0 and b.contains(hz)]

    # Spot frequencies (ADS-B, Mode S) are stored zero-width.
    for b in _SORTED:
        if b.width_hz == 0 and abs(hz - b.lo_hz) <= _SPOT_TOLERANCE_HZ:
            return b.name

    if not matches:
        return None
    return min(matches, key=lambda b: b.width_hz).name


def bands_overlapping(lo_hz: float, hi_hz: float) -> list[Band]:
    """Every allocation intersecting a span, for band-level rollups."""
    return [b for b in _SORTED if b.hi_hz >= lo_hz and b.lo_hz <= hi_hz]


def nearby(hz: float, count: int = 3) -> list[Band]:
    """Allocations closest to a frequency, for when lookup() finds nothing."""
    i = bisect_right(_LOWS, hz)
    window = _SORTED[max(0, i - count):i + count]
    return sorted(window, key=lambda b: min(abs(hz - b.lo_hz), abs(hz - b.hi_hz)))[:count]
