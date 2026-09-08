"""Device access: librtlsdr loading, the real dongle, and a mock.

``pyrtlsdr`` resolves librtlsdr at *import* time, so the vendored ``lib/``
directory has to be on the DLL search path before that import happens. That
is why the import is deferred into :class:`RtlDevice` instead of sitting at
module scope.
"""
from __future__ import annotations


import struct
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from .. import config
from .librtlsdr import LibRtlSdrError, RtlSdrHandle, device_count, device_name


class SdrError(RuntimeError):
    """Anything that went wrong talking to the dongle."""


# --- librtlsdr discovery -------------------------------------------------

def dll_architecture(path: Path) -> str:
    """Read the PE header to tell x64 from x86.

    A 32-bit DLL under 64-bit Python fails with a misleading "not found"
    error, so we check up front and say something useful instead.
    """
    try:
        with open(path, "rb") as fh:
            fh.seek(0x3C)
            pe_off = struct.unpack("<I", fh.read(4))[0]
            fh.seek(pe_off + 4)
            machine = struct.unpack("<H", fh.read(2))[0]
    except (OSError, struct.error):
        return "unknown"
    return {0x8664: "x64", 0x14C: "x86"}.get(machine, "0x%04X" % machine)


def librtlsdr_status() -> tuple[bool, str]:
    """Return (usable, human readable explanation)."""
    dll = config.LIB_DIR / "rtlsdr.dll"
    if not dll.exists():
        return False, (
            "rtlsdr.dll not found in %s.\n"
            "Run tools\\fetch_librtlsdr.ps1 to download the official "
            "rtl-sdr-blog x64 build." % config.LIB_DIR
        )
    arch = dll_architecture(dll)
    if arch != "x64":
        return False, (
            "%s is %s, but this Python is 64-bit.\n"
            "You need the x64 build of librtlsdr; the DLL bundled with the "
            "32-bit SDR# download will not load here." % (dll, arch)
        )
    return True, "%s (%s)" % (dll, arch)


def enumerate_devices() -> list[str]:
    """Names of attached dongles. Empty if librtlsdr will not load."""
    try:
        return [device_name(i) for i in range(device_count())]
    except LibRtlSdrError:
        return []


# --- interface -----------------------------------------------------------

class SdrDevice(ABC):
    """The narrow slice of dongle behaviour the sweep engine needs."""

    sample_rate: int

    @abstractmethod
    def configure(self, sample_rate: int, gain: float | str, ppm: int,
                  direct_sampling: bool) -> None: ...

    @abstractmethod
    def set_center_freq(self, hz: float) -> None: ...

    @abstractmethod
    def read(self, n_samples: int) -> np.ndarray: ...

    @abstractmethod
    def close(self) -> None: ...

    @property
    @abstractmethod
    def valid_gains_db(self) -> list[float]: ...

    @property
    @abstractmethod
    def description(self) -> str: ...


# --- real hardware -------------------------------------------------------

class RtlDevice(SdrDevice):
    """A real dongle, via the direct ctypes binding in :mod:`librtlsdr`."""

    def __init__(self, device_index: int = 0, bias_tee: bool = False) -> None:
        ok, msg = librtlsdr_status()
        if not ok:
            raise SdrError(msg)
        try:
            self._h = RtlSdrHandle(device_index)
        except LibRtlSdrError as exc:
            raise SdrError(str(exc)) from exc

        self._index = device_index
        self._bias_tee = bias_tee
        self.sample_rate = config.DEFAULT_SAMPLE_RATE
        self._direct_mode = 0
        try:
            self._tuner = self._h.get_tuner_type()
            self._name = device_name(device_index)
        except LibRtlSdrError:
            self._tuner, self._name = "unknown", ""

    def configure(self, sample_rate: int, gain: float | str, ppm: int,
                  direct_sampling: bool) -> None:
        h = self._h
        # Only touch direct sampling when the mode actually changes.
        # Setting it redundantly makes librtlsdr retune before a centre
        # frequency exists, which prints a spurious "PLL not locked".
        want_mode = 2 if direct_sampling else 0
        if want_mode != self._direct_mode:
            try:
                h.set_direct_sampling(want_mode)
                self._direct_mode = want_mode
            except LibRtlSdrError as exc:
                if direct_sampling:
                    raise SdrError(
                        "Could not enable HF direct sampling: %s" % exc
                    ) from exc

        # Changing the sample rate makes librtlsdr re-tune. Until a centre
        # frequency has been chosen that means re-tuning to 0 Hz, which trips
        # this fork's automatic direct-sampling switch -- the device flips to
        # HF mode, fails to lock the PLL, and flips back on the first real
        # step. Park the tuner somewhere valid first so none of that happens.
        if not direct_sampling and h.get_center_freq() == 0:
            h.set_center_freq(100_000_000)

        # The RTL2832U divides its 28.8 MHz clock, so not every requested
        # rate is achievable. Report what we actually got: the sweep's bin
        # width is derived from it, so a silent substitution would shift
        # every frequency in the scan.
        self.sample_rate = h.set_sample_rate(int(sample_rate))

        # Only write the correction when it actually differs. Setting it
        # forces librtlsdr to re-tune, and at this point no centre frequency
        # has been chosen yet -- so it re-tunes to 0 Hz, which trips the
        # fork's automatic direct-sampling switch ("Enabled direct sampling
        # mode" followed by "PLL not locked") before flipping back on the
        # first real step. Harmless, but a pointless retune per scan.
        try:
            if ppm != h.get_freq_correction():
                h.set_freq_correction(ppm)
        except LibRtlSdrError as exc:
            raise SdrError("Could not set %d ppm correction: %s" % (ppm, exc)) from exc

        if isinstance(gain, str) and gain.lower() == "auto":
            h.set_auto_gain()
            h.set_agc_mode(True)
        else:
            h.set_agc_mode(False)
            h.set_manual_gain(float(gain))

        if self._bias_tee:
            h.set_bias_tee(True)

        h.reset_buffer()

    def set_center_freq(self, hz: float) -> None:
        try:
            self._h.set_center_freq(hz)
        except LibRtlSdrError as exc:
            raise SdrError(str(exc)) from exc

    def read(self, n_samples: int) -> np.ndarray:
        try:
            return self._h.read_samples(n_samples)
        except LibRtlSdrError as exc:
            raise SdrError("Read failed -- was the dongle unplugged? %s" % exc) from exc

    @property
    def valid_gains_db(self) -> list[float]:
        try:
            return self._h.get_tuner_gains()
        except LibRtlSdrError:
            return []

    @property
    def has_bias_tee(self) -> bool:
        return self._h.has_bias_tee

    @property
    def description(self) -> str:
        label = self._name or "RTL-SDR"
        return "%s (tuner %s)" % (label, self._tuner)

    def close(self) -> None:
        if self._bias_tee:
            try:
                self._h.set_bias_tee(False)   # never leave DC on the antenna
            except LibRtlSdrError:
                pass
        self._h.close()


# --- mock ----------------------------------------------------------------

class MockDevice(SdrDevice):
    """Synthetic dongle, so the GUI and analysis are developable with no
    hardware attached and the statistics are testable against known truth.

    Emits gaussian noise plus a table of tones. Each tone carries a duty
    cycle, which is what makes it possible to assert that peak-hold catches
    bursts the average washes out. A DC offset is included deliberately so
    the spike-removal path gets exercised.
    """

    # (frequency Hz, amplitude, duty cycle)
    DEFAULT_TONES: list[tuple[float, float, float]] = [
        (89_700_000, 0.30, 1.00),     # strong continuous carrier
        (94_100_000, 0.18, 1.00),
        (101_100_000, 0.25, 1.00),
        (107_900_000, 0.08, 1.00),    # weak continuous carrier
        (162_550_000, 0.12, 1.00),    # NOAA weather radio
        (462_562_500, 0.20, 0.10),    # bursty, FRS/GMRS-like
        (1_090_000_000, 0.15, 0.02),  # very bursty, ADS-B-like
    ]

    def __init__(self, tones=None, noise_amp: float = 0.02, seed: int = 0) -> None:
        self.tones = list(self.DEFAULT_TONES if tones is None else tones)
        self.noise_amp = noise_amp
        self.sample_rate = config.DEFAULT_SAMPLE_RATE
        self._fc = 100e6
        self._rng = np.random.default_rng(seed)
        self._phase = 0

    def configure(self, sample_rate: int, gain: float | str, ppm: int,
                  direct_sampling: bool) -> None:
        self.sample_rate = int(sample_rate)

    def set_center_freq(self, hz: float) -> None:
        self._fc = float(hz)

    def read(self, n_samples: int) -> np.ndarray:
        sr = self.sample_rate
        n = int(n_samples)
        sig = (
            self._rng.normal(0.0, self.noise_amp, n)
            + 1j * self._rng.normal(0.0, self.noise_amp, n)
        ).astype(np.complex128)

        # Continuous phase across reads, so a tone stays coherent.
        t = (np.arange(n, dtype=np.float64) + self._phase) / sr
        self._phase += n

        half = sr / 2.0
        for f_hz, amp, duty in self.tones:
            offset = f_hz - self._fc
            if abs(offset) >= half * 0.98:
                continue
            if duty < 1.0 and self._rng.random() > duty:
                continue
            sig += amp * np.exp(2j * np.pi * offset * t)

        sig += 0.05  # DC offset artifact, as the real hardware produces
        return sig.astype(np.complex64)

    @property
    def valid_gains_db(self) -> list[float]:
        return [0.0, 8.7, 16.6, 24.0, 32.8, 40.2, 49.6]

    @property
    def description(self) -> str:
        return "Mock SDR (synthetic signals, no hardware)"

    def close(self) -> None:
        pass


def open_device(use_mock: bool = False, device_index: int = 0) -> SdrDevice:
    return MockDevice() if use_mock else RtlDevice(device_index)
