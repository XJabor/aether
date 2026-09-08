"""Minimal ctypes binding to librtlsdr.

We bind librtlsdr directly rather than using ``pyrtlsdr`` because pyrtlsdr
resolves every symbol it knows about at import time, including extensions
such as ``rtlsdr_set_dithering`` that only exist in some forks. The
rtl-sdr-blog x64 build does not export that one, so importing pyrtlsdr
against it raises AttributeError before any device is even opened.

Here, the handful of functions the app actually uses are bound strictly and
everything else is optional: a build missing ``rtlsdr_set_bias_tee`` simply
reports the bias tee as unavailable instead of failing to load.
"""
from __future__ import annotations

import ctypes
import os
from ctypes import POINTER, byref, c_char_p, c_int, c_ubyte, c_uint, c_uint32
from pathlib import Path

import numpy as np

from .. import config

# librtlsdr requires read lengths to be a multiple of 512 bytes.
BYTE_ALIGNMENT = 512
# Largest single read_sync transfer we will ask for.
MAX_READ_BYTES = 1 << 18

TUNER_TYPES = {
    0: "unknown", 1: "E4000", 2: "FC0012", 3: "FC0013",
    4: "FC2580", 5: "R820T", 6: "R828D",
}

_p_dev = ctypes.c_void_p


class LibRtlSdrError(RuntimeError):
    pass


class _Lib:
    """Lazily loaded, process-wide handle to rtlsdr.dll."""

    _instance: "_Lib | None" = None

    def __init__(self) -> None:
        self.dll = self._load()
        self._bind()

    @classmethod
    def get(cls) -> "_Lib":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # -- loading ---------------------------------------------------------

    @staticmethod
    def _candidate_paths() -> list[Path]:
        vendored = config.LIB_DIR / "rtlsdr.dll"
        out = [vendored] if vendored.exists() else []
        out.append(Path("rtlsdr.dll"))          # anything already on PATH
        return out

    def _load(self):
        if config.LIB_DIR.is_dir():
            # Lets the DLL find its own dependencies sitting alongside it.
            try:
                os.add_dll_directory(str(config.LIB_DIR))
            except (OSError, AttributeError):
                pass

        errors = []
        for path in self._candidate_paths():
            try:
                return ctypes.CDLL(str(path))
            except OSError as exc:
                errors.append("%s: %s" % (path, exc))
        raise LibRtlSdrError(
            "Could not load rtlsdr.dll.\n  "
            + "\n  ".join(errors)
            + "\nRun tools\\fetch_librtlsdr.ps1 to download the x64 build."
        )

    def _bind(self) -> None:
        d = self.dll
        required = {
            "rtlsdr_get_device_count": (c_uint, []),
            "rtlsdr_get_device_name": (c_char_p, [c_uint]),
            "rtlsdr_open": (c_int, [POINTER(_p_dev), c_uint]),
            "rtlsdr_close": (c_int, [_p_dev]),
            "rtlsdr_set_sample_rate": (c_int, [_p_dev, c_uint32]),
            "rtlsdr_get_sample_rate": (c_uint32, [_p_dev]),
            "rtlsdr_set_center_freq": (c_int, [_p_dev, c_uint32]),
            "rtlsdr_get_center_freq": (c_uint32, [_p_dev]),
            "rtlsdr_set_freq_correction": (c_int, [_p_dev, c_int]),
            "rtlsdr_get_freq_correction": (c_int, [_p_dev]),
            "rtlsdr_get_tuner_type": (c_int, [_p_dev]),
            "rtlsdr_get_tuner_gains": (c_int, [_p_dev, POINTER(c_int)]),
            "rtlsdr_set_tuner_gain": (c_int, [_p_dev, c_int]),
            "rtlsdr_set_tuner_gain_mode": (c_int, [_p_dev, c_int]),
            "rtlsdr_set_agc_mode": (c_int, [_p_dev, c_int]),
            "rtlsdr_set_direct_sampling": (c_int, [_p_dev, c_int]),
            "rtlsdr_reset_buffer": (c_int, [_p_dev]),
            "rtlsdr_read_sync": (c_int, [_p_dev, ctypes.c_void_p, c_int, POINTER(c_int)]),
        }
        missing = []
        for name, (restype, argtypes) in required.items():
            fn = getattr(d, name, None)
            if fn is None:
                missing.append(name)
                continue
            fn.restype, fn.argtypes = restype, argtypes
        if missing:
            raise LibRtlSdrError(
                "rtlsdr.dll is missing required functions: %s\n"
                "This does not look like a usable librtlsdr build."
                % ", ".join(missing)
            )

        # Optional across builds; absence is reported, never fatal.
        self.optional: dict[str, bool] = {}
        for name, (restype, argtypes) in {
            "rtlsdr_set_bias_tee": (c_int, [_p_dev, c_int]),
            "rtlsdr_set_tuner_bandwidth": (c_int, [_p_dev, c_uint32]),
        }.items():
            fn = getattr(d, name, None)
            self.optional[name] = fn is not None
            if fn is not None:
                fn.restype, fn.argtypes = restype, argtypes


def device_count() -> int:
    return int(_Lib.get().dll.rtlsdr_get_device_count())


def device_name(index: int = 0) -> str:
    raw = _Lib.get().dll.rtlsdr_get_device_name(index)
    return raw.decode("utf-8", "replace") if raw else ""


def is_available() -> tuple[bool, str]:
    try:
        _Lib.get()
    except LibRtlSdrError as exc:
        return False, str(exc)
    n = device_count()
    if n == 0:
        return False, (
            "librtlsdr loaded, but no RTL-SDR devices were found.\n"
            "Check the dongle is plugged in and has the WinUSB driver (Zadig)."
        )
    return True, "%d device(s): %s" % (n, ", ".join(device_name(i) for i in range(n)))


class RtlSdrHandle:
    """Thin RAII wrapper over an open device handle."""

    def __init__(self, index: int = 0) -> None:
        self._lib = _Lib.get()
        self._dev = _p_dev()
        rc = self._lib.dll.rtlsdr_open(byref(self._dev), c_uint(index))
        if rc != 0 or not self._dev:
            raise LibRtlSdrError(
                "rtlsdr_open(%d) failed with code %d.\n"
                "Another program (SDR#?) may have the device open."
                % (index, rc)
            )
        self._closed = False
        self._buf = None
        self._lut = ((np.arange(256, dtype=np.float32) - 127.5) / 127.5)

    # -- lifecycle -------------------------------------------------------

    def close(self) -> None:
        if not self._closed and self._dev:
            self._lib.dll.rtlsdr_close(self._dev)
            self._closed = True

    def __enter__(self) -> "RtlSdrHandle":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _check(self, rc: int, what: str, tolerate: tuple[int, ...] = ()) -> None:
        if rc != 0 and rc not in tolerate:
            raise LibRtlSdrError("%s failed (code %d)" % (what, rc))

    # -- configuration ---------------------------------------------------

    def set_sample_rate(self, hz: int) -> int:
        self._check(self._lib.dll.rtlsdr_set_sample_rate(self._dev, c_uint32(int(hz))),
                    "set_sample_rate")
        return int(self._lib.dll.rtlsdr_get_sample_rate(self._dev))

    def set_center_freq(self, hz: float) -> None:
        self._check(self._lib.dll.rtlsdr_set_center_freq(self._dev, c_uint32(int(hz))),
                    "set_center_freq(%.0f)" % hz)

    def get_center_freq(self) -> int:
        return int(self._lib.dll.rtlsdr_get_center_freq(self._dev))

    def set_freq_correction(self, ppm: int) -> None:
        # librtlsdr returns -2 when asked to set the value it already holds.
        self._check(self._lib.dll.rtlsdr_set_freq_correction(self._dev, c_int(int(ppm))),
                    "set_freq_correction", tolerate=(-2,))

    def get_freq_correction(self) -> int:
        return int(self._lib.dll.rtlsdr_get_freq_correction(self._dev))

    def get_tuner_type(self) -> str:
        return TUNER_TYPES.get(int(self._lib.dll.rtlsdr_get_tuner_type(self._dev)), "unknown")

    def get_tuner_gains(self) -> list[float]:
        """Available gains in dB. librtlsdr reports tenths of a dB."""
        n = self._lib.dll.rtlsdr_get_tuner_gains(self._dev, None)
        if n <= 0:
            return []
        buf = (c_int * n)()
        got = self._lib.dll.rtlsdr_get_tuner_gains(self._dev, buf)
        return sorted(buf[i] / 10.0 for i in range(max(got, 0)))

    def set_manual_gain(self, gain_db: float) -> None:
        self._check(self._lib.dll.rtlsdr_set_tuner_gain_mode(self._dev, c_int(1)),
                    "set_tuner_gain_mode(manual)")
        self._check(self._lib.dll.rtlsdr_set_tuner_gain(self._dev, c_int(int(round(gain_db * 10)))),
                    "set_tuner_gain(%.1f)" % gain_db)

    def set_auto_gain(self) -> None:
        self._check(self._lib.dll.rtlsdr_set_tuner_gain_mode(self._dev, c_int(0)),
                    "set_tuner_gain_mode(auto)")

    def set_agc_mode(self, on: bool) -> None:
        self._check(self._lib.dll.rtlsdr_set_agc_mode(self._dev, c_int(1 if on else 0)),
                    "set_agc_mode")

    def set_direct_sampling(self, mode: int) -> None:
        """0 = off (tuner), 1 = I branch, 2 = Q branch (the HF mode on a V3)."""
        self._check(self._lib.dll.rtlsdr_set_direct_sampling(self._dev, c_int(mode)),
                    "set_direct_sampling(%d)" % mode)

    @property
    def has_bias_tee(self) -> bool:
        return self._lib.optional.get("rtlsdr_set_bias_tee", False)

    def set_bias_tee(self, on: bool) -> bool:
        """Feed 4.5 V up the coax to power an external LNA. Returns False if
        this build of librtlsdr cannot do it."""
        if not self.has_bias_tee:
            return False
        self._check(self._lib.dll.rtlsdr_set_bias_tee(self._dev, c_int(1 if on else 0)),
                    "set_bias_tee")
        return True

    def reset_buffer(self) -> None:
        self._check(self._lib.dll.rtlsdr_reset_buffer(self._dev), "reset_buffer")

    # -- reading ---------------------------------------------------------

    def read_samples(self, n_samples: int) -> np.ndarray:
        """Read n complex samples, normalised to roughly -1..+1.

        The dongle delivers interleaved unsigned 8-bit I/Q. Conversion goes
        through a 256-entry lookup table, which is markedly faster than
        arithmetic on the raw array at the rates we sweep at.
        """
        n_bytes = int(n_samples) * 2
        if n_bytes % BYTE_ALIGNMENT:
            n_bytes += BYTE_ALIGNMENT - (n_bytes % BYTE_ALIGNMENT)

        if self._buf is None or len(self._buf) < n_bytes:
            self._buf = (c_ubyte * n_bytes)()

        offset = 0
        while offset < n_bytes:
            want = min(n_bytes - offset, MAX_READ_BYTES)
            got = c_int(0)
            rc = self._lib.dll.rtlsdr_read_sync(
                self._dev, ctypes.byref(self._buf, offset), c_int(want), byref(got)
            )
            if rc != 0:
                raise LibRtlSdrError("rtlsdr_read_sync failed (code %d)" % rc)
            if got.value <= 0:
                break
            offset += got.value

        raw = np.frombuffer(self._buf, dtype=np.uint8, count=offset)
        raw = raw[: (raw.size // 2) * 2]
        if raw.size == 0:
            return np.empty(0, dtype=np.complex64)

        vals = self._lut[raw]
        result = np.empty(vals.size // 2, dtype=np.complex64)
        result.real = vals[0::2]
        result.imag = vals[1::2]
        return result[:n_samples]
