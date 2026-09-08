"""Headless smoke test: can we see the dongle, and what can it do?

    .venv\\Scripts\\python.exe tools\\check_device.py
    .venv\\Scripts\\python.exe tools\\check_device.py --mock
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from aether import config
from aether.sdr.device import MockDevice, RtlDevice, SdrError, librtlsdr_status


def main() -> int:
    ap = argparse.ArgumentParser(description="RTL-SDR smoke test")
    ap.add_argument("--mock", action="store_true", help="use the synthetic device")
    ap.add_argument("--index", type=int, default=0, help="device index")
    args = ap.parse_args()

    print("=" * 62)
    print("  RTL-SDR device check")
    print("=" * 62)

    if not args.mock:
        ok, msg = librtlsdr_status()
        print("\nlibrtlsdr: %s" % ("OK" if ok else "NOT USABLE"))
        print("  " + msg.replace("\n", "\n  "))
        if not ok:
            return 2

    try:
        dev = MockDevice() if args.mock else RtlDevice(args.index)
    except SdrError as exc:
        print("\nFAILED to open device:\n  %s" % str(exc).replace("\n", "\n  "))
        return 3

    try:
        print("\nDevice: %s" % dev.description)

        gains = dev.valid_gains_db
        print("Tuner gains (%d): %s" % (
            len(gains), ", ".join("%.1f" % g for g in gains) if gains else "none reported"
        ))

        rate = config.DEFAULT_SAMPLE_RATE
        gain = gains[len(gains) // 2] if gains else "auto"
        print("\nConfiguring: %.1f MS/s, gain %s, 0 ppm" % (rate / 1e6, gain))
        dev.configure(rate, gain, 0, False)

        f_test = 100_000_000
        dev.set_center_freq(f_test)
        dev.read(16384)                      # discard: let the PLL settle
        samples = dev.read(65536)

        print("\nCaptured %d samples at %.3f MHz" % (samples.size, f_test / 1e6))
        print("  dtype           %s" % samples.dtype)
        print("  mean |I+jQ|     %.4f" % np.abs(samples).mean())
        print("  DC offset       %.4f%+.4fj" % (samples.real.mean(), samples.imag.mean()))

        clipped = int((np.abs(samples.real) > 0.99).sum() + (np.abs(samples.imag) > 0.99).sum())
        pct = 100.0 * clipped / (2 * samples.size)
        print("  clipped samples %d (%.3f%%)%s" % (
            clipped, pct, "   <- LOWER THE GAIN" if pct > 0.1 else ""
        ))

        n = 4096
        w = np.hanning(n)
        spec = np.fft.fftshift(np.fft.fft(samples[:n] * w) / w.sum())
        power_db = 10 * np.log10(np.maximum(spec.real ** 2 + spec.imag ** 2, 1e-30))
        print("  spectrum        floor %.1f dBFS, peak %.1f dBFS" % (
            np.percentile(power_db, 10), power_db.max()
        ))

        print("\nDevice is working.")
        return 0
    finally:
        dev.close()


if __name__ == "__main__":
    raise SystemExit(main())
