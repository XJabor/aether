"""Headless scan: sweep a range, save a session, print the strongest signals.

The GUI is a front end for exactly this. Keeping a CLI path means the
capture engine stays testable without Qt.

    python tools/scan_cli.py 88 108 --duration 60
    python tools/scan_cli.py 462 468 --bin 1 --sweeps 20
    python tools/scan_cli.py 88 108 --mock --sweeps 5
"""
from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aether import config
from aether.sdr.device import MockDevice, RtlDevice, SdrError
from aether.sdr.engine import StopPolicy, SweepEngine
from aether.sdr.sweep import SweepPlan
from aether.storage.db import Database
from aether.storage.session import SessionMeta, SpectrumData, utc_now


def main() -> int:
    ap = argparse.ArgumentParser(description="Sweep a frequency range into a baseline.")
    ap.add_argument("start_mhz", type=float)
    ap.add_argument("stop_mhz", type=float)
    ap.add_argument("--bin", type=float, default=10.0, metavar="KHZ",
                    help="resolution bandwidth in kHz (default 10)")
    ap.add_argument("--rate", type=float, default=config.DEFAULT_SAMPLE_RATE / 1e6,
                    metavar="MSPS", help="sample rate in MS/s (default 2.4)")
    ap.add_argument("--gain", default="auto",
                    help="tuner gain in dB, or 'auto' for AGC (default auto)")
    ap.add_argument("--ppm", type=int, default=0)
    ap.add_argument("--dwell", type=float, default=0.05, metavar="SEC",
                    help="capture time per step (default 0.05)")
    ap.add_argument("--direct", action="store_true", help="HF direct sampling")

    stop = ap.add_mutually_exclusive_group()
    stop.add_argument("--duration", type=float, metavar="SEC")
    stop.add_argument("--sweeps", type=int, metavar="N")

    ap.add_argument("--mock", action="store_true", help="synthetic device")
    ap.add_argument("--name", default="")
    ap.add_argument("--notes", default="")
    ap.add_argument("--lat", type=float)
    ap.add_argument("--lon", type=float)
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--csv", type=Path, help="also export a CSV here")
    args = ap.parse_args()

    # -- plan ----------------------------------------------------------
    try:
        plan = SweepPlan.create(
            f_start_hz=args.start_mhz * 1e6,
            f_stop_hz=args.stop_mhz * 1e6,
            bin_hz=args.bin * 1e3,
            sample_rate=int(args.rate * 1e6),
            dwell_s=args.dwell,
            direct_sampling=args.direct,
        )
    except ValueError as exc:
        print("Cannot plan that sweep:\n  %s" % exc, file=sys.stderr)
        return 2

    if args.duration is not None:
        policy = StopPolicy("duration", duration_s=args.duration)
    elif args.sweeps is not None:
        policy = StopPolicy("sweeps", max_sweeps=args.sweeps)
    else:
        policy = StopPolicy("indefinite")

    print(plan.describe())
    print("Scanning %s. Press Ctrl+C to stop early.\n" % policy.describe())

    # -- device --------------------------------------------------------
    try:
        device = MockDevice() if args.mock else RtlDevice()
    except SdrError as exc:
        print("Cannot open device:\n  %s" % exc, file=sys.stderr)
        return 3

    gain = args.gain if args.gain == "auto" else float(args.gain)
    if gain == "auto" and not args.mock:
        print("NOTE: AGC is on. Gain will drift between sweeps, so this "
              "baseline\n      will not be strictly comparable to others. "
              "Pass --gain for a fixed value.\n")

    engine = SweepEngine(device, plan, gain=gain, ppm=args.ppm)

    # Ctrl+C stops the sweep cleanly and keeps what was captured.
    signal.signal(signal.SIGINT, lambda *_: engine.stop())

    last = [0.0]

    def on_progress(p):
        now = time.monotonic()
        if now - last[0] < 0.25:
            return
        last[0] = now
        pct = 100.0 * (p.step_index + 1) / p.n_steps
        sys.stdout.write(
            "\r  sweep %d  step %d/%d (%5.1f%%)  %8.3f MHz  %6.1fs elapsed   "
            % (p.sweep_index + 1, p.step_index + 1, p.n_steps, pct,
               p.center_hz / 1e6, p.elapsed_s)
        )
        sys.stdout.flush()

    started = utc_now()
    acc = engine.run(policy, on_progress=on_progress)
    print("\n\nStopped after %.1fs, %d complete sweeps." % (
        engine.elapsed(), acc.sweeps_completed))
    device.close()

    overload = engine.overload_warning()
    if overload:
        print("\n*** %s ***\n" % overload, file=sys.stderr)
    elif engine.clip_fraction:
        print("Clipping: %.4f%% of samples (acceptable)."
              % (100 * engine.clip_fraction))

    if acc.count.max() == 0:
        print("No data captured.", file=sys.stderr)
        return 4

    # -- report --------------------------------------------------------
    data = SpectrumData.from_accumulator(plan.freqs_hz, acc)
    report_peaks(data)

    # -- save ----------------------------------------------------------
    if args.no_save:
        return 0

    meta = SessionMeta(
        name=args.name,
        started_utc=started,
        ended_utc=utc_now(),
        f_start_hz=plan.f_start_hz,
        f_stop_hz=plan.f_stop_hz,
        bin_hz=plan.bin_hz,
        sample_rate=plan.sample_rate,
        n_fft=plan.n_fft,
        crop=plan.crop,
        gain_db=None if gain == "auto" else float(gain),
        agc=(gain == "auto"),
        ppm=args.ppm,
        direct_sampling=args.direct,
        sweep_count=acc.sweeps_completed,
        duration_s=engine.elapsed(),
        clip_fraction=engine.clip_fraction,
        lat=args.lat,
        lon=args.lon,
        device=device.description,
        notes=args.notes,
    )
    with Database() as db:
        meta = db.save_session(meta, data)
    print("\nSaved session #%d" % meta.id)
    print("  %s" % meta.npz_path)

    if args.csv:
        data.to_csv(args.csv)
        print("  %s" % args.csv)
    return 0


def report_peaks(data: SpectrumData, top: int = 15) -> None:
    from aether.analysis.features import find_signals, noise_floor_db

    floor = noise_floor_db(data.avg_db)
    print("Noise floor: %.1f dBFS" % floor)

    sigs = find_signals(data, prominence_db=6.0)[:top]
    if not sigs:
        print("No signals stood out above the floor.")
        return

    print("\nStrongest signals:")
    print("  %-13s %9s %9s %9s  %s" % (
        "freq (MHz)", "avg dBFS", "peak", "burst dB", "band"))
    for s in sigs:
        print("  %-13.4f %9.1f %9.1f %9.1f  %s" % (
            s.freq_hz / 1e6, s.avg_db, s.peak_hold_db, s.burstiness_db, s.band or ""))


if __name__ == "__main__":
    raise SystemExit(main())
