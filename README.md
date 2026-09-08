# Aether

A desktop app for building **statistical RF baselines** with a software-defined
radio: sweep a frequency range for as long as you like, accumulate per-bin
statistics, tag the result with a GPS location, save it, and later ask
"what changed?" — with an LLM to help interpret the answer.

SDR# shows you the spectrum right now. `rtl_power` gives you the numbers
with no GUI, no location, and no analysis. This sits in between: a survey
tool that remembers.

---

## Quick start

```bash
.venv\Scripts\python.exe run.py
```

Without hardware attached, use **Device → Use mock device** (or `run.py --mock`)
to explore the whole app against synthetic signals.

Headless equivalent:

```bash
.venv\Scripts\python.exe tools\scan_cli.py 88 108 --gain 25.4 --duration 60
```

---

## Setup

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
powershell -ExecutionPolicy Bypass -File .\tools\fetch_librtlsdr.ps1
.venv\Scripts\python.exe tools\check_device.py
```

`check_device.py` should report your tuner, its gain steps, and a sample
capture. If it does, everything else will work.

### The 32-bit trap

64-bit Python **cannot** load the `rtlsdr.dll` bundled with the 32-bit SDR#
download — the failure looks like "library not found", which sends people
hunting for a missing file that is right there. `fetch_librtlsdr.ps1`
downloads the official rtl-sdr-blog **x64** build into `lib/` and verifies
the PE header before accepting it. The app checks architecture at startup
and says so explicitly rather than failing cryptically.

### Why not pyrtlsdr

`pyrtlsdr` resolves every symbol it knows about at import time, including
`rtlsdr_set_dithering`, which the rtl-sdr-blog x64 build does not export.
Importing it against that DLL raises `AttributeError` before you can open a
device. `aether/sdr/librtlsdr.py` is a ~200-line ctypes binding that
binds the ~18 functions actually needed and treats the rest as optional, so
any reasonable librtlsdr build works.

---

## How a sweep works

The RTL-SDR sees ~2.4 MHz at a time, so a wide range is stitched from many
tuner steps.

1. **Crop the edges.** Only the middle 80% of each capture is kept; the
   outer 20% is filter rolloff and is not trustworthy.
2. **Snap every step to the global bin grid.** Each step centre is an exact
   integer number of bins from the start frequency, so stitching is a plain
   array copy — no resampling, no drift accumulating across 900 steps.
3. **Settle after retuning.** The first couple of reads after each retune
   are discarded while the R820T2 PLL locks. Skipping this is the most
   common cause of garbage at every step boundary.
4. **Welch-average** `M` FFT frames per step, then remove the fixed DC spike
   at the tuned centre by interpolation. Left alone that spike becomes a
   comb of ~900 evenly-spaced fake carriers across a full sweep.
5. **Accumulate** into four per-bin traces.

Verified by construction: every bin is covered exactly once, and every step
centre lands on-grid. See the tests below.

### The four traces

| Trace | What it answers |
|---|---|
| `avg` | What is normally here — the baseline |
| `min` | Approximate true noise floor |
| `max` | Strongest value any single sweep averaged to |
| `peak_hold` | Strongest value any single **FFT frame** saw |

The `avg` / `peak_hold` split is the one that matters. A handheld keying up
for two seconds inside a ten-minute scan barely moves `avg`, shows weakly in
`max`, and stands out clearly in `peak_hold`. Their difference is reported
as **burstiness** — near zero means a continuous carrier, large means an
intermittent transmitter.

Averaging happens in the **linear power domain**, converted to dB only on
read. Averaging dB values directly is a different and wrong operation that
biases the floor low.

---

## Reading the numbers honestly

**The axis is dBFS, not dBm.** This is an uncalibrated consumer receiver.
Comparisons *within* a scan, or *between* scans taken at the same gain, are
meaningful. Absolute power claims are not. The calibration offset in
Settings shifts the axis to line up with a reference you measured yourself —
it does not calibrate anything.

**Use a fixed gain for survey work.** With AGC on, gain drifts between
sweeps, so the average is averaging across changing reference levels and two
scans of the same site are not comparable — including with each other. The
app warns whenever AGC is enabled, and again when you compare two scans that
were not measured the same way.

**Watch the overload warning.** Every capture counts samples that hit the
ADC rails. Above 0.1% the app says so loudly, because a clipped baseline is
not merely noisy — the front end compresses, strong signals read *low*, and
intermodulation scatters peaks that are not transmitters at all. A silently
clipped scan is the worst failure this app could have, so it is checked on
every sample and stored with the session.

On the machine this was developed against, local FM is strong enough that
**gain 0 dB is the right setting** and anything above ~8 dB rails the ADC.
Do not assume a mid-scale gain is safe; check the warning, and add
attenuation or an FM trap if even minimum gain overloads.

**This receiver manufactures signals.** Tuner images, mixing products,
harmonics of the 28.8 MHz clock, and front-end intermod from strong local FM
all look exactly like real transmitters in a spectrum plot. Band labels in
this app are *hints about what normally occupies a frequency*, never
identifications. The AI system prompt says so explicitly — without that, a
model will confidently name receiver artifacts as real signals, which is the
most likely way this app could mislead you.

Note that wideband FM broadcast normally shows 3–10 dB of burstiness purely
from modulation. That is not intermittency.

---

## AI analysis

A full-range scan is ~186,000 bins — far too many to send, and unhelpful if
you could. `aether/ai/context.py` compresses a scan to a few thousand
tokens: measurement conditions, noise floor, a ranked table of detected
signals with measured bandwidth and burstiness, band occupancy, and (with a
baseline loaded) the differences. In practice ~12,000 bins reduces to about
600 tokens with nothing important lost.

Keys go into **Windows Credential Manager** via `keyring` — never into the
settings JSON, the session database, or exported scans, any of which you
might reasonably share along with a capture.

The scan summary is attached to the **first** question of a conversation
only; follow-ups do not re-send it.

---

## Storage

`%USERPROFILE%\Documents\Aether\` (override with `AETHER_DATA`).

Installs predating the rename keep working untouched: if `Documents\Aether`
does not exist but `Documents\RTLBaseline` does, that one is used as-is, and
saved API keys are still read from the old credential-store entry. Sessions
record an **absolute** path to their `.npz`, so nothing is ever moved behind
your back -- run `python tools/migrate_data.py --apply` to rename the folder
and rewrite those paths together.

- `sessions.db` — SQLite metadata, plus a `gps_track` table of timestamped
  fixes so mobile surveys record a path, not just a point
- `spectra/*.npz` — the float arrays

Every parameter that affects the measurement is stored per session, because
a baseline taken at a different gain or resolution is not comparable and the
comparison code needs to be able to say so.

---

## Tests

```bash
.venv\Scripts\python.exe -m pytest tests -v
```

The suite covers sweep geometry (exact tiling, on-grid centres), the
statistics (a 10% duty-cycle tone must be strong in peak-hold, weak in max,
nearly absent in avg), storage round-trips, band lookup, and comparison.

**Ground truth check against real RF** — the one that actually matters:

```bash
.venv\Scripts\python.exe tools\scan_cli.py 88 108 --gain 25.4 --sweeps 6
```

Every detected peak must land on a valid FM channel (odd tenths in the US:
88.1, 89.5, 101.1...). This validates step math, axis stitching, crop
boundaries and DC-spike removal at once, against something you can confirm
on a car radio.

---

## Layout

```
run.py                    entry point
lib/                      vendored rtlsdr.dll + rtl_test.exe (gitignored)
tools/                    fetch_librtlsdr.ps1, check_device.py, scan_cli.py,
                          migrate_data.py
aether/
  config.py               paths, hardware constants, settings
  sdr/       librtlsdr.py ctypes binding
             device.py    SdrDevice interface, RtlDevice + MockDevice
             sweep.py     step geometry (pure math)
             engine.py    the capture loop (no Qt)
             accumulate.py per-bin statistics
             worker.py    QThread wrapper
  gps/       nmea.py      port discovery, NMEA reader
  storage/   db.py, session.py
  analysis/  features.py, bands.py, compare.py
  ai/        provider.py (Anthropic), openai_provider.py, context.py, keys.py
  ui/        main_window.py, control_panel.py, spectrum_view.py,
             session_browser.py, gps_panel.py, chat_panel.py, settings_dialog.py
```

The capture engine is deliberately Qt-free so it can be driven from the CLI
and from tests; `worker.py` is a thin QThread wrapper over it.

`MockDevice` synthesises a spectrum with known tones at known duty cycles,
so the GUI and the analysis are fully developable and testable with no
hardware attached.
