# vx2740-python

Python driver and GUI for the CAEN VX2740 64-channel digitizer.

## Modes

| Mode | Description |
|------|-------------|
| `simulation` | Generates synthetic SiPM waveforms. No hardware required. |
| `hardware` | Connects to physical digitizer via CAEN FELib (Ethernet). |

## Quick Start

```bash
pip install -r requirements.txt

# Standalone GUI (simulation mode by default)
python -m vx2740.gui

# Headless scripting
python examples/basic_acquisition.py
```

## Hardware Setup

1. Install CAEN FELib C library from https://www.caen.it/products/caen-felib-library/
2. `pip install caen-felib`
3. Uncomment `caen-felib` in `requirements.txt`
4. Connect digitizer via Ethernet; note IP address
5. Select `hardware` mode in GUI or pass `mode="hardware"` to controller

## Channel Map

| Channel | Signal |
|---------|--------|
| ch0 | SiPM amplifier chain 1 (daughterboard 1) |
| ch1 | SiPM amplifier chain 2 (daughterboard 2) |
| ch2 | SiPM amplifier chain 3 (daughterboard 3) |
| ch3 | SiPM amplifier chain 4 (daughterboard 4) |
| ch4 | PMT (external trigger reference) |

## Record Window

Default: 2 µs pre-trigger + 10 µs post-trigger = **1500 samples** at 125 MS/s (8 ns/sample).

## Trigger Modes

| Mode | Behaviour |
|------|-----------|
| `self` | Each channel triggers independently on threshold crossing |
| `external` | All channels triggered by external input (PMT sync) |

## Threshold Modes

| Mode | Behaviour |
|------|-----------|
| `per_channel` | Each SiPM channel has its own ADC count threshold |
| `global` | All channels share one threshold value |

## API

```python
from vx2740 import VX2740Controller

with VX2740Controller("192.168.0.1", mode="simulation") as ctrl:
    ctrl.configure_record_window(pre_us=2.0, post_us=10.0)
    ctrl.configure_channels(
        sipm_channels  = [0, 1, 2, 3],
        thresholds     = {0: 150, 1: 150, 2: 150, 3: 150},
        threshold_mode = "per_channel",
    )
    ctrl.configure_trigger("self")
    result = ctrl.run(n_waveforms=10_000)

# result.amplitudes[ch]  -> np.ndarray of pulse amplitudes (ADC counts)
# result.timestamps[ch]  -> np.ndarray of arrival times (s)
```

## Hardware API Notes

The hardware driver uses CAEN FELib. All hardware-specific calls are marked with
`# VERIFY` comments in `vx2740/driver.py` — confirm parameter paths against the
[VX2740 FELib documentation](https://www.caen.it/products/caen-felib-library/)
before first use with real hardware.

## CAEN FELib Paths to Verify

| Parameter | Path used | Notes |
|-----------|-----------|-------|
| Record length | `/par/RecordLengthS` | In samples |
| Pre-trigger | `/par/PreTriggerS` | In samples |
| Channel enable | `/ch/N/par/ChEnable` | "True"/"False" |
| Self-trigger threshold | `/ch/N/par/SelfTriggerThreshold` | ADC counts |
| Trigger source | `/par/AcqTriggerSource` | "ChSelfTrigger" / "ExternalTrigger" |
| Scope endpoint | `/endpoint/scope` | For waveform readout |
| Arm | `/cmd/ArmAcquisition` | |
| Start | `/cmd/SwStartAcquisition` | |
| Stop | `/cmd/SwStopAcquisition` | |
| Disarm | `/cmd/DisarmAcquisition` | |
