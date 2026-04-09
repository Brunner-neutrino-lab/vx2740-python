"""
Basic acquisition example — simulation mode.

Run from the repo root:
    python examples/basic_acquisition.py
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vx2740 import VX2740Controller

# --- Configure controller ---
ctrl = VX2740Controller(address="192.168.0.1", mode="simulation")

# Simulate a realistic DCR at ~165 K (low rate)
for ch in range(4):
    ctrl.sim_set_dark_rate(ch, 200.0)       # 200 Hz dark rate (cold)
    ctrl.sim_set_spe_amplitude(ch, 200)     # 200 ADC counts per SPE

ctrl.connect()

# 2 µs pre-trigger, 10 µs post-trigger window
ctrl.configure_record_window(pre_us=2.0, post_us=10.0)

# Enable ch0–3 with individual thresholds; self-trigger
ctrl.configure_channels(
    sipm_channels  = [0, 1, 2, 3],
    thresholds     = {0: 100, 1: 100, 2: 100, 3: 100},
    threshold_mode = "per_channel",
    include_pmt    = True,
)
ctrl.configure_trigger("self")

# Progress callback
def show_progress(acquired, total):
    print(f"\r  {acquired}/{total} waveforms...", end="", flush=True)

ctrl.on_progress = show_progress

# --- Run acquisition ---
print("Acquiring 5000 waveforms (simulation)...")
result = ctrl.run(n_waveforms=5000, batch_size=500, store_waveforms=False)
print()  # newline after progress

# --- Summary ---
for ch in result.channel_ids:
    amps = result.amplitudes[ch]
    if len(amps) == 0:
        print(f"  ch{ch}: 0 pulses found")
    else:
        print(f"  ch{ch}: {len(amps)} pulses | "
              f"mean amp = {amps.mean():.1f} | "
              f"rate = {len(amps)/result.n_waveforms * 125e6 / 1500:.1f} Hz")

ctrl.disconnect()
print("Done.")
