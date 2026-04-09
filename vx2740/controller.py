"""
vx2740/controller.py

High-level controller for the CAEN VX2740 digitizer.

Wraps the driver with acquisition logic:
  - Record window configuration (pre/post trigger in physical units)
  - Per-channel or global threshold setting
  - Batch waveform acquisition with online pulse finding
  - Plugin interface for the ETS DAQ

Usage (headless):
    from vx2740.controller import VX2740Controller

    with VX2740Controller("192.168.0.1", mode="simulation") as vx:
        vx.configure_channels([0, 1, 2, 3], thresholds={0: 150, 1: 150, 2: 150, 3: 150})
        vx.configure_record_window(pre_us=2.0, post_us=10.0)
        vx.arm()
        result = vx.acquire(n_waveforms=1000)
        vx.disarm()
        # result["amplitudes"][ch] -> np.ndarray of pulse amplitudes
        # result["timestamps"][ch] -> np.ndarray of pulse timestamps (s)
"""

import time
import numpy as np
from dataclasses import dataclass, field
from typing import Callable

from .driver import (
    VX2740Driver,
    SAMPLE_RATE_HZ, ADC_MIDSCALE,
    DEFAULT_PRE_SAMPLES, DEFAULT_POST_SAMPLES, DEFAULT_N_SAMPLES,
    N_SIPM_CHANNELS, PMT_CHANNEL,
    SIM_SPE_AMPLITUDE_COUNTS,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AcquisitionResult:
    """
    Result of one acquisition block (N waveforms on one or more channels).

    Waveform-level data (raw):
        waveforms[ch]    -> np.ndarray shape (N, n_samples), int16 ADC counts

    Pulse-level data (from online pulse finding):
        amplitudes[ch]   -> np.ndarray of pulse amplitudes (ADC counts above baseline)
        timestamps[ch]   -> np.ndarray of pulse arrival times (s, relative to run start)

    Metadata:
        channel_ids      -> list of channels in this result
        n_waveforms      -> number of waveforms acquired
        bias_voltage_V   -> bias voltage at time of acquisition (set externally)
        temperature_K    -> temperature at time of acquisition (set externally)
        run_timestamp    -> Unix time at start of this acquisition block
    """
    waveforms:     dict = field(default_factory=dict)   # {ch: ndarray (N, n_samples)}
    amplitudes:    dict = field(default_factory=dict)   # {ch: ndarray (M_pulses,)}
    timestamps:    dict = field(default_factory=dict)   # {ch: ndarray (M_pulses,) in s}
    channel_ids:   list = field(default_factory=list)
    n_waveforms:   int  = 0
    bias_voltage_V: float = 0.0
    temperature_K:  float = 0.0
    run_timestamp:  float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

class VX2740Controller:
    """
    High-level controller for the CAEN VX2740 digitizer.

    Parameters
    ----------
    address : str
        IP address of the digitizer.
    mode : str
        "hardware" or "simulation".
    """

    # ------------------------------------------------------------------
    # Plugin interface (consumed by ETS DAQ auto-discovery)
    # ------------------------------------------------------------------
    MODULE_NAME  = "VX2740"
    DEVICE_NAME  = "CAEN VX2740 Digitizer"
    CONFIG_FIELDS = [
        {"key": "address",        "label": "IP Address",          "type": "str",   "default": "192.168.0.1"},
        {"key": "mode",           "label": "Mode",                "type": "choice","default": "simulation",
         "choices": ["simulation", "hardware"]},
        {"key": "pre_us",         "label": "Pre-trigger (µs)",    "type": "float", "default": 2.0},
        {"key": "post_us",        "label": "Post-trigger (µs)",   "type": "float", "default": 10.0},
        {"key": "threshold_mode", "label": "Threshold mode",      "type": "choice","default": "per_channel",
         "choices": ["per_channel", "global"]},
        {"key": "global_threshold","label": "Global threshold (ADC counts)","type": "int","default": 150},
        {"key": "sipm_channels",  "label": "SiPM channels",       "type": "str",   "default": "0,1,2,3"},
    ]
    DEFAULTS = {
        "address":          "192.168.0.1",
        "mode":             "simulation",
        "pre_us":           2.0,
        "post_us":          10.0,
        "threshold_mode":   "per_channel",
        "global_threshold": 150,
        "sipm_channels":    "0,1,2,3",
    }

    @staticmethod
    def test(config: dict) -> tuple[bool, str]:
        """
        Test connectivity. Called by the DAQ GUI Test Connection button.
        Runs in a worker thread (not the Qt main thread).
        """
        try:
            ctrl = VX2740Controller(
                address=config.get("address", "192.168.0.1"),
                mode=config.get("mode", "simulation"),
            )
            ctrl.connect()
            idn = ctrl.identify()
            ctrl.disconnect()
            return True, f"OK — {idn}"
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

    @staticmethod
    def read(config: dict) -> dict:
        """
        Return a snapshot of instrument state for the DAQ to log.
        """
        return {
            "address": config.get("address", ""),
            "mode":    config.get("mode", "simulation"),
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def __init__(self, address: str = "192.168.0.1", mode: str = "simulation"):
        self._driver = VX2740Driver(address=address, mode=mode)

        # Record window
        self._pre_samples  = DEFAULT_PRE_SAMPLES
        self._post_samples = DEFAULT_POST_SAMPLES
        self._n_samples    = DEFAULT_N_SAMPLES

        # Channel config
        self._sipm_channels:  list[int]       = list(range(N_SIPM_CHANNELS))
        self._channels_enabled: list[int]     = list(range(N_SIPM_CHANNELS)) + [PMT_CHANNEL]
        self._thresholds:       dict[int, int] = {}
        self._threshold_mode:   str            = "per_channel"   # or "global"
        self._global_threshold: int            = 150

        # Trigger mode
        self._trigger_mode: str = "self"  # "self" or "external"

        # Pulse finding parameters
        self._pf_sigma_counts: float = SIM_SPE_AMPLITUDE_COUNTS * 0.5
        self._pf_min_height:   float = SIM_SPE_AMPLITUDE_COUNTS * 0.3

        self._armed = False

        # Run tracking
        self._run_start: float = 0.0
        self._total_waveforms: int = 0

        # Optional progress callback: fn(acquired: int, total: int)
        self.on_progress: Callable | None = None

    def connect(self):
        self._driver.connect()

    def disconnect(self):
        if self._armed:
            self.disarm()
        self._driver.disconnect()

    def identify(self) -> str:
        """Return a short identification string."""
        return f"CAEN VX2740 [{self._driver.mode} mode] @ {self._driver._address}"

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def configure_record_window(self, pre_us: float = 2.0, post_us: float = 10.0):
        """
        Set pre- and post-trigger record window in microseconds.

        Parameters
        ----------
        pre_us : float
            Microseconds of samples before the trigger. Default 2 µs (250 samples).
        post_us : float
            Microseconds of samples after the trigger. Default 10 µs (1250 samples).
        """
        self._pre_samples  = int(round(pre_us  * SAMPLE_RATE_HZ * 1e-6))
        self._post_samples = int(round(post_us * SAMPLE_RATE_HZ * 1e-6))
        self._n_samples    = self._pre_samples + self._post_samples

    def configure_channels(self,
                           sipm_channels: list[int],
                           thresholds: dict[int, int] | None = None,
                           threshold_mode: str = "per_channel",
                           global_threshold: int = 150,
                           include_pmt: bool = True):
        """
        Configure which SiPM channels to enable and their trigger thresholds.

        Parameters
        ----------
        sipm_channels : list[int]
            Channel indices for the SiPM amplifier chains (e.g. [0, 1, 2, 3]).
        thresholds : dict[int, int], optional
            Per-channel threshold in ADC counts above baseline.
            Required when threshold_mode == "per_channel".
        threshold_mode : str
            "per_channel" — each channel has its own threshold (from thresholds dict)
            "global"      — all channels use global_threshold
        global_threshold : int
            ADC counts used when threshold_mode == "global".
        include_pmt : bool
            Whether to also enable the PMT channel (ch4) for readout.
        """
        self._sipm_channels   = sipm_channels
        self._threshold_mode  = threshold_mode
        self._global_threshold = global_threshold

        # Build the threshold dict
        if threshold_mode == "global":
            self._thresholds = {ch: global_threshold for ch in sipm_channels}
        else:
            if thresholds is None:
                raise ValueError("thresholds dict required when threshold_mode='per_channel'")
            self._thresholds = dict(thresholds)

        self._channels_enabled = list(sipm_channels)
        if include_pmt:
            self._channels_enabled.append(PMT_CHANNEL)

    def configure_trigger(self, mode: str = "self"):
        """
        Set the trigger source.

        Parameters
        ----------
        mode : str
            "self"     — channels self-trigger on threshold crossing
            "external" — external trigger (PMT sync via ch4 or external input)
        """
        if mode not in ("self", "external"):
            raise ValueError(f"trigger mode must be 'self' or 'external', got {mode!r}")
        self._trigger_mode = mode

    def set_threshold(self, channel: int, threshold_counts: int):
        """Update threshold for a single channel (per_channel mode)."""
        self._thresholds[channel] = threshold_counts

    def set_global_threshold(self, threshold_counts: int):
        """Update global threshold and apply to all enabled SiPM channels."""
        self._global_threshold = threshold_counts
        if self._threshold_mode == "global":
            for ch in self._sipm_channels:
                self._thresholds[ch] = threshold_counts

    # Simulation helpers
    def sim_set_dark_rate(self, channel: int, rate_hz: float):
        """Set simulated dark count rate (simulation mode only)."""
        self._driver.sim_set_dark_rate(channel, rate_hz)

    def sim_set_spe_amplitude(self, channel: int, amplitude_counts: int):
        """Set simulated SPE amplitude (simulation mode only)."""
        self._driver.sim_set_spe_amplitude(channel, amplitude_counts)

    # ------------------------------------------------------------------
    # Acquisition
    # ------------------------------------------------------------------

    def arm(self):
        """Push current configuration to digitizer and arm for acquisition."""
        self._driver.configure(
            n_samples        = self._n_samples,
            pre_samples      = self._pre_samples,
            channels_enabled = self._channels_enabled,
            thresholds       = self._thresholds,
            trigger_mode     = self._trigger_mode,
        )
        self._driver.arm()
        self._armed         = True
        self._run_start     = time.time()
        self._total_waveforms = 0

    def disarm(self):
        """Stop acquisition."""
        self._driver.disarm()
        self._armed = False

    def acquire(self,
                n_waveforms: int,
                batch_size: int = 1000,
                store_waveforms: bool = False,
                timeout_s: float = 60.0) -> AcquisitionResult:
        """
        Acquire n_waveforms and return pulse amplitudes and timestamps.

        Waveforms are processed in batches to limit memory usage.
        Raw waveform arrays are discarded after pulse finding unless
        store_waveforms=True.

        Parameters
        ----------
        n_waveforms : int
            Total number of waveforms to acquire.
        batch_size : int
            Number of waveforms to read from digitizer per call.
        store_waveforms : bool
            If True, store raw waveforms in AcquisitionResult.waveforms.
            Memory-intensive for large n_waveforms.
        timeout_s : float
            Maximum time to wait for each batch.

        Returns
        -------
        AcquisitionResult
        """
        if not self._armed:
            raise RuntimeError("Not armed — call arm() first.")

        result = AcquisitionResult(
            channel_ids  = self._sipm_channels,
            run_timestamp = self._run_start,
        )
        # Initialize output arrays
        for ch in self._sipm_channels:
            result.amplitudes[ch]  = []
            result.timestamps[ch]  = []
            if store_waveforms:
                result.waveforms[ch] = []

        acquired = 0
        while acquired < n_waveforms:
            this_batch = min(batch_size, n_waveforms - acquired)
            raw = self._driver.read_waveforms(this_batch, timeout_s=timeout_s)

            batch_timestamps = raw.get(-1, np.zeros(this_batch))

            for ch in self._sipm_channels:
                if ch not in raw:
                    continue
                waves = raw[ch]  # shape (this_batch, n_samples)

                if store_waveforms:
                    result.waveforms[ch].append(waves)

                # Online pulse finding per waveform
                amps, ts = self._find_pulses_batch(
                    waves, batch_timestamps, ch
                )
                result.amplitudes[ch].extend(amps)
                result.timestamps[ch].extend(ts)

            acquired += this_batch
            self._total_waveforms += this_batch

            if self.on_progress is not None:
                self.on_progress(acquired, n_waveforms)

        # Consolidate lists to arrays
        for ch in self._sipm_channels:
            result.amplitudes[ch] = np.array(result.amplitudes[ch], dtype=np.float32)
            result.timestamps[ch] = np.array(result.timestamps[ch], dtype=np.float64)
            if store_waveforms and result.waveforms[ch]:
                result.waveforms[ch] = np.concatenate(result.waveforms[ch], axis=0)

        result.n_waveforms = acquired
        return result

    # ------------------------------------------------------------------
    # Pulse finding
    # ------------------------------------------------------------------

    def _find_pulses_batch(self,
                           waves: np.ndarray,
                           batch_timestamps: np.ndarray,
                           channel: int) -> tuple[list, list]:
        """
        Find pulses in a batch of waveforms.

        Strategy:
          1. Subtract per-waveform baseline (mean of pre-trigger samples)
          2. Find peak amplitude in the post-trigger window
          3. Accept peaks above threshold

        Returns (amplitudes, timestamps) lists.
        """
        pre  = self._pre_samples
        amps = []
        ts   = []
        threshold = self._thresholds.get(channel, self._global_threshold)

        # Voltage scale: ADC_MIDSCALE maps to 0; convert to signed
        waves_signed = waves.astype(np.float32) - ADC_MIDSCALE

        # Baseline from pre-trigger region
        baselines = waves_signed[:, :pre].mean(axis=1, keepdims=True)
        waves_bl  = waves_signed - baselines

        # Post-trigger window
        post_window = waves_bl[:, pre:]

        for i in range(len(waves)):
            peak_val = float(post_window[i].max())
            if peak_val >= threshold:
                amps.append(peak_val)
                # Timestamp: waveform start time + time of peak within window
                peak_idx   = int(post_window[i].argmax())
                peak_time  = (batch_timestamps[i]
                              + (pre + peak_idx) / SAMPLE_RATE_HZ)
                ts.append(peak_time)

        return amps, ts

    def configure_pulse_finding(self,
                                min_height_counts: float,
                                sigma_counts: float | None = None):
        """
        Adjust pulse finding parameters.

        Parameters
        ----------
        min_height_counts : float
            Minimum peak height above baseline to accept as a pulse (ADC counts).
        sigma_counts : float, optional
            Expected noise sigma (for future threshold estimation features).
        """
        self._pf_min_height  = min_height_counts
        if sigma_counts is not None:
            self._pf_sigma_counts = sigma_counts

    # ------------------------------------------------------------------
    # Convenience: full acquisition (arm → acquire → disarm)
    # ------------------------------------------------------------------

    def run(self,
            n_waveforms: int,
            batch_size: int = 1000,
            store_waveforms: bool = False,
            timeout_s: float = 60.0) -> AcquisitionResult:
        """
        Arm, acquire n_waveforms, disarm, and return result.

        Convenience wrapper for single-shot acquisitions.
        """
        self.arm()
        try:
            return self.acquire(n_waveforms, batch_size, store_waveforms, timeout_s)
        finally:
            self.disarm()

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *_):
        self.disconnect()
