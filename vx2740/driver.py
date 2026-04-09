"""
vx2740/driver.py

Low-level interface to the CAEN VX2740 digitizer.

Two modes:
  - "hardware": wraps caen_felib (CAEN FELib C library must be installed)
  - "simulation": generates synthetic SiPM waveforms for development/testing

Only bytes-on-the-wire logic lives here. No experiment logic, no Qt.
"""

import time
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SAMPLE_RATE_HZ   = 125e6          # 125 MS/s
SAMPLE_PERIOD_NS = 8.0            # ns per sample
ADC_BITS         = 14
ADC_FULL_SCALE   = 2**ADC_BITS    # 16384 counts
ADC_MIDSCALE     = ADC_FULL_SCALE // 2

N_CHANNELS       = 64             # total channels on VX2740
N_SIPM_CHANNELS  = 4              # ch 0-3: SiPM amplifier chains
PMT_CHANNEL      = 4              # ch 4: PMT (external trigger reference)

# Default record window
DEFAULT_PRE_SAMPLES  = 250        # 2 µs
DEFAULT_POST_SAMPLES = 1250       # 10 µs
DEFAULT_N_SAMPLES    = DEFAULT_PRE_SAMPLES + DEFAULT_POST_SAMPLES  # 1500

# Simulation: shaped pulse parameters (CR112 + CR200-1µs)
# Gaussian approximation: peak at 1 µs after avalanche = 125 samples post-trigger
SIM_PULSE_SIGMA_SAMPLES  = 50     # σ ≈ 0.4 µs → FWHM ≈ 1 µs
SIM_PULSE_PEAK_OFFSET    = 125    # samples after trigger point (= pre_samples)
SIM_NOISE_SIGMA_COUNTS   = 3.0    # baseline RMS in ADC counts
SIM_SPE_AMPLITUDE_COUNTS = 200    # ~1.2% of full scale — typical SPE after shaper
SIM_DARK_RATE_HZ         = 500.0  # default dark count rate per channel


class VX2740Driver:
    """
    Low-level driver for the CAEN VX2740.

    Parameters
    ----------
    address : str
        IP address of the digitizer, e.g. "192.168.0.1".
        Ignored in simulation mode.
    mode : str
        "hardware" or "simulation".
    """

    def __init__(self, address: str = "192.168.0.1", mode: str = "simulation"):
        if mode not in ("hardware", "simulation"):
            raise ValueError(f"mode must be 'hardware' or 'simulation', got {mode!r}")

        self._address = address
        self._mode    = mode
        self._handle  = None        # caen_felib handle (hardware mode)
        self._ep      = None        # endpoint handle (hardware mode)
        self._connected = False

        # Current configuration (populated by configure())
        self._n_samples    = DEFAULT_N_SAMPLES
        self._pre_samples  = DEFAULT_PRE_SAMPLES
        self._channels_enabled = list(range(N_SIPM_CHANNELS + 1))  # ch0–4
        self._thresholds   = {}     # {ch: int (ADC counts)}
        self._armed        = False

        # Simulation state
        self._sim_dark_rates = {ch: SIM_DARK_RATE_HZ for ch in range(N_CHANNELS)}
        self._sim_spe_amplitudes = {ch: SIM_SPE_AMPLITUDE_COUNTS for ch in range(N_CHANNELS)}
        self._sim_rng = np.random.default_rng()

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self):
        """Open connection to the digitizer."""
        if self._connected:
            return

        if self._mode == "hardware":
            self._connect_hardware()
        else:
            self._connect_simulation()

        self._connected = True

    def disconnect(self):
        """Close connection."""
        if not self._connected:
            return

        if self._mode == "hardware":
            self._disconnect_hardware()

        self._connected = False
        self._handle = None
        self._ep     = None
        self._armed  = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def mode(self) -> str:
        return self._mode

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def configure(self,
                  n_samples: int,
                  pre_samples: int,
                  channels_enabled: list[int],
                  thresholds: dict[int, int],
                  trigger_mode: str = "self"):
        """
        Configure acquisition parameters.

        Parameters
        ----------
        n_samples : int
            Total samples per waveform (pre + post trigger).
        pre_samples : int
            Samples before the trigger point.
        channels_enabled : list[int]
            Channel indices to enable.
        thresholds : dict[int, int]
            Trigger threshold per channel in ADC counts.
            For global mode, all channels get the same value.
        trigger_mode : str
            "self"     — each channel self-triggers on threshold
            "external" — trigger from ch4 (PMT) or external input
        """
        if not self._connected:
            raise RuntimeError("Not connected — call connect() first.")

        self._n_samples        = n_samples
        self._pre_samples      = pre_samples
        self._channels_enabled = channels_enabled
        self._thresholds       = thresholds
        self._trigger_mode     = trigger_mode

        if self._mode == "hardware":
            self._configure_hardware(n_samples, pre_samples,
                                     channels_enabled, thresholds, trigger_mode)

    # ------------------------------------------------------------------
    # Acquisition
    # ------------------------------------------------------------------

    def arm(self):
        """Arm the digitizer. Ready to accept triggers."""
        if not self._connected:
            raise RuntimeError("Not connected.")
        self._armed = True
        if self._mode == "hardware":
            self._arm_hardware()

    def disarm(self):
        """Stop acquisition and disarm."""
        self._armed = False
        if self._mode == "hardware":
            self._disarm_hardware()

    def read_waveforms(self, n: int, timeout_s: float = 30.0) -> dict[int, np.ndarray]:
        """
        Read n waveform acquisitions.

        Blocks until n waveforms are collected or timeout_s elapses.

        Returns
        -------
        dict[int, np.ndarray]
            {channel: array of shape (n, n_samples)} in ADC counts (int16).
            Timestamp channel (key = -1) holds relative timestamps in seconds.
        """
        if not self._armed:
            raise RuntimeError("Digitizer is not armed — call arm() first.")

        if self._mode == "hardware":
            return self._read_hardware(n, timeout_s)
        else:
            return self._read_simulation(n)

    # ------------------------------------------------------------------
    # Hardware implementation
    # ------------------------------------------------------------------

    def _connect_hardware(self):
        """
        Open a connection via CAEN FELib.

        VERIFY with hardware: confirm caen_felib import path and open() call.
        See: https://www.caen.it/products/caen-felib-library/
        """
        try:
            import caen_felib.lib as felib
        except ImportError as e:
            raise ImportError(
                "caen_felib not installed. Install with: pip install caen-felib\n"
                "Also ensure the CAEN FELib C library is installed on the host."
            ) from e

        self._felib  = felib
        url          = f"dig2://{self._address}"
        self._handle = felib.open(url)  # VERIFY: exact function name

    def _disconnect_hardware(self):
        if self._handle is not None:
            try:
                if self._armed:
                    self._disarm_hardware()
                self._felib.close(self._handle)  # VERIFY
            except Exception:
                pass

    def _configure_hardware(self, n_samples, pre_samples,
                            channels_enabled, thresholds, trigger_mode):
        """
        Send configuration SCPI/endpoint commands to the board.
        VERIFY all parameter paths against VX2740 FELib documentation.
        """
        felib  = self._felib
        handle = self._handle

        # Record window
        felib.set_value(handle, "/par/RecordLengthS", str(n_samples))      # VERIFY path
        felib.set_value(handle, "/par/PreTriggerS",   str(pre_samples))     # VERIFY path

        # Disable all channels first
        for ch in range(N_CHANNELS):
            felib.set_value(handle, f"/ch/{ch}/par/ChEnable", "False")     # VERIFY path

        # Enable requested channels and set thresholds
        for ch in channels_enabled:
            felib.set_value(handle, f"/ch/{ch}/par/ChEnable", "True")      # VERIFY path
            if ch in thresholds:
                felib.set_value(handle,
                                f"/ch/{ch}/par/SelfTriggerThreshold",
                                str(thresholds[ch]))                        # VERIFY path

        # Trigger source
        if trigger_mode == "self":
            felib.set_value(handle, "/par/AcqTriggerSource",
                            "ChSelfTrigger")                                # VERIFY value
        elif trigger_mode == "external":
            felib.set_value(handle, "/par/AcqTriggerSource",
                            "ExternalTrigger")                              # VERIFY value

        # Configure scope endpoint for waveform readout
        self._ep = felib.get_endpoint(handle, "/endpoint/scope")           # VERIFY path

    def _arm_hardware(self):
        felib  = self._felib
        handle = self._handle
        felib.send_command(handle, "/cmd/ArmAcquisition")                  # VERIFY
        felib.send_command(handle, "/cmd/SwStartAcquisition")              # VERIFY

    def _disarm_hardware(self):
        felib  = self._felib
        handle = self._handle
        try:
            felib.send_command(handle, "/cmd/SwStopAcquisition")           # VERIFY
            felib.send_command(handle, "/cmd/DisarmAcquisition")           # VERIFY
        except Exception:
            pass

    def _read_hardware(self, n: int, timeout_s: float) -> dict[int, np.ndarray]:
        """
        Read n events from the scope endpoint.
        VERIFY event structure against FELib documentation.
        """
        felib = self._felib
        ep    = self._ep

        waveforms  = {ch: [] for ch in self._channels_enabled}
        timestamps = []
        deadline   = time.monotonic() + timeout_s
        collected  = 0

        while collected < n:
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"Timeout after {timeout_s:.1f}s: collected {collected}/{n} events."
                )
            try:
                evt = felib.read_data(ep, timeout=100)                     # VERIFY API
                # VERIFY: event structure — assumed dict with 'waveforms' and 'timestamp'
                ts = evt.get("timestamp", 0.0)
                timestamps.append(ts)
                for ch in self._channels_enabled:
                    raw = evt["waveforms"][ch]                              # VERIFY key
                    waveforms[ch].append(np.asarray(raw, dtype=np.int16))
                collected += 1
            except TimeoutError:
                continue

        result = {ch: np.stack(waveforms[ch]) for ch in self._channels_enabled}
        result[-1] = np.array(timestamps, dtype=np.float64)
        return result

    # ------------------------------------------------------------------
    # Simulation implementation
    # ------------------------------------------------------------------

    def _connect_simulation(self):
        """Simulation mode: no real hardware needed."""
        pass  # Nothing to open

    def _read_simulation(self, n: int) -> dict[int, np.ndarray]:
        """
        Generate n synthetic waveforms per enabled channel.

        Each waveform contains:
        - Gaussian baseline noise
        - Poisson-distributed SPE pulses (Gaussian shaped, σ ≈ 0.4 µs)
        - One trigger pulse at the pre-trigger boundary
        """
        rng        = self._sim_rng
        n_samp     = self._n_samples
        pre        = self._pre_samples
        result     = {}
        timestamps = np.linspace(0.0, n / SAMPLE_RATE_HZ, n, dtype=np.float64)

        t_axis = np.arange(n_samp, dtype=np.float32)  # sample indices

        for ch in self._channels_enabled:
            if ch == PMT_CHANNEL:
                result[ch] = self._sim_pmt_waveforms(n, n_samp, pre, rng)
                continue

            dark_rate     = self._sim_dark_rates.get(ch, SIM_DARK_RATE_HZ)
            spe_amplitude = self._sim_spe_amplitudes.get(ch, SIM_SPE_AMPLITUDE_COUNTS)
            noise_sigma   = SIM_NOISE_SIGMA_COUNTS

            waves = np.zeros((n, n_samp), dtype=np.int16)

            for i in range(n):
                # Baseline noise
                baseline = rng.normal(0.0, noise_sigma, n_samp)

                # Duration of this waveform in seconds
                duration_s = n_samp / SAMPLE_RATE_HZ

                # Expected number of dark pulses in this window
                mean_pulses = dark_rate * duration_s
                n_pulses    = rng.poisson(mean_pulses)

                for _ in range(n_pulses):
                    # Random arrival time anywhere in the window
                    peak_sample = rng.uniform(0, n_samp)
                    # PE multiplicity (1 SPE most common; occasional 2-PE etc.)
                    n_pe = rng.choice([1, 2, 3], p=[0.85, 0.12, 0.03])
                    amp  = n_pe * spe_amplitude * rng.normal(1.0, 0.05)
                    # Gaussian pulse shape from CR200 shaper
                    pulse = amp * np.exp(
                        -0.5 * ((t_axis - peak_sample) / SIM_PULSE_SIGMA_SAMPLES) ** 2
                    )
                    baseline += pulse

                waves[i] = np.clip(
                    baseline + ADC_MIDSCALE, 0, ADC_FULL_SCALE - 1
                ).astype(np.int16)

            result[ch] = waves

        result[-1] = timestamps
        return result

    def _sim_pmt_waveforms(self, n, n_samp, pre, rng) -> np.ndarray:
        """Generate synthetic PMT trigger pulses on ch4."""
        waves = np.zeros((n, n_samp), dtype=np.int16)
        for i in range(n):
            baseline = rng.normal(0.0, SIM_NOISE_SIGMA_COUNTS * 2, n_samp)
            # PMT pulse at the trigger point
            peak = pre
            amp  = SIM_SPE_AMPLITUDE_COUNTS * 5  # PMT much larger
            pulse = amp * np.exp(
                -0.5 * ((np.arange(n_samp) - peak) / 20) ** 2
            )
            baseline += pulse
            waves[i] = np.clip(
                baseline + ADC_MIDSCALE, 0, ADC_FULL_SCALE - 1
            ).astype(np.int16)
        return waves

    # ------------------------------------------------------------------
    # Simulation parameter control
    # ------------------------------------------------------------------

    def sim_set_dark_rate(self, channel: int, rate_hz: float):
        """Set simulated dark count rate for a channel (simulation mode only)."""
        self._sim_dark_rates[channel] = rate_hz

    def sim_set_spe_amplitude(self, channel: int, amplitude_counts: int):
        """Set simulated SPE amplitude for a channel (simulation mode only)."""
        self._sim_spe_amplitudes[channel] = amplitude_counts

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *_):
        self.disconnect()
