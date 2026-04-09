"""
vx2740/gui.py

Standalone PyQt5 GUI for the CAEN VX2740 digitizer.

Launch directly for independent testing:
    python -m vx2740.gui

Or import and embed in the main DAQ window:
    from vx2740.gui import VX2740Window
"""

import sys
import time
import threading
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLabel, QLineEdit, QComboBox, QPushButton, QSpinBox,
    QDoubleSpinBox, QCheckBox, QTextEdit, QTabWidget, QGridLayout,
    QSplitter, QTableWidget, QTableWidgetItem, QHeaderView,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt5.QtGui import QFont

# Optional matplotlib for waveform/spectrum plots
try:
    import matplotlib
    matplotlib.use("Qt5Agg")
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

from .controller import VX2740Controller
from .driver import (
    N_SIPM_CHANNELS, PMT_CHANNEL,
    DEFAULT_PRE_SAMPLES, DEFAULT_POST_SAMPLES, SAMPLE_RATE_HZ,
)


# ---------------------------------------------------------------------------
# Worker signals (thread-safe Qt communication)
# ---------------------------------------------------------------------------

class _Signals(QObject):
    status        = pyqtSignal(str)
    connected     = pyqtSignal(bool, str)          # (success, message)
    test_done     = pyqtSignal(bool, str)
    progress      = pyqtSignal(int, int)           # (acquired, total)
    acquisition_done  = pyqtSignal(object)         # AcquisitionResult
    waveform_ready    = pyqtSignal(object, int)    # (waveform np.ndarray, channel)


class _ConnectWorker(QThread):
    def __init__(self, controller: VX2740Controller, signals: _Signals):
        super().__init__()
        self._ctrl    = controller
        self._signals = signals

    def run(self):
        try:
            self._ctrl.connect()
            idn = self._ctrl.identify()
            self._signals.connected.emit(True, idn)
        except Exception as e:
            self._signals.connected.emit(False, str(e))


class _AcquireWorker(QThread):
    def __init__(self, controller: VX2740Controller, n_waveforms: int,
                 store_waveforms: bool, signals: _Signals):
        super().__init__()
        self._ctrl           = controller
        self._n              = n_waveforms
        self._store_waveforms = store_waveforms
        self._signals        = signals
        self._stop           = threading.Event()

    def stop(self):
        self._stop.set()

    def run(self):
        try:
            self._ctrl.on_progress = lambda a, t: self._signals.progress.emit(a, t)
            result = self._ctrl.run(
                n_waveforms     = self._n,
                store_waveforms = self._store_waveforms,
            )
            self._signals.acquisition_done.emit(result)
        except Exception as e:
            self._signals.status.emit(f"Acquisition error: {e}")
        finally:
            self._ctrl.on_progress = None


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class VX2740Window(QMainWindow):
    """
    Standalone window for the CAEN VX2740 digitizer.

    Tabs:
        Connection    — IP address, mode, connect/disconnect, test
        Channels      — enable/disable, per-channel or global threshold
        Acquisition   — record window, waveform count, trigger mode, start/stop
        Waveforms     — live waveform plot (last acquisition)
        Spectrum      — pulse amplitude histogram (last acquisition)
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("CAEN VX2740 Digitizer Control")
        self.resize(900, 700)

        self._ctrl:    VX2740Controller | None = None
        self._signals: _Signals               = _Signals()
        self._worker:  QThread | None         = None

        self._last_result = None  # most recent AcquisitionResult

        self._build_ui()
        self._connect_signals()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        tabs = QTabWidget()
        tabs.addTab(self._build_connection_tab(), "Connection")
        tabs.addTab(self._build_channels_tab(),   "Channels")
        tabs.addTab(self._build_acquisition_tab(),"Acquisition")
        if HAS_MPL:
            tabs.addTab(self._build_waveform_tab(),  "Waveforms")
            tabs.addTab(self._build_spectrum_tab(),  "Spectrum")

        layout.addWidget(tabs)
        layout.addWidget(self._build_status_log())

    # --- Connection tab ---
    def _build_connection_tab(self) -> QWidget:
        w   = QWidget()
        lay = QVBoxLayout(w)

        box = QGroupBox("Instrument Connection")
        g   = QGridLayout(box)

        g.addWidget(QLabel("IP Address:"), 0, 0)
        self._ip_edit = QLineEdit("192.168.0.1")
        g.addWidget(self._ip_edit, 0, 1)

        g.addWidget(QLabel("Mode:"), 1, 0)
        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["simulation", "hardware"])
        g.addWidget(self._mode_combo, 1, 1)

        btn_row = QHBoxLayout()
        self._connect_btn    = QPushButton("Connect")
        self._disconnect_btn = QPushButton("Disconnect")
        self._test_btn       = QPushButton("Test Connection")
        self._disconnect_btn.setEnabled(False)
        btn_row.addWidget(self._connect_btn)
        btn_row.addWidget(self._disconnect_btn)
        btn_row.addWidget(self._test_btn)
        g.addLayout(btn_row, 2, 0, 1, 2)

        self._status_label = QLabel("Not connected")
        self._status_label.setStyleSheet("color: red; font-weight: bold;")
        g.addWidget(self._status_label, 3, 0, 1, 2)

        lay.addWidget(box)
        lay.addStretch()
        return w

    # --- Channels tab ---
    def _build_channels_tab(self) -> QWidget:
        w   = QWidget()
        lay = QVBoxLayout(w)

        # Threshold mode
        mode_box = QGroupBox("Threshold Mode")
        m_lay    = QHBoxLayout(mode_box)
        self._thresh_mode_combo = QComboBox()
        self._thresh_mode_combo.addItems(["per_channel", "global"])
        self._thresh_mode_combo.currentTextChanged.connect(self._on_threshold_mode_changed)
        m_lay.addWidget(QLabel("Mode:"))
        m_lay.addWidget(self._thresh_mode_combo)
        m_lay.addWidget(QLabel("Global threshold (ADC counts):"))
        self._global_thresh_spin = QSpinBox()
        self._global_thresh_spin.setRange(0, 8191)
        self._global_thresh_spin.setValue(150)
        m_lay.addWidget(self._global_thresh_spin)
        m_lay.addStretch()
        lay.addWidget(mode_box)

        # Per-channel table
        ch_box = QGroupBox("SiPM Channels (ch 0–3) + PMT (ch 4)")
        c_lay  = QVBoxLayout(ch_box)
        self._ch_table = QTableWidget(N_SIPM_CHANNELS + 1, 3)
        self._ch_table.setHorizontalHeaderLabels(["Channel", "Enable", "Threshold (ADC counts)"])
        self._ch_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        self._ch_enable_checks  = []
        self._ch_thresh_spins   = []

        for row in range(N_SIPM_CHANNELS + 1):
            ch   = row
            label = f"ch{ch}" + (" (PMT)" if ch == PMT_CHANNEL else " (SiPM)")
            self._ch_table.setItem(row, 0, QTableWidgetItem(label))

            chk = QCheckBox()
            chk.setChecked(True)
            self._ch_table.setCellWidget(row, 1, chk)
            self._ch_enable_checks.append(chk)

            spin = QSpinBox()
            spin.setRange(0, 8191)
            spin.setValue(150)
            self._ch_table.setCellWidget(row, 2, spin)
            self._ch_thresh_spins.append(spin)

        c_lay.addWidget(self._ch_table)
        lay.addWidget(ch_box)
        return w

    # --- Acquisition tab ---
    def _build_acquisition_tab(self) -> QWidget:
        w   = QWidget()
        lay = QVBoxLayout(w)

        # Record window
        rw_box = QGroupBox("Record Window")
        rw_lay = QGridLayout(rw_box)
        rw_lay.addWidget(QLabel("Pre-trigger (µs):"), 0, 0)
        self._pre_spin = QDoubleSpinBox()
        self._pre_spin.setRange(0.1, 100.0)
        self._pre_spin.setValue(2.0)
        self._pre_spin.setSingleStep(0.5)
        rw_lay.addWidget(self._pre_spin, 0, 1)

        rw_lay.addWidget(QLabel("Post-trigger (µs):"), 1, 0)
        self._post_spin = QDoubleSpinBox()
        self._post_spin.setRange(0.1, 100.0)
        self._post_spin.setValue(10.0)
        self._post_spin.setSingleStep(1.0)
        rw_lay.addWidget(self._post_spin, 1, 1)

        self._samples_label = QLabel(f"Total: {DEFAULT_PRE_SAMPLES + DEFAULT_POST_SAMPLES} samples")
        rw_lay.addWidget(self._samples_label, 2, 0, 1, 2)
        self._pre_spin.valueChanged.connect(self._update_samples_label)
        self._post_spin.valueChanged.connect(self._update_samples_label)
        lay.addWidget(rw_box)

        # Trigger
        trig_box = QGroupBox("Trigger")
        t_lay    = QHBoxLayout(trig_box)
        t_lay.addWidget(QLabel("Trigger source:"))
        self._trig_combo = QComboBox()
        self._trig_combo.addItems(["self", "external"])
        t_lay.addWidget(self._trig_combo)
        t_lay.addStretch()
        lay.addWidget(trig_box)

        # Waveform count
        wf_box = QGroupBox("Acquisition")
        wf_lay = QGridLayout(wf_box)
        wf_lay.addWidget(QLabel("Waveforms to acquire:"), 0, 0)
        self._n_waveforms_spin = QSpinBox()
        self._n_waveforms_spin.setRange(1, 1_000_000)
        self._n_waveforms_spin.setValue(1000)
        wf_lay.addWidget(self._n_waveforms_spin, 0, 1)

        self._store_wf_check = QCheckBox("Store raw waveforms (memory-intensive)")
        self._store_wf_check.setChecked(False)
        wf_lay.addWidget(self._store_wf_check, 1, 0, 1, 2)

        # Progress bar (simple label for now)
        self._progress_label = QLabel("Ready")
        wf_lay.addWidget(self._progress_label, 2, 0, 1, 2)

        btn_row = QHBoxLayout()
        self._start_btn = QPushButton("Start Acquisition")
        self._stop_btn  = QPushButton("Stop")
        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(False)
        btn_row.addWidget(self._start_btn)
        btn_row.addWidget(self._stop_btn)
        wf_lay.addLayout(btn_row, 3, 0, 1, 2)
        lay.addWidget(wf_box)
        lay.addStretch()
        return w

    # --- Waveform plot tab ---
    def _build_waveform_tab(self) -> QWidget:
        w   = QWidget()
        lay = QVBoxLayout(w)

        ctrl_row = QHBoxLayout()
        ctrl_row.addWidget(QLabel("Channel:"))
        self._wf_ch_combo = QComboBox()
        for ch in range(N_SIPM_CHANNELS):
            self._wf_ch_combo.addItem(f"ch{ch}")
        self._wf_ch_combo.currentIndexChanged.connect(self._refresh_waveform_plot)
        ctrl_row.addWidget(self._wf_ch_combo)
        ctrl_row.addWidget(QLabel("Waveform index:"))
        self._wf_idx_spin = QSpinBox()
        self._wf_idx_spin.setRange(0, 0)
        self._wf_idx_spin.valueChanged.connect(self._refresh_waveform_plot)
        ctrl_row.addWidget(self._wf_idx_spin)
        ctrl_row.addStretch()
        lay.addLayout(ctrl_row)

        self._wf_fig    = Figure(figsize=(8, 3))
        self._wf_canvas = FigureCanvas(self._wf_fig)
        self._wf_ax     = self._wf_fig.add_subplot(111)
        self._wf_ax.set_xlabel("Time (µs)")
        self._wf_ax.set_ylabel("ADC counts (baseline subtracted)")
        self._wf_ax.set_title("Waveform")
        lay.addWidget(self._wf_canvas)
        return w

    # --- Spectrum tab ---
    def _build_spectrum_tab(self) -> QWidget:
        w   = QWidget()
        lay = QVBoxLayout(w)

        ctrl_row = QHBoxLayout()
        ctrl_row.addWidget(QLabel("Channel:"))
        self._spec_ch_combo = QComboBox()
        for ch in range(N_SIPM_CHANNELS):
            self._spec_ch_combo.addItem(f"ch{ch}")
        self._spec_ch_combo.currentIndexChanged.connect(self._refresh_spectrum_plot)
        ctrl_row.addWidget(self._spec_ch_combo)
        ctrl_row.addWidget(QLabel("Bins:"))
        self._spec_bins_spin = QSpinBox()
        self._spec_bins_spin.setRange(10, 1000)
        self._spec_bins_spin.setValue(100)
        self._spec_bins_spin.valueChanged.connect(self._refresh_spectrum_plot)
        ctrl_row.addWidget(self._spec_bins_spin)
        ctrl_row.addStretch()
        lay.addLayout(ctrl_row)

        self._spec_fig    = Figure(figsize=(8, 3))
        self._spec_canvas = FigureCanvas(self._spec_fig)
        self._spec_ax     = self._spec_fig.add_subplot(111)
        self._spec_ax.set_xlabel("Amplitude (ADC counts)")
        self._spec_ax.set_ylabel("Counts")
        self._spec_ax.set_title("Pulse amplitude spectrum")
        lay.addWidget(self._spec_canvas)
        return w

    # --- Status log ---
    def _build_status_log(self) -> QWidget:
        box = QGroupBox("Status Log")
        lay = QVBoxLayout(box)
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumHeight(120)
        self._log.setFont(QFont("Courier", 9))
        lay.addWidget(self._log)
        return box

    # ------------------------------------------------------------------
    # Signal wiring
    # ------------------------------------------------------------------

    def _connect_signals(self):
        self._connect_btn.clicked.connect(self._on_connect)
        self._disconnect_btn.clicked.connect(self._on_disconnect)
        self._test_btn.clicked.connect(self._on_test)
        self._start_btn.clicked.connect(self._on_start_acquisition)
        self._stop_btn.clicked.connect(self._on_stop_acquisition)

        self._signals.status.connect(self._log_message)
        self._signals.connected.connect(self._on_connect_result)
        self._signals.progress.connect(self._on_progress)
        self._signals.acquisition_done.connect(self._on_acquisition_done)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_connect(self):
        ip   = self._ip_edit.text().strip()
        mode = self._mode_combo.currentText()
        self._ctrl = VX2740Controller(address=ip, mode=mode)
        self._log_message(f"Connecting to {ip} ({mode} mode)...")
        self._connect_btn.setEnabled(False)

        worker = _ConnectWorker(self._ctrl, self._signals)
        worker.start()
        self._worker = worker

    def _on_connect_result(self, success: bool, message: str):
        self._connect_btn.setEnabled(True)
        if success:
            self._status_label.setText(f"Connected: {message}")
            self._status_label.setStyleSheet("color: green; font-weight: bold;")
            self._disconnect_btn.setEnabled(True)
            self._start_btn.setEnabled(True)
            self._log_message(f"Connected: {message}")
        else:
            self._status_label.setText("Connection failed")
            self._status_label.setStyleSheet("color: red; font-weight: bold;")
            self._log_message(f"Connection failed: {message}")
            self._ctrl = None

    def _on_disconnect(self):
        if self._ctrl is not None:
            try:
                self._ctrl.disconnect()
            except Exception as e:
                self._log_message(f"Disconnect error: {e}")
            self._ctrl = None
        self._status_label.setText("Not connected")
        self._status_label.setStyleSheet("color: red; font-weight: bold;")
        self._disconnect_btn.setEnabled(False)
        self._start_btn.setEnabled(False)
        self._log_message("Disconnected.")

    def _on_test(self):
        config = {
            "address": self._ip_edit.text().strip(),
            "mode":    self._mode_combo.currentText(),
        }
        self._log_message("Testing connection...")

        class _TestWorker(QThread):
            done = pyqtSignal(bool, str)
            def run(self_):
                ok, msg = VX2740Controller.test(config)
                self_.done.emit(ok, msg)

        w = _TestWorker(self)
        w.done.connect(lambda ok, msg: self._log_message(
            f"Test {'OK' if ok else 'FAILED'}: {msg}"
        ))
        w.start()
        self._worker = w

    def _on_start_acquisition(self):
        if self._ctrl is None:
            return

        # Push config from UI to controller
        self._ctrl.configure_record_window(
            pre_us  = self._pre_spin.value(),
            post_us = self._post_spin.value(),
        )
        self._ctrl.configure_trigger(self._trig_combo.currentText())

        # Build channel + threshold config from table
        sipm_chs   = []
        thresholds = {}
        for row in range(N_SIPM_CHANNELS):
            if self._ch_enable_checks[row].isChecked():
                sipm_chs.append(row)
                thresholds[row] = self._ch_thresh_spins[row].value()

        thresh_mode = self._thresh_mode_combo.currentText()
        self._ctrl.configure_channels(
            sipm_channels     = sipm_chs,
            thresholds        = thresholds,
            threshold_mode    = thresh_mode,
            global_threshold  = self._global_thresh_spin.value(),
            include_pmt       = self._ch_enable_checks[PMT_CHANNEL].isChecked(),
        )

        n = self._n_waveforms_spin.value()
        store = self._store_wf_check.isChecked()

        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._log_message(f"Starting acquisition: {n} waveforms...")

        acq_worker = _AcquireWorker(self._ctrl, n, store, self._signals)
        acq_worker.start()
        self._worker = acq_worker

    def _on_stop_acquisition(self):
        if self._worker is not None and hasattr(self._worker, "stop"):
            self._worker.stop()
        self._stop_btn.setEnabled(False)
        self._log_message("Stop requested.")

    def _on_progress(self, acquired: int, total: int):
        pct = 100 * acquired // total
        self._progress_label.setText(f"Acquired {acquired}/{total} ({pct}%)")

    def _on_acquisition_done(self, result):
        self._last_result = result
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)

        total_pulses = sum(len(result.amplitudes.get(ch, [])) for ch in result.channel_ids)
        self._progress_label.setText(
            f"Done — {result.n_waveforms} waveforms, {total_pulses} pulses found"
        )
        self._log_message(
            f"Acquisition complete: {result.n_waveforms} waveforms, "
            f"{total_pulses} total pulses across {len(result.channel_ids)} channels."
        )

        if HAS_MPL:
            self._refresh_waveform_plot()
            self._refresh_spectrum_plot()

    def _on_threshold_mode_changed(self, mode: str):
        is_global = (mode == "global")
        self._global_thresh_spin.setEnabled(is_global)
        for spin in self._ch_thresh_spins:
            spin.setEnabled(not is_global)

    def _update_samples_label(self):
        pre  = int(round(self._pre_spin.value()  * SAMPLE_RATE_HZ * 1e-6))
        post = int(round(self._post_spin.value() * SAMPLE_RATE_HZ * 1e-6))
        self._samples_label.setText(
            f"Total: {pre + post} samples  ({pre} pre + {post} post)"
        )

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def _refresh_waveform_plot(self):
        if not HAS_MPL or self._last_result is None:
            return
        result  = self._last_result
        ch      = self._wf_ch_combo.currentIndex()
        waveforms = result.waveforms.get(ch)
        if waveforms is None or len(waveforms) == 0:
            return

        max_idx = len(waveforms) - 1
        self._wf_idx_spin.setMaximum(max_idx)
        idx = min(self._wf_idx_spin.value(), max_idx)

        wave = waveforms[idx].astype(np.float32)
        wave -= wave[:DEFAULT_PRE_SAMPLES].mean()  # baseline subtract
        t_us = np.arange(len(wave)) / (SAMPLE_RATE_HZ * 1e-6)

        self._wf_ax.clear()
        self._wf_ax.plot(t_us, wave, lw=0.8)
        self._wf_ax.axvline(DEFAULT_PRE_SAMPLES / (SAMPLE_RATE_HZ * 1e-6),
                            color="red", ls="--", lw=0.8, label="trigger")
        self._wf_ax.set_xlabel("Time (µs)")
        self._wf_ax.set_ylabel("ADC counts (baseline sub.)")
        self._wf_ax.set_title(f"Waveform — ch{ch}, index {idx}")
        self._wf_ax.legend(fontsize=8)
        self._wf_fig.tight_layout()
        self._wf_canvas.draw()

    def _refresh_spectrum_plot(self):
        if not HAS_MPL or self._last_result is None:
            return
        result = self._last_result
        ch     = self._spec_ch_combo.currentIndex()
        amps   = result.amplitudes.get(ch)
        if amps is None or len(amps) == 0:
            self._log_message(f"No pulses found on ch{ch} for spectrum.")
            return

        bins = self._spec_bins_spin.value()
        self._spec_ax.clear()
        self._spec_ax.hist(amps, bins=bins, color="steelblue", edgecolor="none")
        self._spec_ax.set_xlabel("Amplitude (ADC counts)")
        self._spec_ax.set_ylabel("Counts")
        self._spec_ax.set_title(f"Pulse spectrum — ch{ch}  (N={len(amps)} pulses)")
        self._spec_fig.tight_layout()
        self._spec_canvas.draw()

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _log_message(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        self._log.append(f"[{ts}] {msg}")

    def closeEvent(self, event):
        if self._ctrl is not None:
            try:
                self._ctrl.disconnect()
            except Exception:
                pass
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    app = QApplication(sys.argv)
    win = VX2740Window()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
