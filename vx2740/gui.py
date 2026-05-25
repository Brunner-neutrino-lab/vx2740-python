"""
vx2740/gui.py

NiceGUI control panel for the CAEN VX2740 digitizer.

Same two-mode pattern as b2987b/gui.py:

  - Standalone (`python -m vx2740.gui`): opens a browser served by
    NiceGUI with a Connection panel that creates and owns its own
    VX2740Controller. Useful for digitizer bring-up without the rest
    of the DAQ.

  - Embedded (`build_page(get_controller=..., show_connection=False)`):
    called from a parent NiceGUI app (the ETS DAQ web shell). The
    parent passes a getter that returns the shared controller; this
    panel hides its Connection card and drives the parent's
    controller, so configuration changes here apply to the same
    instrument used by the rest of the DAQ.

Feature parity with the previous PyQt5 GUI: Connection, Channels
(threshold mode + per-channel enable/threshold), Acquisition
(record window + trigger + start/stop), Waveforms (one-trace plot),
Spectrum (pulse-amplitude histogram).
"""

from __future__ import annotations

import asyncio
import time
from typing import Callable, Optional

import numpy as np
from nicegui import ui

from .controller import VX2740Controller
from .driver import (
    N_SIPM_CHANNELS, PMT_CHANNEL,
    DEFAULT_PRE_SAMPLES, DEFAULT_POST_SAMPLES, SAMPLE_RATE_HZ,
)


# ---------------------------------------------------------------------------
# Style — matches xsphere/DAQ "register" theme
# ---------------------------------------------------------------------------

_CSS = """
:root {
  --bg:#11151c; --panel:#1b2230; --panel2:#232c3d;
  --fg:#dde3ee; --mut:#8a93a6;
  --ok:#3fb950; --warn:#d29922; --bad:#f85149; --acc:#58a6ff;
  --line:#2d3648;
}
html, body, .nicegui-content { background:var(--bg) !important; color:var(--fg);
  font:14px/1.45 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif; margin:0; }
.pill { padding:.15rem .55rem; border-radius:999px; font-size:.78rem;
  font-weight:600; white-space:nowrap; display:inline-flex; align-items:center; gap:.3rem; }
.pill.ok   { background:rgba(63,185,80,.18);  color:var(--ok); }
.pill.bad  { background:rgba(248,81,73,.18);  color:var(--bad); }
.pill.warn { background:rgba(210,153,34,.18); color:var(--warn); }
.pill.mut  { background:rgba(138,147,166,.15);color:var(--mut); }
.q-card, .vx-card {
  background:var(--panel) !important; color:var(--fg) !important;
  border:1px solid var(--line); border-radius:10px;
  box-shadow:none !important; padding:.55rem .85rem .7rem !important;
}
.vx-card h2 { font-size:.92rem; margin:.05rem 0 .45rem; color:var(--acc);
  font-weight:600; letter-spacing:.3px; }
.q-btn { background:var(--panel2) !important; color:var(--fg) !important;
  border:1px solid var(--line) !important; border-radius:6px !important;
  box-shadow:none !important; padding:.18rem .65rem !important;
  min-height:32px !important; text-transform:none !important; }
.q-btn:hover { border-color:var(--acc) !important; }
.q-btn[data-q-color="primary"], .q-btn.bg-primary {
  background:var(--acc) !important; color:#08111f !important;
  border-color:var(--acc) !important; font-weight:600 !important; }
.q-btn[data-q-color="negative"], .q-btn.bg-negative {
  background:transparent !important; color:var(--bad) !important;
  border-color:var(--bad) !important; }
.q-field__control, .q-field--filled .q-field__control {
  background:var(--panel2) !important; border:1px solid var(--line) !important;
  border-radius:6px !important; min-height:32px !important; color:var(--fg) !important; }
.q-field__label, .q-field__native, .q-field input { color:var(--fg) !important; }
.q-field__label { color:var(--mut) !important; }
.q-field--filled .q-field__control:before,
.q-field--filled .q-field__control:after { display:none !important; }
.q-tab { color:var(--mut) !important; text-transform:none !important; }
.q-tab--active { color:var(--acc) !important; }
.q-tab__indicator { background:var(--acc) !important; }
.q-log, .nicegui-log { background:var(--panel2) !important; color:var(--fg) !important;
  border:1px solid var(--line); border-radius:6px;
  font-family:ui-monospace,Menlo,Consolas,monospace; font-size:.82rem; }
.vx-ch-row { display:grid; grid-template-columns: 5rem 4rem 9rem; gap:.5rem;
  align-items:center; padding:.2rem 0; border-top:1px solid var(--line); }
.vx-ch-row:first-of-type { border-top:0; }
.num { font-variant-numeric:tabular-nums; }
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _in_thread(fn, *a, **kw):
    return await asyncio.to_thread(fn, *a, **kw)


def _samples_label(pre_us: float, post_us: float) -> str:
    n = int(round(pre_us  * SAMPLE_RATE_HZ * 1e-6)) + \
        int(round(post_us * SAMPLE_RATE_HZ * 1e-6))
    return f"{n} samples total ({pre_us + post_us:.1f} µs at {SAMPLE_RATE_HZ/1e6:.0f} MS/s)"


# ===========================================================================
# build_page — reusable GUI
# ===========================================================================

def build_page(get_controller: Optional[Callable[[], Optional[VX2740Controller]]] = None,
               *, show_connection: Optional[bool] = None) -> None:
    """
    Render the VX2740 control panel into the current NiceGUI container.

    Parameters
    ----------
    get_controller : callable returning VX2740Controller | None, optional
        Returns the current shared controller. If `None`, the panel
        manages its own controller via the Connection card.
    show_connection : bool, optional
        Whether to render the Connection card. Defaults to True for
        standalone (no `get_controller`), False for embedded.
    """
    if show_connection is None:
        show_connection = (get_controller is None)

    _own = {"ctrl": None, "result": None}

    if get_controller is None:
        def get_controller():
            return _own["ctrl"]

    # --- log + helpers ----------------------------------------------------

    log = ui.log(max_lines=160).classes("h-32 w-full")
    def log_msg(s: str): log.push(f"[{time.strftime('%H:%M:%S')}] {s}")

    def ensure_ctrl() -> Optional[VX2740Controller]:
        c = get_controller()
        if c is None:
            log_msg("not connected" + (" — use the Connection tab"
                                       if show_connection else
                                       " — connect on the DAQ's Connections tab"))
            return None
        return c

    # --- Tabs -------------------------------------------------------------

    with ui.tabs().classes("w-full") as tabs:
        t_conn  = ui.tab("connection") if show_connection else None
        t_chans = ui.tab("channels")
        t_acq   = ui.tab("acquisition")
        t_wf    = ui.tab("waveforms")
        t_spec  = ui.tab("spectrum")

    initial = t_conn if t_conn is not None else t_chans
    with ui.tab_panels(tabs, value=initial).classes("w-full"):

        # ----------- Connection (standalone only) -----------
        if t_conn is not None:
            with ui.tab_panel(t_conn):
                with ui.card().classes("vx-card"):
                    ui.html("<h2>digitizer connection</h2>")
                    addr_in = ui.input(label="address (IP)",
                                       value="172.16.0.51").classes("w-72 num")
                    mode_in = ui.select(["simulation", "hardware"],
                                        value="simulation",
                                        label="mode").classes("w-40")
                    conn_pill = ui.html('<span class="pill mut">disconnected</span>')

                    def set_pill(text: str, cls: str):
                        conn_pill.content = f'<span class="pill {cls}">{text}</span>'

                    async def do_connect():
                        c = VX2740Controller(address=addr_in.value.strip(),
                                             mode=mode_in.value)
                        set_pill("connecting…", "warn")
                        log_msg(f"connecting to dig2://{addr_in.value.strip()} ({mode_in.value})…")
                        try:
                            await _in_thread(c.connect)
                            _own["ctrl"] = c
                            set_pill(f"OK — {c.identify()[:60]}", "ok")
                            log_msg(f"connected: {c.identify()}")
                        except Exception as e:
                            set_pill(f"FAIL: {type(e).__name__}", "bad")
                            log_msg(f"connect FAIL: {type(e).__name__}: {e}")

                    async def do_disconnect():
                        c = _own["ctrl"]
                        if c is None:
                            return
                        try:
                            await _in_thread(c.disconnect)
                        except Exception as e:
                            log_msg(f"disconnect warn: {e}")
                        _own["ctrl"] = None
                        set_pill("disconnected", "mut")
                        log_msg("disconnected")

                    with ui.row().classes("mt-1 gap-2"):
                        ui.button("connect",    on_click=do_connect).props("color=primary")
                        ui.button("disconnect", on_click=do_disconnect).props("color=negative flat")

        # ----------- Channels tab -----------
        per_ch_enable: list = []
        per_ch_thresh: list = []
        with ui.tab_panel(t_chans):
            with ui.card().classes("vx-card"):
                ui.html("<h2>threshold mode</h2>")
                with ui.row().classes("items-center gap-3"):
                    thresh_mode = ui.select(["per_channel", "global"],
                                             value="per_channel",
                                             label="mode").classes("w-40")
                    global_thresh = ui.number(label="global (ADC counts)",
                                              value=150, step=1).classes("w-40 num")
                global_thresh.bind_visibility_from(thresh_mode, "value",
                                                    lambda v: v == "global")

            with ui.card().classes("vx-card"):
                ui.html("<h2>SiPM channels (0–3) + PMT (4)</h2>")
                # Header row
                ui.html('<div class="vx-ch-row" style="font-weight:600">'
                        '<span>channel</span><span>enable</span><span>threshold</span></div>')
                for ch in range(N_SIPM_CHANNELS + 1):
                    label = f"ch{ch}" + (" (PMT)" if ch == PMT_CHANNEL else " (SiPM)")
                    with ui.row().classes("vx-ch-row"):
                        ui.label(label).classes("num text-sm")
                        chk = ui.switch(value=True)
                        per_ch_enable.append(chk)
                        spin = ui.number(value=150, step=1).classes("w-40 num")
                        # Per-channel thresholds disabled when global mode
                        spin.bind_enabled_from(thresh_mode, "value",
                                                lambda v: v == "per_channel")
                        per_ch_thresh.append(spin)

        # ----------- Acquisition tab -----------
        with ui.tab_panel(t_acq):
            with ui.row().classes("w-full gap-3 items-start"):
                with ui.card().classes("vx-card"):
                    ui.html("<h2>record window</h2>")
                    pre_us  = ui.number(label="pre-trigger (µs)",  value=2.0,  step=0.5).classes("w-36 num")
                    post_us = ui.number(label="post-trigger (µs)", value=10.0, step=1.0).classes("w-36 num")
                    samples_lbl = ui.label(_samples_label(2.0, 10.0)).classes("num text-xs text-gray-400")
                    def _update_samples():
                        samples_lbl.text = _samples_label(float(pre_us.value), float(post_us.value))
                    pre_us.on("update:model-value", lambda _e: _update_samples())
                    post_us.on("update:model-value", lambda _e: _update_samples())

                with ui.card().classes("vx-card"):
                    ui.html("<h2>trigger</h2>")
                    trig_mode = ui.select(["self", "external", "software"],
                                          value="external",
                                          label="source").classes("w-40")

                with ui.card().classes("vx-card"):
                    ui.html("<h2>acquisition</h2>")
                    n_wf = ui.number(label="waveforms", value=1000, step=100).classes("w-36 num")
                    store_wf = ui.switch("store raw waveforms (memory-heavy)", value=False)
                    timeout_s = ui.number(label="timeout (s)", value=60.0, step=10).classes("w-36 num")
                    progress_lbl = ui.label("ready").classes("num text-sm")

                    async def push_config():
                        c = ensure_ctrl()
                        if c is None: return None
                        # Record window
                        c.configure_record_window(pre_us=float(pre_us.value),
                                                  post_us=float(post_us.value))
                        # Channels + thresholds
                        sipm_chs = [i for i in range(N_SIPM_CHANNELS)
                                    if per_ch_enable[i].value]
                        include_pmt = bool(per_ch_enable[PMT_CHANNEL].value)
                        if thresh_mode.value == "global":
                            c.configure_channels(
                                sipm_channels    = sipm_chs,
                                threshold_mode   = "global",
                                global_threshold = int(global_thresh.value),
                                include_pmt      = include_pmt,
                            )
                        else:
                            thresholds = {i: int(per_ch_thresh[i].value)
                                          for i in sipm_chs}
                            c.configure_channels(
                                sipm_channels    = sipm_chs,
                                thresholds       = thresholds,
                                threshold_mode   = "per_channel",
                                include_pmt      = include_pmt,
                            )
                        c.configure_trigger(mode=str(trig_mode.value))
                        return c

                    async def run_acq():
                        c = await push_config()
                        if c is None: return
                        n = int(n_wf.value)
                        store = bool(store_wf.value)
                        is_sw = (str(trig_mode.value) == "software")
                        progress_lbl.text = f"acquiring 0 / {n}…"
                        log_msg(f"acquire {n} waveforms (trigger={trig_mode.value}, store={store})")
                        # Progress callback
                        def _on_prog(done, total):
                            progress_lbl.text = f"acquiring {done} / {total}…"
                        c.on_progress = _on_prog

                        trigger_task = None
                        try:
                            if is_sw:
                                # In software-trigger mode, run() would block forever
                                # because no events fire on their own. Decompose into
                                # arm + parallel-trigger-fire + acquire + disarm so the
                                # acquire read loop sees events arriving from a side
                                # task that calls sendswtrigger N times.
                                await _in_thread(c.arm)
                                async def _fire():
                                    # Tiny lead so acquire is in its read loop first
                                    await asyncio.sleep(0.05)
                                    for _ in range(n):
                                        await _in_thread(c.send_software_trigger)
                                        await asyncio.sleep(0.005)
                                trigger_task = asyncio.create_task(_fire())
                                try:
                                    result = await _in_thread(c.acquire, n,
                                                               1000, store,
                                                               float(timeout_s.value))
                                finally:
                                    await _in_thread(c.disarm)
                            else:
                                result = await _in_thread(c.run, n,
                                                           1000, store,
                                                           float(timeout_s.value))
                            _own["result"] = result
                            progress_lbl.text = (f"done — {result.n_waveforms} waveforms, "
                                                  f"{sum(len(result.amplitudes.get(ch, [])) for ch in result.channel_ids)} pulses")
                            log_msg(progress_lbl.text)
                            _refresh_wf_controls()
                            _refresh_plots()
                        except Exception as e:
                            progress_lbl.text = "FAIL"
                            log_msg(f"acquire FAIL: {type(e).__name__}: {e}")
                        finally:
                            c.on_progress = None
                            if trigger_task is not None and not trigger_task.done():
                                trigger_task.cancel()

                    def send_sw_trig():
                        c = ensure_ctrl()
                        if c is None: return
                        try:
                            c.send_software_trigger()
                            log_msg("software trigger sent")
                        except Exception as e:
                            log_msg(f"software trigger FAIL: {type(e).__name__}: {e}")

                    with ui.row().classes("gap-2 mt-1"):
                        ui.button("apply config", on_click=push_config)
                        ui.button("run acquisition", on_click=run_acq).props("color=primary")
                        ui.button("send SW trigger", on_click=send_sw_trig)

        # ----------- Waveforms tab -----------
        wf_plot = None
        wf_ax   = None
        wf_ch_sel = None
        wf_idx    = None
        with ui.tab_panel(t_wf):
            with ui.card().classes("vx-card w-full"):
                ui.html("<h2>waveform viewer</h2>")
                with ui.row().classes("items-center gap-2"):
                    wf_ch_sel = ui.select([f"ch{i}" for i in range(N_SIPM_CHANNELS)],
                                           value="ch0", label="channel").classes("w-32")
                    wf_idx = ui.number(label="waveform #", value=0,
                                       step=1, min=0).classes("w-32 num")
                wf_plot = ui.matplotlib(figsize=(9, 3.2)).classes("w-full")
                wf_ax   = wf_plot.figure.add_subplot(111)
                wf_ax.set_xlabel("time (µs)"); wf_ax.set_ylabel("ADC counts (baseline-subtracted)")
                wf_ax.grid(True, alpha=.3); wf_plot.figure.tight_layout()
                wf_ch_sel.on("update:model-value", lambda _e: _refresh_wf_plot())
                wf_idx.on("update:model-value",    lambda _e: _refresh_wf_plot())

        def _refresh_wf_controls():
            r = _own["result"]
            if r is None: return
            # Set the upper bound of the waveform index
            max_idx = max(0, r.n_waveforms - 1)
            wf_idx.props(f"max={max_idx}")
            if wf_idx.value > max_idx:
                wf_idx.value = max_idx

        def _refresh_wf_plot():
            r = _own["result"]
            if r is None or wf_ax is None: return
            ch_str = wf_ch_sel.value
            try:
                ch = int(ch_str.replace("ch", ""))
            except Exception:
                return
            waves = r.waveforms.get(ch)
            if waves is None or len(waves) == 0:
                wf_ax.clear()
                wf_ax.text(0.5, 0.5,
                           "no stored waveforms — enable 'store raw waveforms' and re-run",
                           ha="center", va="center", color="#8a93a6",
                           transform=wf_ax.transAxes)
                wf_plot.update()
                return
            i = int(min(int(wf_idx.value), len(waves) - 1))
            w = np.asarray(waves[i], dtype=np.float64)
            # Baseline-subtract using first ~25 % of samples
            n = len(w)
            base = w[:max(1, n // 4)].mean()
            t_us = np.arange(n) / SAMPLE_RATE_HZ * 1e6
            wf_ax.clear()
            wf_ax.plot(t_us, w - base, lw=1, color="#58a6ff")
            wf_ax.axvline(float(np.asarray(r.amplitudes.get(ch, [0]))[0] if False else 0),
                          color="#8a93a6", alpha=0.0)
            wf_ax.set_xlabel("time (µs)")
            wf_ax.set_ylabel("ADC counts (baseline-subtracted)")
            wf_ax.set_title(f"ch{ch}  waveform #{i}")
            wf_ax.grid(True, alpha=.3)
            wf_plot.figure.tight_layout()
            wf_plot.update()

        # ----------- Spectrum tab -----------
        spec_plot = None
        spec_ax   = None
        spec_ch_sel = None
        spec_bins   = None
        with ui.tab_panel(t_spec):
            with ui.card().classes("vx-card w-full"):
                ui.html("<h2>pulse-amplitude spectrum</h2>")
                with ui.row().classes("items-center gap-2"):
                    spec_ch_sel = ui.select([f"ch{i}" for i in range(N_SIPM_CHANNELS)],
                                             value="ch0", label="channel").classes("w-32")
                    spec_bins   = ui.number(label="bins", value=100, step=10).classes("w-32 num")
                spec_plot = ui.matplotlib(figsize=(9, 3.2)).classes("w-full")
                spec_ax   = spec_plot.figure.add_subplot(111)
                spec_ax.set_xlabel("amplitude (ADC counts above baseline)")
                spec_ax.set_ylabel("counts")
                spec_ax.grid(True, alpha=.3); spec_plot.figure.tight_layout()
                spec_ch_sel.on("update:model-value", lambda _e: _refresh_spec_plot())
                spec_bins.on  ("update:model-value", lambda _e: _refresh_spec_plot())

        def _refresh_spec_plot():
            r = _own["result"]
            if r is None or spec_ax is None: return
            try:
                ch = int(spec_ch_sel.value.replace("ch", ""))
            except Exception:
                return
            amps = np.asarray(r.amplitudes.get(ch, []), dtype=np.float64)
            spec_ax.clear()
            if amps.size == 0:
                spec_ax.text(0.5, 0.5, "no pulses found — try lowering threshold",
                             ha="center", va="center", color="#8a93a6",
                             transform=spec_ax.transAxes)
            else:
                spec_ax.hist(amps, bins=int(spec_bins.value),
                             color="#58a6ff", edgecolor="#11151c", linewidth=0.5)
                spec_ax.set_xlabel("amplitude (ADC counts above baseline)")
                spec_ax.set_ylabel("counts")
                spec_ax.set_title(f"ch{ch}  ({amps.size} pulses)")
            spec_ax.grid(True, alpha=.3)
            spec_plot.figure.tight_layout()
            spec_plot.update()

        def _refresh_plots():
            _refresh_wf_plot()
            _refresh_spec_plot()


# ---------------------------------------------------------------------------
# Standalone entry — `python -m vx2740.gui`
# ---------------------------------------------------------------------------

def main():
    import argparse
    p = argparse.ArgumentParser(description="CAEN VX2740 web GUI")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8767)
    args = p.parse_args()

    @ui.page("/")
    def index():
        ui.add_head_html(f"<style>{_CSS}</style>")
        ui.dark_mode().enable()
        with ui.element("header").style(
            "display:flex;align-items:center;gap:.8rem;"
            "padding:.55rem 1rem;background:var(--panel);"
            "border-bottom:1px solid var(--line);position:sticky;top:0;z-index:5"
        ):
            ui.html("<h1 style='font-size:1.05rem;font-weight:600;margin:0'>"
                    "VX2740 · digitizer</h1>")
        build_page()

    ui.run(host=args.host, port=args.port, reload=False,
           title="VX2740 Digitizer", show=False)


if __name__ in {"__main__", "__mp_main__"}:
    main()
