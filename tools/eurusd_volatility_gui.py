#!/usr/bin/env python3
"""
================================================================================
  TRD QUANTITATIVE TERMINAL: EUR/USD TIME-INVARIANT VOLATILITY CHART GUI
================================================================================
Interactive Desktop GUI application built with Tkinter + Matplotlib.
Renders EUR/USD tick data into non-time-based (pure volatility) representations:
  1. Range Bars: Constant price range (H - L = constant pips)
  2. Renko Bricks: Constant momentum bricks with 2-brick reversals
  3. Cumulative Volatility Path: Price plotted against Total Variation Traversed

Eliminates the chronological time component to expose pure volatility structure,
institutional absorption, and momentum expansion without time dilation noise.
================================================================================
"""

from __future__ import annotations
import os
import sys
import glob
import time
from pathlib import Path
from typing import Tuple, Optional, List

import numpy as np
import pandas as pd
import numba

import tkinter as tk
from tkinter import ttk, messagebox

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.collections import LineCollection, PolyCollection
import matplotlib.ticker as ticker


PIP_SIZE = 0.0001
DATA_DIR = Path(__file__).resolve().parent / "eurusd_weekly_data"


# ==============================================================================
# NUMBA ACCELERATED VOLATILITY BAR BUILDERS (SUB-MILLISECOND EXECUTION)
# ==============================================================================
@numba.njit(fastmath=True)
def numba_build_range_bars(prices: np.ndarray, timestamps: np.ndarray, r_val: float):
    n = len(prices)
    max_bars = n
    opens = np.empty(max_bars, dtype=np.float64)
    highs = np.empty(max_bars, dtype=np.float64)
    lows = np.empty(max_bars, dtype=np.float64)
    closes = np.empty(max_bars, dtype=np.float64)
    start_ts = np.empty(max_bars, dtype=np.int64)
    end_ts = np.empty(max_bars, dtype=np.int64)
    tick_cts = np.empty(max_bars, dtype=np.int64)

    bar_cnt = 0
    cur_open = prices[0]
    cur_high = prices[0]
    cur_low = prices[0]
    cur_start_t = timestamps[0]
    cur_ticks = 0

    for i in range(n):
        p = prices[i]
        cur_ticks += 1
        if p > cur_high:
            cur_high = p
        if p < cur_low:
            cur_low = p

        if (cur_high - cur_low) >= r_val:
            cur_close = p
            opens[bar_cnt] = cur_open
            highs[bar_cnt] = cur_high
            lows[bar_cnt] = cur_low
            closes[bar_cnt] = cur_close
            start_ts[bar_cnt] = cur_start_t
            end_ts[bar_cnt] = timestamps[i]
            tick_cts[bar_cnt] = cur_ticks
            bar_cnt += 1

            cur_open = cur_close
            cur_high = cur_close
            cur_low = cur_close
            cur_start_t = timestamps[i]
            cur_ticks = 0

    return (
        opens[:bar_cnt],
        highs[:bar_cnt],
        lows[:bar_cnt],
        closes[:bar_cnt],
        start_ts[:bar_cnt],
        end_ts[:bar_cnt],
        tick_cts[:bar_cnt],
    )


@numba.njit(fastmath=True)
def numba_build_renko_bars(prices: np.ndarray, timestamps: np.ndarray, brick_size: float):
    n = len(prices)
    max_bars = n
    opens = np.empty(max_bars, dtype=np.float64)
    highs = np.empty(max_bars, dtype=np.float64)
    lows = np.empty(max_bars, dtype=np.float64)
    closes = np.empty(max_bars, dtype=np.float64)
    start_ts = np.empty(max_bars, dtype=np.int64)
    end_ts = np.empty(max_bars, dtype=np.int64)
    directions = np.empty(max_bars, dtype=np.int8)

    bar_cnt = 0
    ref_price = prices[0]
    cur_dir = 0
    start_t = timestamps[0]

    for i in range(1, n):
        p = prices[i]
        diff = p - ref_price

        if cur_dir == 0:
            if diff >= brick_size:
                opens[bar_cnt] = ref_price
                closes[bar_cnt] = ref_price + brick_size
                highs[bar_cnt] = closes[bar_cnt]
                lows[bar_cnt] = opens[bar_cnt]
                start_ts[bar_cnt] = start_t
                end_ts[bar_cnt] = timestamps[i]
                directions[bar_cnt] = 1
                ref_price = closes[bar_cnt]
                cur_dir = 1
                bar_cnt += 1
                start_t = timestamps[i]
            elif diff <= -brick_size:
                opens[bar_cnt] = ref_price
                closes[bar_cnt] = ref_price - brick_size
                highs[bar_cnt] = opens[bar_cnt]
                lows[bar_cnt] = closes[bar_cnt]
                start_ts[bar_cnt] = start_t
                end_ts[bar_cnt] = timestamps[i]
                directions[bar_cnt] = -1
                ref_price = closes[bar_cnt]
                cur_dir = -1
                bar_cnt += 1
                start_t = timestamps[i]
        elif cur_dir == 1:
            if diff >= brick_size:
                opens[bar_cnt] = ref_price
                closes[bar_cnt] = ref_price + brick_size
                highs[bar_cnt] = closes[bar_cnt]
                lows[bar_cnt] = opens[bar_cnt]
                start_ts[bar_cnt] = start_t
                end_ts[bar_cnt] = timestamps[i]
                directions[bar_cnt] = 1
                ref_price = closes[bar_cnt]
                bar_cnt += 1
                start_t = timestamps[i]
            elif diff <= -2.0 * brick_size:
                opens[bar_cnt] = ref_price - brick_size
                closes[bar_cnt] = ref_price - 2.0 * brick_size
                highs[bar_cnt] = opens[bar_cnt]
                lows[bar_cnt] = closes[bar_cnt]
                start_ts[bar_cnt] = start_t
                end_ts[bar_cnt] = timestamps[i]
                directions[bar_cnt] = -1
                ref_price = closes[bar_cnt]
                cur_dir = -1
                bar_cnt += 1
                start_t = timestamps[i]
        elif cur_dir == -1:
            if diff <= -brick_size:
                opens[bar_cnt] = ref_price
                closes[bar_cnt] = ref_price - brick_size
                highs[bar_cnt] = opens[bar_cnt]
                lows[bar_cnt] = closes[bar_cnt]
                start_ts[bar_cnt] = start_t
                end_ts[bar_cnt] = timestamps[i]
                directions[bar_cnt] = -1
                ref_price = closes[bar_cnt]
                bar_cnt += 1
                start_t = timestamps[i]
            elif diff >= 2.0 * brick_size:
                opens[bar_cnt] = ref_price + brick_size
                closes[bar_cnt] = ref_price + 2.0 * brick_size
                highs[bar_cnt] = closes[bar_cnt]
                lows[bar_cnt] = opens[bar_cnt]
                start_ts[bar_cnt] = start_t
                end_ts[bar_cnt] = timestamps[i]
                directions[bar_cnt] = 1
                ref_price = closes[bar_cnt]
                cur_dir = 1
                bar_cnt += 1
                start_t = timestamps[i]

    return (
        opens[:bar_cnt],
        highs[:bar_cnt],
        lows[:bar_cnt],
        closes[:bar_cnt],
        start_ts[:bar_cnt],
        end_ts[:bar_cnt],
        directions[:bar_cnt],
    )


@numba.njit(fastmath=True)
def numba_cumulative_volatility(prices: np.ndarray, pip_size: float, step: int = 10):
    n = len(prices)
    sub_n = (n + step - 1) // step
    cum_vol = np.empty(sub_n, dtype=np.float64)
    sub_prices = np.empty(sub_n, dtype=np.float64)

    accum = 0.0
    out_idx = 0
    cum_vol[0] = 0.0
    sub_prices[0] = prices[0]
    out_idx += 1

    for i in range(1, n):
        accum += abs(prices[i] - prices[i - 1]) / pip_size
        if i % step == 0:
            if out_idx < sub_n:
                cum_vol[out_idx] = accum
                sub_prices[out_idx] = prices[i]
                out_idx += 1

    return cum_vol[:out_idx], sub_prices[:out_idx]


# Warm-up JIT compilers
_dummy_p = np.array([1.0, 1.0005, 1.0010], dtype=np.float64)
_dummy_t = np.array([100, 200, 300], dtype=np.int64)
_ = numba_build_range_bars(_dummy_p, _dummy_t, 0.0005)
_ = numba_build_renko_bars(_dummy_p, _dummy_t, 0.0005)
_ = numba_cumulative_volatility(_dummy_p, 0.0001, 1)


# ==============================================================================
# MAIN GUI APPLICATION
# ==============================================================================
class VolatilityChartApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("TRD Terminal — EUR/USD Volatility Chart (Time-Component Removed)")
        self.root.geometry("1400x900")
        self.root.minsize(1050, 700)
        self.root.configure(bg="#0d1117")

        # Application state
        self.df_raw: Optional[pd.DataFrame] = None
        self.mid_prices: Optional[np.ndarray] = None
        self.timestamps: Optional[np.ndarray] = None
        self.current_bars: Optional[pd.DataFrame] = None

        self._setup_style()
        self._build_ui()
        self._load_available_files()

        # Auto-load initial dataset if available
        self.root.after(100, self.auto_load_initial_data)

    def _setup_style(self):
        self.style = ttk.Style()
        self.style.theme_use("clam")

        # Dark theme palette
        bg_dark = "#0d1117"
        bg_card = "#161b22"
        border_col = "#30363d"
        accent_blue = "#58a6ff"
        text_fg = "#c9d1d9"

        self.style.configure(".", background=bg_dark, foreground=text_fg, font=("Segoe UI", 9))
        self.style.configure("TFrame", background=bg_dark)
        self.style.configure("Card.TFrame", background=bg_card, relief="solid", borderwidth=1)
        self.style.configure("TLabel", background=bg_dark, foreground=text_fg)
        self.style.configure("Card.TLabel", background=bg_card, foreground=text_fg)
        self.style.configure("Title.TLabel", background=bg_dark, foreground="#58a6ff", font=("Segoe UI", 12, "bold"))
        self.style.configure("Header.TLabel", background=bg_card, foreground="#58a6ff", font=("Segoe UI", 9, "bold"))

        self.style.configure("TCombobox", fieldbackground="#21262d", background=bg_card, foreground="#ffffff", bordercolor=border_col)
        self.style.map("TCombobox", fieldbackground=[("readonly", "#21262d")], selectbackground=[("readonly", "#1f6feb")])

        self.style.configure("TButton", background="#238636", foreground="#ffffff", font=("Segoe UI", 9, "bold"), borderwidth=0, padding=6)
        self.style.map("TButton", background=[("active", "#2ea043"), ("disabled", "#21262d")])

        self.style.configure("Secondary.TButton", background="#21262d", foreground="#c9d1d9", font=("Segoe UI", 9), borderwidth=1, bordercolor=border_col)
        self.style.map("Secondary.TButton", background=[("active", "#30363d")])

        self.style.configure("TCheckbutton", background=bg_card, foreground=text_fg)
        self.style.map("TCheckbutton", background=[("active", bg_card)])

        self.style.configure("TScale", background=bg_card, troughcolor="#21262d")

    def _build_ui(self):
        # 1. Top Header Banner
        header_frame = tk.Frame(self.root, bg="#0d1117", height=50)
        header_frame.pack(fill=tk.X, padx=15, pady=(10, 5))

        title_lbl = tk.Label(
            header_frame,
            text="⚡ EUR/USD INTRINSIC VOLATILITY CHART (TIME-INVARIANT)",
            font=("Segoe UI", 13, "bold"),
            bg="#0d1117",
            fg="#58a6ff",
        )
        title_lbl.pack(side=tk.LEFT)

        subtitle_lbl = tk.Label(
            header_frame,
            text=" |  Constant-Volatility Domain (X-Axis: Traversal Units, Time Axis Discarded)",
            font=("Segoe UI", 10),
            bg="#0d1117",
            fg="#8b949e",
        )
        subtitle_lbl.pack(side=tk.LEFT, pady=(3, 0))

        # 2. Controls Toolbar Card
        ctrl_card = ttk.Frame(self.root, style="Card.TFrame")
        ctrl_card.pack(fill=tk.X, padx=15, pady=5)

        ctrl_inner = tk.Frame(ctrl_card, bg="#161b22", padx=12, pady=10)
        ctrl_inner.pack(fill=tk.X)

        # File Dropdown
        tk.Label(ctrl_inner, text="Dataset (Week):", font=("Segoe UI", 9, "bold"), bg="#161b22", fg="#c9d1d9").grid(row=0, column=0, sticky="w", padx=(0, 5))
        self.combo_file = ttk.Combobox(ctrl_inner, width=32, state="readonly")
        self.combo_file.grid(row=0, column=1, padx=(0, 15))

        # Ticks sample limit
        tk.Label(ctrl_inner, text="Tick Sample:", font=("Segoe UI", 9, "bold"), bg="#161b22", fg="#c9d1d9").grid(row=0, column=2, sticky="w", padx=(0, 5))
        self.combo_ticks = ttk.Combobox(ctrl_inner, width=12, state="readonly", values=["50,000", "100,000", "250,000", "500,000", "Full Week"])
        self.combo_ticks.set("100,000")
        self.combo_ticks.grid(row=0, column=3, padx=(0, 15))

        # Load button
        self.btn_load = ttk.Button(ctrl_inner, text="↻ Load Ticks", command=self.load_dataset)
        self.btn_load.grid(row=0, column=4, padx=(0, 20))

        # Chart Mode
        tk.Label(ctrl_inner, text="Volatility Mode:", font=("Segoe UI", 9, "bold"), bg="#161b22", fg="#c9d1d9").grid(row=0, column=5, sticky="w", padx=(0, 5))
        self.combo_mode = ttk.Combobox(
            ctrl_inner,
            width=26,
            state="readonly",
            values=[
                "Range Bars (Constant H-L)",
                "Renko Bricks (Momentum Steps)",
                "Cumulative Volatility (Total Path)",
            ],
        )
        self.combo_mode.set("Range Bars (Constant H-L)")
        self.combo_mode.grid(row=0, column=6, padx=(0, 20))
        self.combo_mode.bind("<<ComboboxSelected>>", lambda e: self.recompute_and_draw())

        # Volatility Threshold (Pips)
        tk.Label(ctrl_inner, text="Threshold (Pips):", font=("Segoe UI", 9, "bold"), bg="#161b22", fg="#c9d1d9").grid(row=0, column=7, sticky="w", padx=(0, 5))
        self.val_pips = tk.DoubleVar(value=5.0)

        self.slider_pips = ttk.Scale(ctrl_inner, from_=1.0, to=20.0, variable=self.val_pips, orient=tk.HORIZONTAL, length=120, command=self._on_slider_change)
        self.slider_pips.grid(row=0, column=8, padx=(0, 5))

        self.lbl_pips_display = tk.Label(ctrl_inner, text="5.0p", font=("Consolas", 10, "bold"), bg="#161b22", fg="#58a6ff", width=5)
        self.lbl_pips_display.grid(row=0, column=9, padx=(0, 20))

        # Overlays Checkboxes
        self.var_ema = tk.BooleanVar(value=True)
        self.chk_ema = ttk.Checkbutton(ctrl_inner, text="EMA 9/21", variable=self.var_ema, command=self.recompute_and_draw)
        self.chk_ema.grid(row=0, column=10, padx=8)

        self.var_bands = tk.BooleanVar(value=True)
        self.chk_bands = ttk.Checkbutton(ctrl_inner, text="Vol Bands", variable=self.var_bands, command=self.recompute_and_draw)
        self.chk_bands.grid(row=0, column=11, padx=8)

        self.var_sub = tk.BooleanVar(value=True)
        self.chk_sub = ttk.Checkbutton(ctrl_inner, text="Time-Distortion Subplot", variable=self.var_sub, command=self.recompute_and_draw)
        self.chk_sub.grid(row=0, column=12, padx=8)

        # 3. Chart Container
        chart_card = ttk.Frame(self.root, style="Card.TFrame")
        chart_card.pack(fill=tk.BOTH, expand=True, padx=15, pady=5)

        # Matplotlib Figure
        self.fig = plt.Figure(facecolor="#0d1117", dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=chart_card)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)

        # Matplotlib Toolbar
        toolbar_frame = tk.Frame(chart_card, bg="#161b22")
        toolbar_frame.pack(fill=tk.X, side=tk.BOTTOM)
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.config(background="#161b22")
        for button in self.toolbar.winfo_children():
            button.config(background="#161b22")
        self.toolbar.update()

        # 4. Bottom Info Status Bar
        self.status_bar = tk.Label(
            self.root,
            text="Ready. Select dataset and click Load Ticks.",
            font=("Segoe UI", 9),
            bg="#161b22",
            fg="#8b949e",
            anchor="w",
            padx=15,
            pady=5,
        )
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM)

    def _setup_axes(self):
        self.fig.clear()
        if self.var_sub.get() and self.combo_mode.get() != "Cumulative Volatility (Total Path)":
            gs = self.fig.add_gridspec(4, 1, hspace=0.15)
            self.ax_main = self.fig.add_subplot(gs[:3, 0])
            self.ax_sub = self.fig.add_subplot(gs[3, 0], sharex=self.ax_main)
            self.ax_sub.set_facecolor("#0d1117")
        else:
            self.ax_main = self.fig.add_subplot(111)
            self.ax_sub = None

        self.ax_main.set_facecolor("#0d1117")

        for ax in [self.ax_main] + ([self.ax_sub] if self.ax_sub else []):
            ax.tick_params(colors="#8b949e", labelsize=8)
            ax.grid(True, linestyle="--", linewidth=0.5, color="#21262d", alpha=0.7)
            for spine in ax.spines.values():
                spine.set_color("#30363d")

    def _on_slider_change(self, val):
        pips = round(float(val), 1)
        self.lbl_pips_display.config(text=f"{pips:.1f}p")
        # Throttle redraw
        if hasattr(self, "_slider_timer") and self._slider_timer is not None:
            self.root.after_cancel(self._slider_timer)
        self._slider_timer = self.root.after(150, self.recompute_and_draw)

    def _load_available_files(self):
        csv_files = sorted(glob.glob(str(DATA_DIR / "week_*.csv")))
        if csv_files:
            file_names = [os.path.basename(f) for f in csv_files]
            self.combo_file["values"] = file_names
            self.combo_file.set(file_names[0])
        else:
            self.combo_file["values"] = ["No CSV data found in eurusd_weekly_data"]

    def auto_load_initial_data(self):
        if self.combo_file.get() and "No CSV" not in self.combo_file.get():
            self.load_dataset()

    def load_dataset(self):
        selected_file = self.combo_file.get()
        if not selected_file or "No CSV" in selected_file:
            messagebox.showerror("Error", f"No tick data found in {DATA_DIR}")
            return

        file_path = DATA_DIR / selected_file
        if not file_path.exists():
            messagebox.showerror("Error", f"File does not exist: {file_path}")
            return

        # Parse limit
        limit_str = self.combo_ticks.get()
        nrows = None
        if limit_str == "50,000":
            nrows = 50000
        elif limit_str == "100,000":
            nrows = 100000
        elif limit_str == "250,000":
            nrows = 250000
        elif limit_str == "500,000":
            nrows = 500000

        self.status_bar.config(text=f"Loading {selected_file} ({limit_str} ticks)...", fg="#58a6ff")
        self.root.update_idletasks()

        try:
            t0 = time.time()
            df = pd.read_csv(file_path, nrows=nrows)
            self.df_raw = df
            self.mid_prices = (0.5 * (df["askPrice"] + df["bidPrice"])).to_numpy(dtype=np.float64)
            self.timestamps = df["timestamp"].to_numpy(dtype=np.int64)
            load_time = time.time() - t0

            self.status_bar.config(
                text=f"Loaded {len(self.mid_prices):,} ticks from {selected_file} in {load_time:.2f}s. Recomputing chart...",
                fg="#3fb950",
            )
            self.recompute_and_draw()
        except Exception as e:
            messagebox.showerror("Load Failed", str(e))
            self.status_bar.config(text=f"Error loading file: {e}", fg="#f85149")

    # ==========================================================================
    # CHART RECOMPUTATION & RENDERING
    # ==========================================================================
    def recompute_and_draw(self):
        if self.mid_prices is None or len(self.mid_prices) == 0:
            return

        t_calc_0 = time.time()
        mode = self.combo_mode.get()
        thresh_pips = round(self.val_pips.get(), 1)
        thresh_val = thresh_pips * PIP_SIZE

        self._setup_axes()

        # ----------------------------------------------------------------------
        # MODE 1: RANGE BARS (CONSTANT VOLATILITY HIGH-LOW)
        # ----------------------------------------------------------------------
        if mode == "Range Bars (Constant H-L)":
            o, h, l, c, st, et, tc = numba_build_range_bars(self.mid_prices, self.timestamps, thresh_val)
            n_bars = len(o)
            if n_bars == 0:
                self.ax_main.text(0.5, 0.5, "No range bars formed. Reduce pip threshold.", color="white", ha="center")
                self.canvas.draw()
                return

            indices = np.arange(n_bars)

            # Draw Candlestick Wicks (LineCollection)
            wicks = [[(i, l[i]), (i, h[i])] for i in range(n_bars)]
            wick_coll = LineCollection(wicks, colors="#6e7681", linewidths=1.0, alpha=0.9)
            self.ax_main.add_collection(wick_coll)

            # Draw Candlestick Bodies (Bar collections)
            up = c >= o
            down = ~up

            # Green bullish bodies
            if np.any(up):
                self.ax_main.bar(
                    indices[up],
                    np.maximum(c[up] - o[up], 0.00002),
                    bottom=o[up],
                    color="#26a69a",
                    edgecolor="#26a69a",
                    width=0.75,
                    alpha=0.95,
                    label="Bullish Expansion"
                )
            # Red bearish bodies
            if np.any(down):
                self.ax_main.bar(
                    indices[down],
                    np.maximum(o[down] - c[down], 0.00002),
                    bottom=c[down],
                    color="#ef5350",
                    edgecolor="#ef5350",
                    width=0.75,
                    alpha=0.95,
                    label="Bearish Expansion"
                )

            # Overlays: Fast & Slow EMA on Range Bars
            if self.var_ema.get() and n_bars >= 10:
                s_c = pd.Series(c)
                ema9 = s_c.ewm(span=9, adjust=False).mean().to_numpy()
                ema21 = s_c.ewm(span=21, adjust=False).mean().to_numpy()
                self.ax_main.plot(indices, ema9, color="#00d2ff", linewidth=1.2, label="EMA 9 (Volatility-Time)")
                self.ax_main.plot(indices, ema21, color="#ffb300", linewidth=1.2, label="EMA 21 (Volatility-Time)")

            # Overlays: Volatility Bands
            if self.var_bands.get() and n_bars >= 20:
                s_c = pd.Series(c)
                sma20 = s_c.rolling(20).mean().bfill().to_numpy()
                std20 = s_c.rolling(20).std().bfill().to_numpy()
                upper = sma20 + 2.0 * std20
                lower = sma20 - 2.0 * std20
                self.ax_main.plot(indices, upper, color="#a371f7", linewidth=0.8, linestyle="--", alpha=0.8)
                self.ax_main.plot(indices, lower, color="#a371f7", linewidth=0.8, linestyle="--", alpha=0.8)
                self.ax_main.fill_between(indices, lower, upper, color="#8957e5", alpha=0.07, label="Volatility Envelope (2.0 Std)")

            self.ax_main.set_title(
                f"EUR/USD Range Bars (Fixed Height = {thresh_pips:.1f} Pips)  |  Total Bars: {n_bars:,}  |  Zero Time Gaps",
                color="#58a6ff",
                fontsize=11,
                fontweight="bold",
                pad=10,
            )
            self.ax_main.set_ylabel("Price (EUR/USD)", color="#8b949e", fontsize=9)
            self.ax_main.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.5f"))
            self.ax_main.set_xlim(-1, n_bars)
            self.ax_main.legend(loc="upper left", facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9", fontsize=8)

            # Subplot: Time-Distortion Metric (Minutes per Range Bar)
            if self.ax_sub is not None:
                dur_minutes = np.maximum((et - st) / (1000.0 * 60.0), 0.05)
                # Cap extreme outliers for visualization
                norm_colors = np.clip(dur_minutes / np.percentile(dur_minutes, 90), 0.1, 1.0)

                # Fast bar plot
                self.ax_sub.bar(indices, dur_minutes, color="#388bfd", width=0.8, alpha=0.7)
                self.ax_sub.set_ylabel("Duration\n(Minutes)", color="#8b949e", fontsize=8)
                self.ax_sub.set_xlabel("Time-Invariant Volatility Bar Index", color="#58a6ff", fontsize=9, fontweight="bold")
                self.ax_sub.set_ylim(0, np.percentile(dur_minutes, 98) * 1.2)

        # ----------------------------------------------------------------------
        # MODE 2: RENKO VOLATILITY BRICKS
        # ----------------------------------------------------------------------
        elif mode == "Renko Bricks (Momentum Steps)":
            o, h, l, c, st, et, dirs = numba_build_renko_bars(self.mid_prices, self.timestamps, thresh_val)
            n_bars = len(o)
            if n_bars == 0:
                self.ax_main.text(0.5, 0.5, "No Renko bricks formed. Reduce brick size.", color="white", ha="center")
                self.canvas.draw()
                return

            indices = np.arange(n_bars)
            up = dirs == 1
            down = ~up

            # Draw Renko Bricks
            if np.any(up):
                self.ax_main.bar(
                    indices[up],
                    c[up] - o[up],
                    bottom=o[up],
                    color="#26a69a",
                    edgecolor="#161b22",
                    linewidth=0.5,
                    width=0.85,
                    label="Bullish Brick",
                )
            if np.any(down):
                self.ax_main.bar(
                    indices[down],
                    o[down] - c[down],
                    bottom=c[down],
                    color="#ef5350",
                    edgecolor="#161b22",
                    linewidth=0.5,
                    width=0.85,
                    label="Bearish Brick",
                )

            # Overlays: Moving Average on Renko
            if self.var_ema.get() and n_bars >= 10:
                s_c = pd.Series(c)
                ema9 = s_c.ewm(span=9, adjust=False).mean().to_numpy()
                ema21 = s_c.ewm(span=21, adjust=False).mean().to_numpy()
                self.ax_main.plot(indices, ema9, color="#00d2ff", linewidth=1.2, label="EMA 9 (Renko)")
                self.ax_main.plot(indices, ema21, color="#ffb300", linewidth=1.2, label="EMA 21 (Renko)")

            self.ax_main.set_title(
                f"EUR/USD Renko Bricks (Brick Size = {thresh_pips:.1f} Pips, 2-Brick Reversal)  |  Total Bricks: {n_bars:,}",
                color="#58a6ff",
                fontsize=11,
                fontweight="bold",
                pad=10,
            )
            self.ax_main.set_ylabel("Price (EUR/USD)", color="#8b949e", fontsize=9)
            self.ax_main.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.5f"))
            self.ax_main.set_xlim(-1, n_bars)
            self.ax_main.legend(loc="upper left", facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9", fontsize=8)

            if self.ax_sub is not None:
                dur_minutes = np.maximum((et - st) / (1000.0 * 60.0), 0.05)
                self.ax_sub.bar(indices, dur_minutes, color="#a371f7", width=0.8, alpha=0.7)
                self.ax_sub.set_ylabel("Duration\n(Minutes)", color="#8b949e", fontsize=8)
                self.ax_sub.set_xlabel("Renko Momentum Brick Index", color="#58a6ff", fontsize=9, fontweight="bold")
                self.ax_sub.set_ylim(0, np.percentile(dur_minutes, 98) * 1.2)

        # ----------------------------------------------------------------------
        # MODE 3: CUMULATIVE VOLATILITY PATH (TOTAL VARIATION TRAVERSED)
        # ----------------------------------------------------------------------
        elif mode == "Cumulative Volatility (Total Path)":
            step = max(len(self.mid_prices) // 4000, 1)
            cum_vol, sub_p = numba_cumulative_volatility(self.mid_prices, PIP_SIZE, step=step)

            self.ax_main.plot(cum_vol, sub_p, color="#00d2ff", linewidth=1.2, alpha=0.9, label="EUR/USD Trajectory")

            # Fill gradient area to visually represent volatility mass
            self.ax_main.fill_between(cum_vol, np.min(sub_p), sub_p, color="#00d2ff", alpha=0.08)

            self.ax_main.set_title(
                f"EUR/USD Price vs. Total Traversed Volatility (Intrinsic Distance Path)  |  Total Path: {cum_vol[-1]:,.1f} Pips",
                color="#58a6ff",
                fontsize=11,
                fontweight="bold",
                pad=10,
            )
            self.ax_main.set_xlabel("Total Volatility Accumulated (Cumulative Absolute Pips Traversed)", color="#58a6ff", fontsize=9, fontweight="bold")
            self.ax_main.set_ylabel("EUR/USD Price", color="#8b949e", fontsize=9)
            self.ax_main.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.5f"))
            self.ax_main.set_xlim(0, cum_vol[-1])
            self.ax_main.legend(loc="upper left", facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9", fontsize=8)

        # Redraw Canvas
        self.canvas.draw()
        calc_time = (time.time() - t_calc_0) * 1000.0

        ratio_str = f"{len(self.mid_prices) / max(n_bars, 1):.1f}:1" if mode != "Cumulative Volatility (Total Path)" else "Continuous"
        bars_str = f"{n_bars:,} Bars" if mode != "Cumulative Volatility (Total Path)" else f"{len(cum_vol):,} Points"

        self.status_bar.config(
            text=f"Rendered in {calc_time:.1f}ms  |  Source Ticks: {len(self.mid_prices):,}  |  Output: {bars_str}  |  Compression Ratio: {ratio_str}  |  Threshold: {thresh_pips:.1f} pips",
            fg="#c9d1d9",
        )


def main():
    root = tk.Tk()
    app = VolatilityChartApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
