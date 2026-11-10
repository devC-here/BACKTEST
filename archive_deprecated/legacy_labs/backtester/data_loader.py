"""
Data Loader and Resampling Pipeline for EURUSD Weekly Datasets.
Loads raw Dukascopy tick CSVs (53 weekly sets) and resamples them into
multi-timeframe OHLCV candles (M1, M5, M15, M30, H1, H4, D1) with bid/ask spreads.
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np


class DataLoader:
    TIMEFRAME_MAP = {
        "M1": "1min",
        "M5": "5min",
        "M15": "15min",
        "M30": "30min",
        "H1": "1h",
        "H2": "2h",
        "H4": "4h",
        "D1": "1D"
    }

    def __init__(self, base_dir: Optional[Path] = None):
        if base_dir is None:
            # Default to repo root
            self.base_dir = Path(__file__).resolve().parent.parent
        else:
            self.base_dir = Path(base_dir)

        self.weekly_data_dir = self.base_dir / "eurusd_weekly_data"
        self.manifest_file = self.weekly_data_dir / "manifest_weekly_sets.json"
        self.cache_dir = self.base_dir / "data" / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._manifest = None

    @property
    def manifest(self) -> List[Dict[str, Any]]:
        if self._manifest is None:
            if not self.manifest_file.exists():
                raise FileNotFoundError(f"Manifest not found at {self.manifest_file}")
            with open(self.manifest_file, "r", encoding="utf-8") as f:
                self._manifest = json.load(f)
        return self._manifest

    def list_weeks(self) -> List[Dict[str, Any]]:
        """Returns metadata for all available weekly datasets."""
        return self.manifest

    def get_week_info(self, week_num: int) -> Dict[str, Any]:
        """Returns metadata for a specific week number (1 to 53)."""
        for item in self.manifest:
            if item.get("week") == week_num:
                return item
        raise ValueError(f"Week {week_num} not found in manifest (available: 1 to {len(self.manifest)})")

    def load_week_ticks(self, week_num: int) -> pd.DataFrame:
        """Loads raw tick data for a given week."""
        w_info = self.get_week_info(week_num)
        file_path = self.weekly_data_dir / w_info["filename"]
        if not file_path.exists():
            # Check absolute path fallback
            file_path = Path(w_info.get("absolute_path", ""))
        if not file_path.exists():
            raise FileNotFoundError(f"Weekly CSV not found: {file_path}")

        df = pd.read_csv(
            file_path,
            usecols=["timestamp", "askPrice", "bidPrice", "askVolume", "bidVolume"],
            dtype={
                "timestamp": "int64",
                "askPrice": "float64",
                "bidPrice": "float64",
                "askVolume": "float32",
                "bidVolume": "float32"
            }
        )
        return df

    def load_week_candles(self, week_num: int, timeframe: str = "M5", use_cache: bool = True) -> pd.DataFrame:
        """
        Loads candles for a given week and timeframe.
        If cached parquet file exists, loads in under 5ms.
        Otherwise resamples ticks and caches the result.
        """
        tf_upper = timeframe.upper()
        if tf_upper not in self.TIMEFRAME_MAP:
            raise ValueError(f"Unsupported timeframe: {timeframe}. Choose from {list(self.TIMEFRAME_MAP.keys())}")

        cache_file = self.cache_dir / f"week_{week_num:02d}_{tf_upper}.parquet"
        if use_cache and cache_file.exists():
            try:
                df_cached = pd.read_parquet(cache_file)
                return df_cached
            except Exception:
                pass  # Recompute if cache is corrupted

        # Load raw ticks and resample
        df_ticks = self.load_week_ticks(week_num)
        rule = self.TIMEFRAME_MAP[tf_upper]

        df_ticks["dt"] = pd.to_datetime(df_ticks["timestamp"], unit="ms", utc=True)
        df_ticks.set_index("dt", inplace=True)

        # Resample bid OHLC
        bid_ohlc = df_ticks["bidPrice"].resample(rule).ohlc()

        # Resample ask OHLC
        ask_ohlc = df_ticks["askPrice"].resample(rule).ohlc()

        # Resample volumes and tick counts
        tick_counts = df_ticks["bidPrice"].resample(rule).count()
        bid_vol = df_ticks["bidVolume"].resample(rule).sum()
        ask_vol = df_ticks["askVolume"].resample(rule).sum()

        # Combine and drop empty periods (e.g. weekend flat hours)
        df_candles = pd.DataFrame({
            "open": bid_ohlc["open"],
            "high": bid_ohlc["high"],
            "low": bid_ohlc["low"],
            "close": bid_ohlc["close"],
            "ask_open": ask_ohlc["open"],
            "ask_high": ask_ohlc["high"],
            "ask_low": ask_ohlc["low"],
            "ask_close": ask_ohlc["close"],
            "tick_volume": tick_counts,
            "bid_volume": bid_vol,
            "ask_volume": ask_vol
        }).dropna()

        # Spread in pips (1 pip = 0.0001)
        df_candles["spread_pips"] = ((df_candles["ask_close"] - df_candles["close"]) / 0.0001).round(1)

        # Metadata columns
        df_candles["time"] = df_candles.index
        df_candles["week"] = week_num
        df_candles["hour_utc"] = df_candles.index.hour
        df_candles["minute_utc"] = df_candles.index.minute
        df_candles["day_name"] = df_candles.index.day_name()
        df_candles.reset_index(drop=True, inplace=True)

        # Save to parquet cache
        if use_cache:
            try:
                df_candles.to_parquet(cache_file, index=False)
            except Exception:
                pass

        return df_candles
