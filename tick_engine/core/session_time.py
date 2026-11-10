"""
================================================================================
  INSTITUTIONAL SESSION TIME & DST CONVERTER
================================================================================
Provides daylight-saving-time (DST) aware market session windows.
Standardizes London, New York, and Tokyo trading sessions across UTC transitions,
preventing the 1-hour session drift between UTC and local financial centers.

Standard Market Center Clocks:
- New York (US/Eastern):
    * Regular Session: 08:00 - 17:00 Eastern (EDT / EST)
    * NY Sweep / Open Window: 08:00 - 13:00 Eastern
- London (Europe/London):
    * Regular Session: 08:00 - 16:30 London (BST / GMT)
    * London Sweep / Open Window: 07:00 - 11:00 London
- Tokyo (Asia/Tokyo):
    * Asian Session: 09:00 - 15:00 Tokyo (JST, fixed UTC+9)
================================================================================
"""

from __future__ import annotations
from typing import Union, Tuple
import numpy as np
import pandas as pd


class SessionClock:
    """
    Translates UTC timestamps or Pandas DatetimeIndex into local financial center hours,
    automatically resolving daylight saving transitions.
    """

    @staticmethod
    def get_market_hours(
        index: Union[pd.DatetimeIndex, pd.Series, np.ndarray],
        market_tz: str = "America/New_York"
    ) -> np.ndarray:
        """
        Converts UTC index or millisecond timestamps into local market hours (0-23).
        
        Args:
            index: pd.DatetimeIndex (assumed UTC) or numpy array of UTC millisecond timestamps.
            market_tz: 'America/New_York' (EDT/EST), 'Europe/London' (BST/GMT), or 'Asia/Tokyo' (JST).
            
        Returns:
            np.ndarray of integer hours (0..23) in local market time.
        """
        if isinstance(index, np.ndarray):
            # Convert millisecond timestamps to UTC DatetimeIndex
            dti = pd.to_datetime(index, unit='ms', utc=True)
        elif isinstance(index, pd.DatetimeIndex):
            if index.tz is None:
                dti = index.tz_localize('UTC')
            else:
                dti = index.tz_convert('UTC')
        elif isinstance(index, pd.Series):
            if index.dt.tz is None:
                dti = index.dt.tz_localize('UTC')
            else:
                dti = index.dt.tz_convert('UTC')
        else:
            dti = pd.to_datetime(index, utc=True)

        local_time = dti.tz_convert(market_tz)
        return local_time.hour.values if hasattr(local_time, 'hour') else local_time.dt.hour.values

    @classmethod
    def is_new_york_window(
        cls,
        index: Union[pd.DatetimeIndex, np.ndarray],
        start_hour: int = 8,
        end_hour: int = 13
    ) -> np.ndarray:
        """Checks if timestamps fall within New York Eastern time window [start_hour, end_hour]."""
        ny_hours = cls.get_market_hours(index, "America/New_York")
        return (ny_hours >= start_hour) & (ny_hours <= end_hour)

    @classmethod
    def is_london_window(
        cls,
        index: Union[pd.DatetimeIndex, np.ndarray],
        start_hour: int = 8,
        end_hour: int = 16
    ) -> np.ndarray:
        """Checks if timestamps fall within London time window [start_hour, end_hour]."""
        ldn_hours = cls.get_market_hours(index, "Europe/London")
        return (ldn_hours >= start_hour) & (ldn_hours <= end_hour)

    @classmethod
    def is_tokyo_window(
        cls,
        index: Union[pd.DatetimeIndex, np.ndarray],
        start_hour: int = 0,
        end_hour: int = 6
    ) -> np.ndarray:
        """Checks if timestamps fall within Tokyo Asian session (Tokyo has no DST, fixed UTC+9)."""
        tky_hours = cls.get_market_hours(index, "Asia/Tokyo")
        return (tky_hours >= start_hour) & (tky_hours <= end_hour)

