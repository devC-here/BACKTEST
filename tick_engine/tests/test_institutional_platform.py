"""
================================================================================
  INSTITUTIONAL STRATEGY VERIFICATION PLATFORM: INTEGRATION TEST SUITE
================================================================================
Implements Requirement 9 of the Institutional Verification Standard:
1. test_continuous_split_integrity: Zero random weeks, unbroken 1..210 and 211..300.
2. test_data_quality_gate_rejections: Inverted quotes, non-positive prices, short span.
3. test_order_timing_contract: No pre-bar fills, fill_ts strictly after bar close.
4. test_causal_warmup_exclusion: Zero trade entries during warmup period.
5. test_broker_cost_deductions: Precise commission and entry slippage debit.
6. test_gap_through_stop_loss: Exit at actual market gap price when slippage active.
7. test_deterministic_fixture: Bit-for-bit reproducible trade log on fixed fixture.
================================================================================
"""

import unittest
import numpy as np
import pandas as pd
from pathlib import Path
import json

from tick_engine.core.engine import (
    TICK_DTYPE,
    TRADE_DTYPE,
    run_matching_loop,
    run_matching_loop_signals,
    EXIT_REASON_SL,
    EXIT_REASON_TP,
    EXIT_REASON_SIGNAL,
    SIGNAL_HOLD,
    SIGNAL_LONG,
    SIGNAL_SHORT,
    SIGNAL_FORCE_EXIT,
    DEFAULT_COMMISSION_PIPS,
    DEFAULT_ENTRY_SLIPPAGE_PIPS
)
from tick_engine.quality.validator import validate_tick_stream
from tick_engine.core.dataloader import load_manifest, get_discovery_weeks, get_holdout_test_weeks
from tick_engine.core.config import SPLITS_FILE

class TestInstitutionalPlatform(unittest.TestCase):

    def setUp(self):
        self.pip_size = 0.0001

    def test_continuous_split_integrity(self):
        """Validates that splits across all 10 assets are 100% continuous with zero overlap."""
        self.assertTrue(SPLITS_FILE.exists(), "Splits file must exist")
        with open(SPLITS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        assets = data["assets"]
        self.assertEqual(len(assets), 10, "Must cover all 10 assets")
        for sym, d in assets.items():
            disc = d["discovery_70"]["weeks"]
            hold = d["holdout_test_30"]["weeks"]
            
            # 1..210 and 211..300
            self.assertEqual(disc, list(range(1, 211)), f"{sym} discovery must be unbroken 1..210")
            self.assertEqual(hold, list(range(211, 301)), f"{sym} holdout must be unbroken 211..300")
            
            # Disjoint
            intersection = set(disc).intersection(set(hold))
            self.assertEqual(len(intersection), 0, f"{sym} discovery and holdout must have zero overlap")

    def test_data_quality_gate_rejections(self):
        """Validates that inverted quotes, non-positive prices, and short spans are rejected."""
        # 1. Inverted quote (ask < bid)
        bad_ticks = np.zeros(20000, dtype=TICK_DTYPE)
        bad_ticks['timestamp'] = np.arange(1000, 1000 + 20000 * 100, 100)
        bad_ticks['bid'] = 1.1000
        bad_ticks['ask'] = 1.1001
        bad_ticks['ask'][50] = 1.0990  # Inverted!

        rep = validate_tick_stream(bad_ticks, "EURUSD", 1, min_span_hours=0.1)
        self.assertFalse(rep.is_valid)
        self.assertGreater(rep.inverted_quote_count, 0)
        self.assertTrue(any("inverted quotes" in r for r in rep.rejection_reasons))

        # 2. Non-positive price
        bad_ticks2 = np.zeros(20000, dtype=TICK_DTYPE)
        bad_ticks2['timestamp'] = np.arange(1000, 1000 + 20000 * 100, 100)
        bad_ticks2['bid'] = 1.1000
        bad_ticks2['ask'] = 1.1001
        bad_ticks2['bid'][10] = -0.5  # Negative!

        rep2 = validate_tick_stream(bad_ticks2, "EURUSD", 1, min_span_hours=0.1)
        self.assertFalse(rep2.is_valid)
        self.assertGreater(rep2.non_positive_price_count, 0)

        # 3. Truncated span (< 80 hours)
        short_ticks = np.zeros(20000, dtype=TICK_DTYPE)
        # 23 hours span = 23 * 3600 * 1000 ms
        short_ticks['timestamp'] = np.linspace(0, 23 * 3600 * 1000, 20000).astype(np.int64)
        short_ticks['bid'] = 1.1000
        short_ticks['ask'] = 1.1001
        rep3 = validate_tick_stream(short_ticks, "EURUSD", 1, min_span_hours=80.0)
        self.assertFalse(rep3.is_valid)
        self.assertTrue(any("Truncated session span" in r for r in rep3.rejection_reasons))

    def test_broker_cost_deductions(self):
        """Validates exact deduction of commission and adverse entry slippage."""
        ticks = np.zeros(3, dtype=TICK_DTYPE)
        ticks['timestamp'] = [1000, 2000, 3000]
        # Spread is 1.0 pip (Bid: 1.1000, Ask: 1.1001)
        ticks['bid'] = [1.1000, 1.1005, 1.1005]
        ticks['ask'] = [1.1001, 1.1006, 1.1006]

        signals = np.array([SIGNAL_LONG, SIGNAL_HOLD, SIGNAL_FORCE_EXIT], dtype=np.int8)

        # Apply 0.40p commission and 0.20p slippage
        comm = 0.40
        slip = 0.20

        trades = run_matching_loop(
            ticks=ticks,
            signals=signals,
            pip_size=self.pip_size,
            sl_pips=50.0,
            tp_pips=50.0,
            close_at_end=False,
            commission_pips=comm,
            entry_slippage_pips=slip
        )

        self.assertEqual(len(trades), 1)
        tr = trades[0]

        # Raw Ask = 1.1001. With 0.20p slippage (+0.00002), entry price = 1.10012
        self.assertAlmostEqual(tr['entry_price'], 1.10012, places=6)
        # Exit at Bid = 1.1005
        self.assertAlmostEqual(tr['exit_price'], 1.10050, places=6)

        # Raw pips = (1.10050 - 1.10012) / 0.0001 = 3.80 pips
        # Net pips = 3.80 - 0.40 commission = 3.40 pips
        self.assertAlmostEqual(tr['pips'], 3.40, places=4)

    def test_gap_through_stop_loss(self):
        """Validates that price gapping past SL executes at actual gap price (adverse fill)."""
        ticks = np.zeros(3, dtype=TICK_DTYPE)
        ticks['timestamp'] = [1000, 2000, 3000]
        # Long entry at 1.1000. SL 10 pips -> nominal SL = 1.0990
        ticks['bid'] = [1.0999, 1.0975, 1.0975]  # Gaps down 15 pips past SL!
        ticks['ask'] = [1.1000, 1.0976, 1.0976]

        signals = np.array([SIGNAL_LONG, SIGNAL_HOLD, SIGNAL_HOLD], dtype=np.int8)

        trades = run_matching_loop(
            ticks=ticks,
            signals=signals,
            pip_size=self.pip_size,
            sl_pips=10.0,
            tp_pips=20.0,
            close_at_end=False,
            use_slippage=True,
            commission_pips=0.0,
            entry_slippage_pips=0.0
        )

        self.assertEqual(len(trades), 1)
        tr = trades[0]
        self.assertEqual(tr['exit_reason'], EXIT_REASON_SL)
        # MUST exit at actual gap bid price (1.0975), NOT the nominal SL (1.0990)
        self.assertAlmostEqual(tr['exit_price'], 1.0975, places=6)
        # Loss must be -25.0 pips (1.0975 - 1.1000), not -10.0 pips
        self.assertAlmostEqual(tr['pips'], -25.0, places=4)

    def test_deterministic_tick_fixture(self):
        """Validates that a fixed synthetic tick fixture produces bit-for-bit identical trade results."""
        np.random.seed(12345)
        n = 50
        ts = np.arange(1000, 1000 + n * 1000, 1000, dtype=np.int64)
        mids = 1.1000 + np.cumsum(np.random.choice([-0.0002, 0.0002], size=n))
        ticks = np.zeros(n, dtype=TICK_DTYPE)
        ticks['timestamp'] = ts
        ticks['bid'] = mids - 0.00005
        ticks['ask'] = mids + 0.00005

        signals = np.zeros(n, dtype=np.int8)
        signals[5] = SIGNAL_LONG
        signals[25] = SIGNAL_SHORT

        trades1 = run_matching_loop(ticks, signals, pip_size=0.0001, sl_pips=15.0, tp_pips=30.0, commission_pips=0.4, entry_slippage_pips=0.2)
        trades2 = run_matching_loop(ticks, signals, pip_size=0.0001, sl_pips=15.0, tp_pips=30.0, commission_pips=0.4, entry_slippage_pips=0.2)

        self.assertEqual(len(trades1), len(trades2))
        np.testing.assert_array_equal(trades1, trades2)

    def test_order_timing_contract_subsequent_tick(self):
        """Validates that orders NEVER fill on the same tick when enforce_subsequent_tick is active."""
        ticks = np.zeros(4, dtype=TICK_DTYPE)
        ticks['timestamp'] = [1000, 2000, 3000, 4000]
        ticks['bid'] = [1.1000, 1.1005, 1.1008, 1.1010]
        ticks['ask'] = [1.1001, 1.1006, 1.1009, 1.1011]

        signals = np.array([SIGNAL_LONG, SIGNAL_HOLD, SIGNAL_FORCE_EXIT, SIGNAL_HOLD], dtype=np.int8)

        # Enforce order timing contract: tick 0 signal MUST fill on tick 1, never tick 0
        trades = run_matching_loop(
            ticks=ticks,
            signals=signals,
            pip_size=self.pip_size,
            sl_pips=50.0,
            tp_pips=50.0,
            close_at_end=False,
            commission_pips=0.0,
            entry_slippage_pips=0.0,
            enforce_subsequent_tick=True
        )

        self.assertEqual(len(trades), 1)
        tr = trades[0]
        # Signal was generated at tick 0 (ts=1000)
        self.assertEqual(tr['signal_time'], 1000)
        # Entry MUST be at tick 1 (ts=2000), strictly subsequent to signal
        self.assertEqual(tr['entry_time'], 2000)
        self.assertGreater(tr['entry_time'], tr['signal_time'])
        # Entry price must be Ask at tick 1 (1.1006)
        self.assertAlmostEqual(tr['entry_price'], 1.1006, places=6)

    def test_causal_warmup_exclusion(self):
        """Validates that signals generated during the causal warmup window are prohibited from entering."""
        ticks = np.zeros(6, dtype=TICK_DTYPE)
        ticks['timestamp'] = [1000, 2000, 3000, 4000, 5000, 6000]
        ticks['bid'] = [1.1000, 1.1002, 1.1005, 1.1007, 1.1010, 1.1012]
        ticks['ask'] = [1.1001, 1.1003, 1.1006, 1.1008, 1.1011, 1.1013]

        # Signal at tick 1 (warmup), and signal at tick 4 (test window)
        signals = np.array([SIGNAL_HOLD, SIGNAL_LONG, SIGNAL_FORCE_EXIT, SIGNAL_HOLD, SIGNAL_LONG, SIGNAL_HOLD], dtype=np.int8)

        # Set min_entry_timestamp = 4000 (ticks 0..2 are warmup)
        trades = run_matching_loop(
            ticks=ticks,
            signals=signals,
            pip_size=self.pip_size,
            sl_pips=50.0,
            tp_pips=50.0,
            close_at_end=True,
            commission_pips=0.0,
            entry_slippage_pips=0.0,
            min_entry_timestamp=4000
        )

        # The warmup trade at tick 1 must be strictly excluded!
        self.assertEqual(len(trades), 1)
        tr = trades[0]
        self.assertGreaterEqual(tr['entry_time'], 4000)
        self.assertEqual(tr['entry_time'], 5000)

    def test_session_clock_dst_handling(self):
        """Validates that SessionClock converts UTC timestamps correctly across DST boundaries."""
        from tick_engine.core.session_time import SessionClock
        # 1. Summer EDT (July 1): UTC 12:00 = 08:00 AM EDT (UTC - 4)
        ts_summer = pd.to_datetime(['2024-07-01 12:00:00']).tz_localize('UTC')
        hours_ny_summer = SessionClock.get_market_hours(ts_summer, "America/New_York")
        self.assertEqual(hours_ny_summer[0], 8)

        # 2. Winter EST (January 15): UTC 13:00 = 08:00 AM EST (UTC - 5)
        ts_winter = pd.to_datetime(['2024-01-15 13:00:00']).tz_localize('UTC')
        hours_ny_winter = SessionClock.get_market_hours(ts_winter, "America/New_York")
        self.assertEqual(hours_ny_winter[0], 8)

        # Both dates correctly map to 08:00 AM New York opening bell!
        self.assertTrue(SessionClock.is_new_york_window(ts_summer, 8, 13)[0])
        self.assertTrue(SessionClock.is_new_york_window(ts_winter, 8, 13)[0])

    def test_broker_cost_stress_battery(self):
        """Validates that the 5 broker stress profiles evaluate correctly."""
        from tick_engine.stress.stress_suite import evaluate_stress_battery, STRESS_PROFILES
        ticks = np.zeros(10, dtype=TICK_DTYPE)
        ticks['timestamp'] = np.arange(1000, 1000 + 10 * 1000, 1000)
        ticks['bid'] = 1.1000 + np.sin(np.linspace(0, 3.14, 10)) * 0.0020
        ticks['ask'] = ticks['bid'] + 0.0001
        signals = np.zeros(10, dtype=np.int8)
        signals[2] = SIGNAL_LONG
        signals[7] = SIGNAL_FORCE_EXIT

        res = evaluate_stress_battery(ticks, signals, symbol="EURUSD", pip_size=0.0001, sl_pips=20.0, tp_pips=40.0)
        self.assertEqual(len(res["profiles"]), len(STRESS_PROFILES))
        self.assertIn("BASELINE", res["profiles"])
        self.assertIn("SEVERE_COMPOSITE", res["profiles"])
        # Composite severe must have less or equal pips than baseline
        self.assertLessEqual(res["profiles"]["SEVERE_COMPOSITE"]["net_pips"], res["profiles"]["BASELINE"]["net_pips"])

    def test_canonical_run_manifest_generation(self):
        """Validates that run manifest computes SHA256 hashes, CI 95%, bootstrap DD, and outlier sensitivity."""
        from tick_engine.reporting.run_manifest import generate_canonical_manifest
        dummy_trades = np.zeros(20, dtype=TRADE_DTYPE)
        dummy_trades['pips'] = [5.0, -3.0, 4.0, -2.0, 6.0, -1.0, 3.0, -2.5, 4.5, -1.5,
                                7.0, -2.0, 5.0, -3.0, 4.0, -2.0, 6.0, -1.0, 20.0, -2.0]
        dummy_trades['entry_time'] = np.arange(1000, 1000 + 20 * 86400 * 1000, 86400 * 1000)
        dummy_trades['exit_time'] = dummy_trades['entry_time'] + 3600 * 1000
        dummy_trades['signal_time'] = dummy_trades['entry_time'] - 100

        manifest = generate_canonical_manifest(
            strategy_name="Mock_Institutional_Strategy",
            strategy_file=Path(__file__),
            symbol="EURUSD",
            split_version="canonical_continuous_v1.0",
            start_week=211,
            end_week=300,
            pip_size=0.0001,
            commission_pips=0.40,
            entry_slippage_pips=0.20,
            spread_policy="Floating Raw Dukascopy",
            warmup_ticks=100000,
            trades=dummy_trades,
            candidates_screened=1
        )

        self.assertIn("strategy", manifest)
        self.assertIn("code_sha256", manifest["strategy"])
        self.assertIn("engine", manifest)
        self.assertIn("engine_code_sha256", manifest["engine"])
        self.assertIn("statistical_metrics", manifest)
        self.assertGreater(manifest["statistical_metrics"]["t_statistic"], 0)
        self.assertIn("expectancy_ci_95", manifest["statistical_metrics"])
        self.assertIn("pips_ex_top5", manifest["statistical_metrics"]["outlier_sensitivity"])

if __name__ == '__main__':
    unittest.main()
