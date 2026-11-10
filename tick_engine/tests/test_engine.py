"""
Unit Tests for Fast Minimal Tick-Level Strategy Tester.
Validates execution rules, fill prices (Long at Ask, Short at Bid),
SL/TP matching, and pip accounting accuracy.
"""

import unittest
import numpy as np
from tick_engine.core.engine import (
    TICK_DTYPE,
    TRADE_DTYPE,
    run_matching_loop,
    EXIT_REASON_SL,
    EXIT_REASON_TP,
    EXIT_REASON_SIGNAL,
    SIGNAL_HOLD,
    SIGNAL_LONG,
    SIGNAL_SHORT,
    SIGNAL_FORCE_EXIT
)
from tick_engine.core.metrics import compute_metrics


class TestTickEngineExecution(unittest.TestCase):

    def setUp(self):
        self.pip_size = 0.0001

    def test_long_fills_at_ask_never_bid(self):
        """Validates that Long entries fill strictly at Ask, never at Bid."""
        ticks = np.zeros(3, dtype=TICK_DTYPE)
        ticks['timestamp'] = [1000, 2000, 3000]
        # Spread is 1.0 pip (Bid: 1.1000, Ask: 1.1001)
        ticks['bid'] = [1.1000, 1.1005, 1.1005]
        ticks['ask'] = [1.1001, 1.1006, 1.1006]

        signals = np.array([SIGNAL_LONG, SIGNAL_HOLD, SIGNAL_FORCE_EXIT], dtype=np.int8)

        trades = run_matching_loop(
            ticks=ticks,
            signals=signals,
            pip_size=self.pip_size,
            sl_pips=50.0,
            tp_pips=50.0,
            close_at_end=False,
            commission_pips=0.0,
            entry_slippage_pips=0.0
        )

        self.assertEqual(len(trades), 1)
        trade = trades[0]
        self.assertEqual(trade['direction'], 1)
        # Entry price must be Ask at tick 0 (1.1001), NOT Bid (1.1000)
        self.assertAlmostEqual(trade['entry_price'], 1.1001, places=6)
        self.assertNotEqual(trade['entry_price'], 1.1000)
        # Exit price for Long signal exit must be Bid at tick 2 (1.1005)
        self.assertAlmostEqual(trade['exit_price'], 1.1005, places=6)
        # Pips: (1.1005 - 1.1001) / 0.0001 = 4.0 pips
        self.assertAlmostEqual(trade['pips'], 4.0, places=4)
        self.assertEqual(trade['exit_reason'], EXIT_REASON_SIGNAL)

    def test_short_fills_at_bid_never_ask(self):
        """Validates that Short entries fill strictly at Bid, never at Ask."""
        ticks = np.zeros(3, dtype=TICK_DTYPE)
        ticks['timestamp'] = [1000, 2000, 3000]
        # Spread is 1.0 pip (Bid: 1.2000, Ask: 1.2001)
        ticks['bid'] = [1.2000, 1.1995, 1.1995]
        ticks['ask'] = [1.2001, 1.1996, 1.1996]

        signals = np.array([SIGNAL_SHORT, SIGNAL_HOLD, SIGNAL_FORCE_EXIT], dtype=np.int8)

        trades = run_matching_loop(
            ticks=ticks,
            signals=signals,
            pip_size=self.pip_size,
            sl_pips=50.0,
            tp_pips=50.0,
            close_at_end=False,
            commission_pips=0.0,
            entry_slippage_pips=0.0
        )

        self.assertEqual(len(trades), 1)
        trade = trades[0]
        self.assertEqual(trade['direction'], -1)
        # Entry price must be Bid at tick 0 (1.2000), NOT Ask (1.2001)
        self.assertAlmostEqual(trade['entry_price'], 1.2000, places=6)
        self.assertNotEqual(trade['entry_price'], 1.2001)
        # Exit price for Short signal exit must be Ask at tick 2 (1.1996)
        self.assertAlmostEqual(trade['exit_price'], 1.1996, places=6)
        # Pips: (1.2000 - 1.1996) / 0.0001 = 4.0 pips
        self.assertAlmostEqual(trade['pips'], 4.0, places=4)
        self.assertEqual(trade['exit_reason'], EXIT_REASON_SIGNAL)

    def test_long_stop_loss_trigger(self):
        """Validates Long SL trigger when Bid <= SL Price."""
        ticks = np.zeros(3, dtype=TICK_DTYPE)
        ticks['timestamp'] = [1000, 2000, 3000]
        # Entry at Ask: 1.1000. SL 10 pips -> SL Price = 1.0990
        ticks['bid'] = [1.0999, 1.0990, 1.0980]
        ticks['ask'] = [1.1000, 1.0991, 1.0981]

        signals = np.array([SIGNAL_LONG, SIGNAL_HOLD, SIGNAL_HOLD], dtype=np.int8)

        trades = run_matching_loop(
            ticks=ticks,
            signals=signals,
            pip_size=self.pip_size,
            sl_pips=10.0,
            tp_pips=20.0,
            close_at_end=False,
            commission_pips=0.0,
            entry_slippage_pips=0.0
        )

        self.assertEqual(len(trades), 1)
        trade = trades[0]
        self.assertEqual(trade['direction'], 1)
        self.assertEqual(trade['exit_reason'], EXIT_REASON_SL)
        self.assertAlmostEqual(trade['entry_price'], 1.1000, places=6)
        self.assertAlmostEqual(trade['exit_price'], 1.0990, places=6)
        self.assertAlmostEqual(trade['pips'], -10.0, places=4)
        self.assertEqual(trade['exit_time'], 2000)

    def test_long_take_profit_trigger(self):
        """Validates Long TP trigger when Bid >= TP Price."""
        ticks = np.zeros(3, dtype=TICK_DTYPE)
        ticks['timestamp'] = [1000, 2000, 3000]
        # Entry at Ask: 1.1000. TP 15 pips -> TP Price = 1.1015
        ticks['bid'] = [1.0999, 1.1015, 1.1020]
        ticks['ask'] = [1.1000, 1.1016, 1.1021]

        signals = np.array([SIGNAL_LONG, SIGNAL_HOLD, SIGNAL_HOLD], dtype=np.int8)

        trades = run_matching_loop(
            ticks=ticks,
            signals=signals,
            pip_size=self.pip_size,
            sl_pips=10.0,
            tp_pips=15.0,
            close_at_end=False,
            commission_pips=0.0,
            entry_slippage_pips=0.0
        )

        self.assertEqual(len(trades), 1)
        trade = trades[0]
        self.assertEqual(trade['direction'], 1)
        self.assertEqual(trade['exit_reason'], EXIT_REASON_TP)
        self.assertAlmostEqual(trade['entry_price'], 1.1000, places=6)
        self.assertAlmostEqual(trade['exit_price'], 1.1015, places=6)
        self.assertAlmostEqual(trade['pips'], 15.0, places=4)

    def test_short_stop_loss_trigger(self):
        """Validates Short SL trigger when Ask >= SL Price."""
        ticks = np.zeros(3, dtype=TICK_DTYPE)
        ticks['timestamp'] = [1000, 2000, 3000]
        # Entry at Bid: 1.2000. SL 10 pips -> SL Price = 1.2010
        ticks['bid'] = [1.2000, 1.2009, 1.2020]
        ticks['ask'] = [1.2001, 1.2010, 1.2021]

        signals = np.array([SIGNAL_SHORT, SIGNAL_HOLD, SIGNAL_HOLD], dtype=np.int8)

        trades = run_matching_loop(
            ticks=ticks,
            signals=signals,
            pip_size=self.pip_size,
            sl_pips=10.0,
            tp_pips=20.0,
            close_at_end=False,
            commission_pips=0.0,
            entry_slippage_pips=0.0
        )

        self.assertEqual(len(trades), 1)
        trade = trades[0]
        self.assertEqual(trade['direction'], -1)
        self.assertEqual(trade['exit_reason'], EXIT_REASON_SL)
        self.assertAlmostEqual(trade['entry_price'], 1.2000, places=6)
        self.assertAlmostEqual(trade['exit_price'], 1.2010, places=6)
        self.assertAlmostEqual(trade['pips'], -10.0, places=4)

    def test_short_take_profit_trigger(self):
        """Validates Short TP trigger when Ask <= TP Price."""
        ticks = np.zeros(3, dtype=TICK_DTYPE)
        ticks['timestamp'] = [1000, 2000, 3000]
        # Entry at Bid: 1.2000. TP 15 pips -> TP Price = 1.1985
        ticks['bid'] = [1.2000, 1.1984, 1.1980]
        ticks['ask'] = [1.2001, 1.1985, 1.1981]

        signals = np.array([SIGNAL_SHORT, SIGNAL_HOLD, SIGNAL_HOLD], dtype=np.int8)

        trades = run_matching_loop(
            ticks=ticks,
            signals=signals,
            pip_size=self.pip_size,
            sl_pips=10.0,
            tp_pips=15.0,
            close_at_end=False,
            commission_pips=0.0,
            entry_slippage_pips=0.0
        )

        self.assertEqual(len(trades), 1)
        trade = trades[0]
        self.assertEqual(trade['direction'], -1)
        self.assertEqual(trade['exit_reason'], EXIT_REASON_TP)
        self.assertAlmostEqual(trade['entry_price'], 1.2000, places=6)
        self.assertAlmostEqual(trade['exit_price'], 1.1985, places=6)
        self.assertAlmostEqual(trade['pips'], 15.0, places=4)

    def test_reversal_trade(self):
        """Validates reversing from Long to Short directly."""
        ticks = np.zeros(3, dtype=TICK_DTYPE)
        ticks['timestamp'] = [1000, 2000, 3000]
        ticks['bid'] = [1.1000, 1.1020, 1.1010]
        ticks['ask'] = [1.1001, 1.1021, 1.1011]

        # Tick 0: Long. Tick 1: Reverse to Short. Tick 2: Exit
        signals = np.array([SIGNAL_LONG, SIGNAL_SHORT, SIGNAL_FORCE_EXIT], dtype=np.int8)

        trades = run_matching_loop(
            ticks=ticks,
            signals=signals,
            pip_size=self.pip_size,
            sl_pips=50.0,
            tp_pips=50.0,
            close_at_end=False,
            commission_pips=0.0,
            entry_slippage_pips=0.0
        )

        self.assertEqual(len(trades), 2)
        # Trade 1: Long entry 1.1001, exit at Bid 1.1020 (+19.0 pips)
        self.assertEqual(trades[0]['direction'], 1)
        self.assertAlmostEqual(trades[0]['pips'], 19.0, places=4)
        # Trade 2: Short entry at Bid 1.1020, exit at Ask 1.1011 (+9.0 pips)
        self.assertEqual(trades[1]['direction'], -1)
        self.assertAlmostEqual(trades[1]['entry_price'], 1.1020, places=6)
        self.assertAlmostEqual(trades[1]['exit_price'], 1.1011, places=6)
        self.assertAlmostEqual(trades[1]['pips'], 9.0, places=4)

    def test_metrics_evaluation(self):
        """Validates metrics calculation against deterministic pip series."""
        trades = np.zeros(4, dtype=TRADE_DTYPE)
        trades['pips'] = [20.0, -10.0, 30.0, -5.0]
        trades['direction'] = [1, 1, -1, -1]
        trades['exit_reason'] = [2, 1, 2, 1]

        m = compute_metrics(trades)
        self.assertEqual(m['total_trades'], 4)
        self.assertAlmostEqual(m['net_pips'], 35.0, places=2)
        self.assertAlmostEqual(m['win_rate_pct'], 50.0, places=1)
        # Gross profit: 50.0, Gross loss: 15.0 -> Profit factor = 50 / 15 = 3.33
        self.assertAlmostEqual(m['profit_factor'], 3.33, places=2)
        # Cum pips: [20, 10, 40, 35]. Cum max: [20, 20, 40, 40]. Drawdowns: [0, 10, 0, 5]. Max DD = 10.0
        self.assertAlmostEqual(m['max_drawdown_pips'], 10.0, places=2)


if __name__ == '__main__':
    unittest.main()
