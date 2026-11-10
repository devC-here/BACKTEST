"""
Quantitative Performance Analytics and Institutional Scorecards.
All statistics strictly prioritize PIP metrics, Win Rates, and Risk/Reward Ratios.
"""

from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd

from backtester.models import Trade


class PerformanceMetrics:
    @staticmethod
    def calculate(trades: List[Trade], initial_balance: float = 10000.0) -> Dict[str, Any]:
        """Calculates comprehensive performance metrics from a list of closed trades."""
        if not trades:
            return {
                "total_trades": 0,
                "win_rate_pct": 0.0,
                "net_pips": 0.0,
                "profit_factor": 0.0,
                "gross_gain_pips": 0.0,
                "gross_loss_pips": 0.0,
                "expectancy_pips": 0.0,
                "max_drawdown_pips": 0.0,
                "sharpe_ratio": 0.0,
                "sqn": 0.0,
                "pnl_usd": 0.0
            }

        pips = np.array([t.pnl_pips for t in trades], dtype=np.float64)
        usds = np.array([t.pnl_usd for t in trades], dtype=np.float64)
        durations = [
            (t.exit_time - t.entry_time).total_seconds() / 60.0
            for t in trades if t.exit_time and t.entry_time
        ]

        n_trades = len(trades)
        wins = pips[pips > 0]
        losses = pips[pips < 0]
        evens = pips[pips == 0]

        n_wins = len(wins)
        n_losses = len(losses)
        n_evens = len(evens)
        win_rate = (n_wins / n_trades) * 100.0

        net_pips = float(np.sum(pips))
        gross_gain = float(np.sum(wins)) if n_wins > 0 else 0.0
        gross_loss = float(np.abs(np.sum(losses))) if n_losses > 0 else 0.0
        profit_factor = (gross_gain / gross_loss) if gross_loss > 0 else (999.0 if gross_gain > 0 else 0.0)

        avg_trade = float(np.mean(pips))
        avg_win = float(np.mean(wins)) if n_wins > 0 else 0.0
        avg_loss = float(np.mean(losses)) if n_losses > 0 else 0.0
        win_loss_ratio = (avg_win / abs(avg_loss)) if avg_loss != 0 else 0.0
        expectancy = (win_rate / 100.0 * avg_win) + ((1.0 - win_rate / 100.0) * avg_loss)

        # Drawdown calculation
        cumulative_pips = np.cumsum(pips)
        peak_pips = np.maximum.accumulate(cumulative_pips)
        drawdown_pips_curve = peak_pips - cumulative_pips
        max_dd_pips = float(np.max(drawdown_pips_curve)) if len(drawdown_pips_curve) > 0 else 0.0

        recovery_factor = (net_pips / max_dd_pips) if max_dd_pips > 0 else 0.0

        # Sharpe & Sortino (per-trade)
        std_pips = float(np.std(pips)) if len(pips) > 1 else 0.0
        downside_std = float(np.std(losses)) if len(losses) > 1 else 0.0

        sharpe = (avg_trade / std_pips * np.sqrt(n_trades)) if std_pips > 0 else 0.0
        sortino = (avg_trade / downside_std * np.sqrt(n_trades)) if downside_std > 0 else 0.0
        sqn = (np.sqrt(n_trades) * avg_trade / std_pips) if std_pips > 0 else 0.0

        # Long vs Short
        long_trades = [t for t in trades if t.direction == "BUY"]
        short_trades = [t for t in trades if t.direction == "SELL"]

        long_pips = sum(t.pnl_pips for t in long_trades)
        long_wins = sum(1 for t in long_trades if t.pnl_pips > 0)
        long_wr = (long_wins / len(long_trades) * 100.0) if long_trades else 0.0

        short_pips = sum(t.pnl_pips for t in short_trades)
        short_wins = sum(1 for t in short_trades if t.pnl_pips > 0)
        short_wr = (short_wins / len(short_trades) * 100.0) if short_trades else 0.0

        # Exit breakdown
        reasons = {}
        for t in trades:
            r = t.exit_reason or "UNKNOWN"
            reasons[r] = reasons.get(r, 0) + 1

        total_usd = float(np.sum(usds))
        avg_dur = float(np.mean(durations)) if durations else 0.0

        return {
            "total_trades": n_trades,
            "winning_trades": n_wins,
            "losing_trades": n_losses,
            "breakeven_trades": n_evens,
            "win_rate_pct": round(win_rate, 1),
            "net_pips": round(net_pips, 1),
            "gross_gain_pips": round(gross_gain, 1),
            "gross_loss_pips": round(gross_loss, 1),
            "profit_factor": round(profit_factor, 2),
            "avg_trade_pips": round(avg_trade, 2),
            "avg_win_pips": round(avg_win, 2),
            "avg_loss_pips": round(avg_loss, 2),
            "win_loss_ratio": round(win_loss_ratio, 2),
            "expectancy_pips": round(expectancy, 2),
            "max_drawdown_pips": round(max_dd_pips, 1),
            "recovery_factor": round(recovery_factor, 2),
            "sharpe_ratio": round(sharpe, 2),
            "sortino_ratio": round(sortino, 2),
            "sqn": round(sqn, 2),
            "net_usd": round(total_usd, 2),
            "avg_duration_minutes": round(avg_dur, 1),
            "long_trades": len(long_trades),
            "long_win_rate": round(long_wr, 1),
            "long_pips": round(long_pips, 1),
            "short_trades": len(short_trades),
            "short_win_rate": round(short_wr, 1),
            "short_pips": round(short_pips, 1),
            "exit_reasons": reasons
        }

    @staticmethod
    def format_terminal_card(metrics: Dict[str, Any], title: str = "INSTITUTIONAL FOREX SCORECARD") -> str:
        """Formats performance metrics as an ASCII scorecard."""
        n = metrics.get("total_trades", 0)
        wr = metrics.get("win_rate_pct", 0.0)
        net_p = metrics.get("net_pips", 0.0)
        pf = metrics.get("profit_factor", 0.0)
        gain = metrics.get("gross_gain_pips", 0.0)
        loss = metrics.get("gross_loss_pips", 0.0)
        dd = metrics.get("max_drawdown_pips", 0.0)
        exp = metrics.get("expectancy_pips", 0.0)
        sh = metrics.get("sharpe_ratio", 0.0)
        sqn = metrics.get("sqn", 0.0)
        usd = metrics.get("net_usd", 0.0)
        dur = metrics.get("avg_duration_minutes", 0.0)

        lines = [
            "=" * 78,
            f"  {title}",
            "=" * 78,
            f"  Total Trades:       {n:>6d}       |  Win Rate:           {wr:>6.1f}%",
            f"  Net Performance:  {net_p:>+8.1f} pips  |  Profit Factor:      {pf:>6.2f}",
            f"  Gross Gain:       {gain:>+8.1f} pips  |  Gross Loss:        {loss:>+8.1f} pips",
            f"  Average Trade:    {metrics.get('avg_trade_pips', 0.0):>+8.2f} pips  |  Expectancy / Trade: {exp:>+6.2f} pips",
            f"  Max Drawdown:      {dd:>7.1f} pips  |  Recovery Factor:    {metrics.get('recovery_factor', 0.0):>6.2f}",
            f"  Sharpe Ratio:       {sh:>6.2f}       |  System Quality SQN: {sqn:>6.2f}",
            f"  Net Return (USD): {usd:>+8.2f} $     |  Avg Trade Duration: {dur:>6.1f} min",
            "-" * 78,
            f"  LONG (BUY):  {metrics.get('long_trades', 0):>3d} trds | {metrics.get('long_win_rate', 0.0):>5.1f}% WR | Net: {metrics.get('long_pips', 0.0):>+7.1f} pips",
            f"  SHORT (SELL): {metrics.get('short_trades', 0):>3d} trds | {metrics.get('short_win_rate', 0.0):>5.1f}% WR | Net: {metrics.get('short_pips', 0.0):>+7.1f} pips",
            "-" * 78,
        ]

        # Exit reasons breakdown
        reasons = metrics.get("exit_reasons", {})
        if reasons:
            reasons_str = " | ".join([f"{k}: {v}" for k, v in sorted(reasons.items())])
            lines.append(f"  Exit Breakdown: {reasons_str}")
            lines.append("=" * 78)
        else:
            lines.append("=" * 78)

        return "\n".join(lines)

    @staticmethod
    def format_weekly_table(weekly_metrics_list: List[Dict[str, Any]]) -> str:
        """Formats a tabular week-by-week summary."""
        header = f"{'Week':<6} {'Date Range':<24} {'Trades':>7} {'Win Rate':>9} {'Net Pips':>11} {'PF':>7} {'Max DD':>9} {'Avg Win':>8} {'Avg Loss':>9}"
        sep = "-" * len(header)
        rows = [sep, header, sep]

        cum_pips = 0.0
        tot_trades = 0

        for m in weekly_metrics_list:
            w = m.get("week", 0)
            dr = f"{m.get('start_date', '')} -> {m.get('end_date', '')}"
            t = m.get("total_trades", 0)
            wr = m.get("win_rate_pct", 0.0)
            np_val = m.get("net_pips", 0.0)
            pf = m.get("profit_factor", 0.0)
            dd = m.get("max_drawdown_pips", 0.0)
            aw = m.get("avg_win_pips", 0.0)
            al = m.get("avg_loss_pips", 0.0)

            cum_pips += np_val
            tot_trades += t

            rows.append(
                f"W{w:02d}    {dr:<24} {t:>7d} {wr:>8.1f}% {np_val:>+10.1f}p {pf:>7.2f} {dd:>8.1f}p {aw:>7.1f}p {al:>8.1f}p"
            )

        rows.append(sep)
        rows.append(f"TOTAL: {len(weekly_metrics_list)} Weeks | {tot_trades:,d} Trades | Cumulative: {cum_pips:+,.1f} pips")
        rows.append(sep)
        return "\n".join(rows)
