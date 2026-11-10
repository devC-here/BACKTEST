"""
Reporting and Export Pipeline for Backtest Results.
Exports to SQLite database, Excel workbook with styled summaries, and CSV audit logs.
"""

import os
import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd

from backtester.models import Trade


class ResultsExporter:
    def __init__(self, output_dir: Optional[Path] = None, db_path: Optional[Path] = None):
        base_dir = Path(__file__).resolve().parent.parent
        self.output_dir = output_dir or (base_dir / "results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path or (base_dir / "data" / "backtest_runs.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_db()

    def _init_db(self):
        """Initializes SQLite schema for runs and trades."""
        conn = sqlite3.connect(str(self.db_path))
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS backtest_runs (
                run_id TEXT PRIMARY KEY,
                strategy_name TEXT,
                timeframe TEXT,
                spread_pips REAL,
                weeks_tested TEXT,
                total_trades INTEGER,
                win_rate REAL,
                net_pips REAL,
                profit_factor REAL,
                max_drawdown_pips REAL,
                sharpe_ratio REAL,
                sqn REAL,
                net_usd REAL,
                executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS backtest_trades (
                trade_id TEXT PRIMARY KEY,
                run_id TEXT,
                week INTEGER,
                strategy TEXT,
                symbol TEXT,
                direction TEXT,
                entry_time TEXT,
                entry_price REAL,
                sl_price REAL,
                tp_price REAL,
                exit_time TEXT,
                exit_price REAL,
                exit_reason TEXT,
                pnl_pips REAL,
                pnl_usd REAL,
                mfe_pips REAL,
                mae_pips REAL,
                bars_held INTEGER,
                duration_minutes REAL,
                comment TEXT
            )
        """)
        conn.commit()
        conn.close()

    def save_run(
        self,
        run_id: str,
        strategy_name: str,
        timeframe: str,
        spread_pips: float,
        weeks_tested: str,
        metrics: Dict[str, Any],
        trades: List[Trade],
        weekly_summaries: Optional[List[Dict[str, Any]]] = None,
        export_excel: bool = True
    ):
        """Persists a complete backtest run across SQLite, CSV, and Excel."""
        conn = sqlite3.connect(str(self.db_path))
        cur = conn.cursor()

        # Insert run
        cur.execute("""
            INSERT OR REPLACE INTO backtest_runs (
                run_id, strategy_name, timeframe, spread_pips, weeks_tested,
                total_trades, win_rate, net_pips, profit_factor, max_drawdown_pips,
                sharpe_ratio, sqn, net_usd
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run_id, strategy_name, timeframe, spread_pips, weeks_tested,
            metrics.get("total_trades", 0), metrics.get("win_rate_pct", 0.0),
            metrics.get("net_pips", 0.0), metrics.get("profit_factor", 0.0),
            metrics.get("max_drawdown_pips", 0.0), metrics.get("sharpe_ratio", 0.0),
            metrics.get("sqn", 0.0), metrics.get("net_usd", 0.0)
        ))

        # Insert trades
        trade_dicts = [t.to_dict() for t in trades]
        for td in trade_dicts:
            cur.execute("""
                INSERT OR REPLACE INTO backtest_trades (
                    trade_id, run_id, week, strategy, symbol, direction,
                    entry_time, entry_price, sl_price, tp_price, exit_time,
                    exit_price, exit_reason, pnl_pips, pnl_usd, mfe_pips,
                    mae_pips, bars_held, duration_minutes, comment
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                td["trade_id"], run_id, td["week"], td["strategy"], td["symbol"], td["direction"],
                td["entry_time"], td["entry_price"], td["sl_price"], td["tp_price"], td["exit_time"],
                td["exit_price"], td["exit_reason"], td["pnl_pips"], td["pnl_usd"], td["mfe_pips"],
                td["mae_pips"], td["bars_held"], td["duration_minutes"], td["comment"]
            ))

        conn.commit()
        conn.close()

        # Export CSVs
        df_trades = pd.DataFrame(trade_dicts)
        csv_trades_path = self.output_dir / "latest_trades_audit.csv"
        df_trades.to_csv(csv_trades_path, index=False)

        if weekly_summaries:
            df_weekly = pd.DataFrame(weekly_summaries)
            csv_weekly_path = self.output_dir / "latest_weekly_performance.csv"
            df_weekly.to_csv(csv_weekly_path, index=False)

        # Export Excel Workbook
        if export_excel:
            excel_path = self.output_dir / "BACKTEST_REPORT.xlsx"
            with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
                # Sheet 1: Executive Summary Metrics
                summary_data = [
                    {"Metric": "Strategy Name", "Value": strategy_name},
                    {"Metric": "Timeframe", "Value": timeframe},
                    {"Metric": "Spread (pips)", "Value": spread_pips},
                    {"Metric": "Weeks Tested", "Value": weeks_tested},
                    {"Metric": "Total Trades", "Value": metrics.get("total_trades", 0)},
                    {"Metric": "Win Rate %", "Value": f"{metrics.get('win_rate_pct', 0.0):.1f}%"},
                    {"Metric": "Net Pips", "Value": metrics.get("net_pips", 0.0)},
                    {"Metric": "Profit Factor", "Value": metrics.get("profit_factor", 0.0)},
                    {"Metric": "Gross Gain (pips)", "Value": metrics.get("gross_gain_pips", 0.0)},
                    {"Metric": "Gross Loss (pips)", "Value": metrics.get("gross_loss_pips", 0.0)},
                    {"Metric": "Avg Trade (pips)", "Value": metrics.get("avg_trade_pips", 0.0)},
                    {"Metric": "Avg Win (pips)", "Value": metrics.get("avg_win_pips", 0.0)},
                    {"Metric": "Avg Loss (pips)", "Value": metrics.get("avg_loss_pips", 0.0)},
                    {"Metric": "Win / Loss Ratio", "Value": metrics.get("win_loss_ratio", 0.0)},
                    {"Metric": "Expectancy / Trade (pips)", "Value": metrics.get("expectancy_pips", 0.0)},
                    {"Metric": "Max Drawdown (pips)", "Value": metrics.get("max_drawdown_pips", 0.0)},
                    {"Metric": "Recovery Factor", "Value": metrics.get("recovery_factor", 0.0)},
                    {"Metric": "Sharpe Ratio", "Value": metrics.get("sharpe_ratio", 0.0)},
                    {"Metric": "System Quality SQN", "Value": metrics.get("sqn", 0.0)},
                    {"Metric": "Net Return (USD)", "Value": f"${metrics.get('net_usd', 0.0):,.2f}"},
                    {"Metric": "Avg Duration (min)", "Value": metrics.get("avg_duration_minutes", 0.0)},
                    {"Metric": "LONG Win Rate", "Value": f"{metrics.get('long_win_rate', 0.0):.1f}% ({metrics.get('long_trades', 0)} trds)"},
                    {"Metric": "SHORT Win Rate", "Value": f"{metrics.get('short_win_rate', 0.0):.1f}% ({metrics.get('short_trades', 0)} trds)"},
                ]
                pd.DataFrame(summary_data).to_excel(writer, sheet_name="Executive Summary", index=False)

                # Sheet 2: Weekly Breakdown
                if weekly_summaries:
                    pd.DataFrame(weekly_summaries).to_excel(writer, sheet_name="Weekly Breakdown", index=False)

                # Sheet 3: Trade Audit Log
                if not df_trades.empty:
                    df_trades.to_excel(writer, sheet_name="Trade Audit Log", index=False)

            return excel_path

        return None
