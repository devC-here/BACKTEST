"""
Pip Analytics Reporter.
Computes institutional-grade pip performance metrics and formats ASCII scorecards.
"""

from __future__ import annotations
from typing import Dict, Any
import numpy as np


def compute_metrics(trades: np.ndarray) -> Dict[str, Any]:
    """
    Computes core summary statistics from the TRADE_DTYPE array:
    - Net Pip Return: sum(pips)
    - Win Rate: (wins / total trades) * 100%
    - Profit Factor: sum(win pips) / |sum(loss pips)|
    - Max Drawdown (Pips): Peak-to-trough decline over running cumulative sum
    """
    total_trades = len(trades)
    if total_trades == 0:
        return {
            "total_trades": 0,
            "net_pips": 0.0,
            "gross_profit_pips": 0.0,
            "gross_loss_pips": 0.0,
            "profit_factor": 0.0,
            "win_rate_pct": 0.0,
            "win_count": 0,
            "loss_count": 0,
            "breakeven_count": 0,
            "avg_trade_pips": 0.0,
            "avg_win_pips": 0.0,
            "avg_loss_pips": 0.0,
            "win_loss_ratio": 0.0,
            "max_drawdown_pips": 0.0,
            "return_to_dd": 0.0,
            "long_trades": 0,
            "long_net_pips": 0.0,
            "long_win_rate_pct": 0.0,
            "short_trades": 0,
            "short_net_pips": 0.0,
            "short_win_rate_pct": 0.0,
            "sl_exits": 0,
            "tp_exits": 0,
            "signal_exits": 0,
            "max_consecutive_wins": 0,
            "max_consecutive_losses": 0,
        }

    pips = trades['pips']
    net_pips = float(np.sum(pips))

    wins_mask = pips > 0.0
    loss_mask = pips < 0.0
    be_mask = pips == 0.0

    win_count = int(np.sum(wins_mask))
    loss_count = int(np.sum(loss_mask))
    be_count = int(np.sum(be_mask))

    win_rate_pct = (win_count / total_trades) * 100.0 if total_trades > 0 else 0.0

    gross_profit = float(np.sum(pips[wins_mask])) if win_count > 0 else 0.0
    gross_loss = float(abs(np.sum(pips[loss_mask]))) if loss_count > 0 else 0.0

    if gross_loss > 0.0:
        profit_factor = gross_profit / gross_loss
    else:
        profit_factor = float('inf') if gross_profit > 0 else 0.0

    avg_trade_pips = net_pips / total_trades
    avg_win_pips = (gross_profit / win_count) if win_count > 0 else 0.0
    avg_loss_pips = (gross_loss / loss_count) if loss_count > 0 else 0.0
    win_loss_ratio = (avg_win_pips / avg_loss_pips) if avg_loss_pips > 0 else 0.0

    # Max Drawdown (Pips): Peak-to-trough pip decline over running cumulative sum
    cum_pips = np.cumsum(pips)
    cum_max = np.maximum.accumulate(cum_pips)
    drawdowns = cum_max - cum_pips
    max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

    return_to_dd = (net_pips / max_dd) if max_dd > 0 else (net_pips if net_pips > 0 else 0.0)

    # Direction breakdown
    long_mask = trades['direction'] == 1
    short_mask = trades['direction'] == -1

    long_trades = int(np.sum(long_mask))
    short_trades = int(np.sum(short_mask))

    long_net = float(np.sum(pips[long_mask])) if long_trades > 0 else 0.0
    short_net = float(np.sum(pips[short_mask])) if short_trades > 0 else 0.0

    long_wins = int(np.sum((trades['direction'] == 1) & wins_mask))
    short_wins = int(np.sum((trades['direction'] == -1) & wins_mask))

    long_win_rate = (long_wins / long_trades * 100.0) if long_trades > 0 else 0.0
    short_win_rate = (short_wins / short_trades * 100.0) if short_trades > 0 else 0.0

    # Exit reason breakdown (1=SL, 2=TP, 3=Signal Exit)
    reasons = trades['exit_reason']
    sl_exits = int(np.sum(reasons == 1))
    tp_exits = int(np.sum(reasons == 2))
    sig_exits = int(np.sum(reasons == 3))

    # Consecutive streak tracking
    max_c_wins = 0
    max_c_losses = 0
    curr_c_wins = 0
    curr_c_losses = 0

    for p in pips:
        if p > 0:
            curr_c_wins += 1
            curr_c_losses = 0
            if curr_c_wins > max_c_wins:
                max_c_wins = curr_c_wins
        elif p < 0:
            curr_c_losses += 1
            curr_c_wins = 0
            if curr_c_losses > max_c_losses:
                max_c_losses = curr_c_losses
        else:
            curr_c_wins = 0
            curr_c_losses = 0

    return {
        "total_trades": total_trades,
        "net_pips": round(net_pips, 2),
        "gross_profit_pips": round(gross_profit, 2),
        "gross_loss_pips": round(gross_loss, 2),
        "profit_factor": round(profit_factor, 2) if profit_factor != float('inf') else "Inf",
        "win_rate_pct": round(win_rate_pct, 1),
        "win_count": win_count,
        "loss_count": loss_count,
        "breakeven_count": be_count,
        "avg_trade_pips": round(avg_trade_pips, 2),
        "avg_win_pips": round(avg_win_pips, 2),
        "avg_loss_pips": round(avg_loss_pips, 2),
        "win_loss_ratio": round(win_loss_ratio, 2),
        "max_drawdown_pips": round(max_dd, 2),
        "return_to_dd": round(return_to_dd, 2),
        "long_trades": long_trades,
        "long_net_pips": round(long_net, 2),
        "long_win_rate_pct": round(long_win_rate, 1),
        "short_trades": short_trades,
        "short_net_pips": round(short_net, 2),
        "short_win_rate_pct": round(short_win_rate, 1),
        "sl_exits": sl_exits,
        "tp_exits": tp_exits,
        "signal_exits": sig_exits,
        "max_consecutive_wins": max_c_wins,
        "max_consecutive_losses": max_c_losses,
    }


def print_scorecard(metrics: Dict[str, Any], title: str = "TICK BACKTEST SCORECARD") -> None:
    """
    Renders an ASCII scorecard summarizing pip metrics.
    """
    border = "=" * 65
    sub_border = "-" * 65

    print("\n" + border)
    print(f" {title.center(63)} ")
    print(border)

    print(f"  Total Trades        : {metrics['total_trades']:<12} "
          f"| Win Rate      : {metrics['win_rate_pct']}%")
    print(f"  Winning Trades      : {metrics['win_count']:<12} "
          f"| Losing Trades : {metrics['loss_count']}")
    print(f"  Breakeven Trades    : {metrics['breakeven_count']:<12} "
          f"| Profit Factor : {metrics['profit_factor']}")

    print(sub_border)
    sign = "+" if metrics['net_pips'] > 0 else ""
    print(f"  NET RETURN (PIPS)   : {sign}{metrics['net_pips']:<12} "
          f"| Max Drawdown  : {metrics['max_drawdown_pips']} pips")
    print(f"  Gross Profit (Pips) : +{metrics['gross_profit_pips']:<11} "
          f"| Gross Loss    : -{metrics['gross_loss_pips']} pips")
    print(f"  Average Trade       : {metrics['avg_trade_pips']:<12} pips "
          f"| Ret / Max DD  : {metrics['return_to_dd']}")

    print(sub_border)
    print(f"  Avg Win / Loss      : +{metrics['avg_win_pips']} / -{metrics['avg_loss_pips']} pips "
          f"| Win/Loss Ratio: {metrics['win_loss_ratio']}")
    print(f"  Max Win Streak      : {metrics['max_consecutive_wins']:<12} "
          f"| Max Loss Strk : {metrics['max_consecutive_losses']}")

    print(sub_border)
    print(f"  Long Trades         : {metrics['long_trades']} ({metrics['long_win_rate_pct']}% win) "
          f"-> Net: {metrics['long_net_pips']} pips")
    print(f"  Short Trades        : {metrics['short_trades']} ({metrics['short_win_rate_pct']}% win) "
          f"-> Net: {metrics['short_net_pips']} pips")

    print(sub_border)
    tot = max(1, metrics['total_trades'])
    print(f"  Exit Types          : SL: {metrics['sl_exits']} ({metrics['sl_exits']/tot*100:.1f}%) | "
          f"TP: {metrics['tp_exits']} ({metrics['tp_exits']/tot*100:.1f}%) | "
          f"Signal: {metrics['signal_exits']} ({metrics['signal_exits']/tot*100:.1f}%)")
    print(border + "\n")


def print_weekly_progress_header() -> None:
    """Prints tabular header for batch weekly progress."""
    header = (
        f"{'Week':<6} | {'Dates':<23} | {'Ticks':<9} | {'Trades':<7} | "
        f"{'Win%':<6} | {'Net Pips':<10} | {'Max DD':<9} | {'PF':<6} | {'Time':<7}"
    )
    div = "-" * len(header)
    print("\n" + div)
    print(header)
    print(div)


def print_weekly_progress_row(week: int, meta: dict, metrics: dict, elapsed: float) -> None:
    """Prints a clean tabular row for a single week's batch results."""
    date_str = f"{meta.get('start_date', 'N/A')} -> {meta.get('end_date', 'N/A')}"
    sign = "+" if metrics['net_pips'] > 0 else ""
    net_str = f"{sign}{metrics['net_pips']:.1f}"
    ticks = meta.get('tick_count', 0)
    tick_str = f"{ticks:,}" if ticks > 0 else "N/A"
    pf = metrics['profit_factor']
    pf_str = f"{pf:.2f}" if isinstance(pf, (int, float)) else str(pf)

    row = (
        f"W{week:02d}    | {date_str:<23} | {tick_str:<9} | {metrics['total_trades']:<7} | "
        f"{metrics['win_rate_pct']:>5.1f}% | {net_str:>10} | {metrics['max_drawdown_pips']:>7.1f}p | "
        f"{pf_str:>6} | {elapsed:>6.3f}s"
    )
    print(row, flush=True)


def compute_annual_metrics(all_trades: np.ndarray, weekly_records: list[dict]) -> dict:
    """
    Computes annual metrics combining tick-level trade logs with weekly distribution stats.
    """
    base_metrics = compute_metrics(all_trades)

    weekly_pips = [r['metrics']['net_pips'] for r in weekly_records]
    total_weeks = len(weekly_records)
    win_weeks = sum(1 for p in weekly_pips if p > 0)
    loss_weeks = sum(1 for p in weekly_pips if p < 0)
    flat_weeks = sum(1 for p in weekly_pips if p == 0)

    win_week_pct = (win_weeks / total_weeks * 100.0) if total_weeks > 0 else 0.0

    best_w = max(weekly_records, key=lambda r: r['metrics']['net_pips']) if weekly_records else None
    worst_w = min(weekly_records, key=lambda r: r['metrics']['net_pips']) if weekly_records else None

    avg_weekly_pips = float(np.mean(weekly_pips)) if weekly_pips else 0.0
    std_weekly_pips = float(np.std(weekly_pips)) if len(weekly_pips) > 1 else 0.0

    # Annualized weekly Sharpe approximation (assuming 52 weeks, 0% risk free)
    weekly_sharpe = (avg_weekly_pips / std_weekly_pips * np.sqrt(52)) if std_weekly_pips > 0 else 0.0

    base_metrics.update({
        "total_weeks": total_weeks,
        "win_weeks": win_weeks,
        "loss_weeks": loss_weeks,
        "flat_weeks": flat_weeks,
        "win_week_pct": round(win_week_pct, 1),
        "avg_weekly_pips": round(avg_weekly_pips, 2),
        "std_weekly_pips": round(std_weekly_pips, 2),
        "weekly_sharpe": round(weekly_sharpe, 2),
        "best_week": {
            "week": best_w['week'] if best_w else 0,
            "pips": best_w['metrics']['net_pips'] if best_w else 0.0
        },
        "worst_week": {
            "week": worst_w['week'] if worst_w else 0,
            "pips": worst_w['metrics']['net_pips'] if worst_w else 0.0
        }
    })
    return base_metrics


def print_annual_scorecard(annual_metrics: dict, title: str = "ANNUAL 52-WEEK BACKTEST SCORECARD") -> None:
    """
    Renders an ASCII scorecard summarizing full-year performance.
    """
    border = "=" * 70
    sub_border = "-" * 70

    print("\n" + border)
    print(f" {title.center(68)} ")
    print(border)

    print(f"  Weeks Tested        : {annual_metrics['total_weeks']:<12} "
          f"| Total Trades  : {annual_metrics['total_trades']}")
    print(f"  Profitable Weeks    : {annual_metrics['win_weeks']} ({annual_metrics['win_week_pct']}%) "
          f"| Losing Weeks  : {annual_metrics['loss_weeks']}")
    print(f"  Best Week (Pips)    : W{annual_metrics['best_week']['week']:02d} ({annual_metrics['best_week']['pips']:+.1f}p) "
          f"| Worst Week    : W{annual_metrics['worst_week']['week']:02d} ({annual_metrics['worst_week']['pips']:+.1f}p)")

    print(sub_border)
    sign = "+" if annual_metrics['net_pips'] > 0 else ""
    print(f"  CUMULATIVE RETURN   : {sign}{annual_metrics['net_pips']:<12} pips "
          f"| Max Drawdown  : {annual_metrics['max_drawdown_pips']} pips")
    print(f"  Annual Win Rate     : {annual_metrics['win_rate_pct']}% "
          f"| Profit Factor : {annual_metrics['profit_factor']}")
    print(f"  Avg Weekly Return   : {annual_metrics['avg_weekly_pips']:+.2f} pips "
          f"| Weekly Sharpe : {annual_metrics['weekly_sharpe']}")

    print(sub_border)
    print(f"  Gross Profit (Pips) : +{annual_metrics['gross_profit_pips']:<11} "
          f"| Gross Loss    : -{annual_metrics['gross_loss_pips']} pips")
    print(f"  Avg Trade Pips      : {annual_metrics['avg_trade_pips']:<12} "
          f"| Ret / Max DD  : {annual_metrics['return_to_dd']}")
    print(f"  Long Pips (Trades)  : {annual_metrics['long_net_pips']:+.1f}p ({annual_metrics['long_trades']}) "
          f"| Short Pips    : {annual_metrics['short_net_pips']:+.1f}p ({annual_metrics['short_trades']})")

    print(sub_border)
    tot = max(1, annual_metrics['total_trades'])
    print(f"  Exits Breakdown     : SL: {annual_metrics['sl_exits']} ({annual_metrics['sl_exits']/tot*100:.1f}%) | "
          f"TP: {annual_metrics['tp_exits']} ({annual_metrics['tp_exits']/tot*100:.1f}%) | "
          f"Signal: {annual_metrics['signal_exits']} ({annual_metrics['signal_exits']/tot*100:.1f}%)")
    print(border + "\n")

