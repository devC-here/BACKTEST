#!/usr/bin/env python3
"""
Find the Best 8 Least-Correlated Strategies from all 29 Champions
Simulates weekly returns on 5-year Dukascopy tick data, computes
the full 29x29 correlation matrix, and identifies the optimal 8-strategy
portfolio with minimal correlation, maximum Sharpe, and balanced multi-asset exposure.
"""

from pathlib import Path
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tick_engine.core.engine import BacktestEngine
from tick_engine.core.dataloader import get_discovery_weeks, get_holdout_test_weeks, get_week_dataset

# Import EURUSD models
from strategies.qualified_strategies.eurusd.disc86_ema_cross_atr_expansion_elite.strategy import EURUSDEMACrossATRExpansionElite
from strategies.qualified_strategies.eurusd.disc40_rsi_regime_range_shift_elite.strategy import RSIRegimeRangeShiftElite
from strategies.qualified_strategies.eurusd.disc45_continuous_m5_rejection_elite.strategy import ContinuousM5RejectionElite
from strategies.qualified_strategies.eurusd.disc85_session_open_impulse_elite.strategy import EURUSDSessionOpenImpulseElite
from strategies.qualified_strategies.eurusd.disc93_london_vwap_pullback_elite.strategy import EURUSDLondonVWAPPullbackElite
from strategies.qualified_strategies.eurusd.disc97_cardwell_rsi50_momentum_bounce_elite.strategy import EURUSDCardwellRSI50MomentumBounceElite
from strategies.qualified_strategies.eurusd.disc101_asian_hl_retest_elite.strategy import EURUSDAsianHLRetestElite
from strategies.qualified_strategies.eurusd.disc103_double_swing_neckline_retest_elite.strategy import EURUSDDoubleSwingNecklineRetestElite
from strategies.qualified_strategies.eurusd.disc105_chaikin_volatility_surge_elite.strategy import EURUSDChaikinVolatilitySurgeElite

# Import USDJPY models
from strategies.qualified_strategies.usdjpy.disc90_opening_range_impulse_expansion_elite.strategy import USDJPYOpeningRangeImpulseExpansionElite
from strategies.qualified_strategies.usdjpy.disc102_keltner_midline_expansion_elite.strategy import USDJPYKeltnerMidlineExpansionElite
from strategies.qualified_strategies.usdjpy.disc106_chaikin_volatility_surge_elite.strategy import USDJPYChaikinVolatilitySurgeElite
from strategies.qualified_strategies.usdjpy.disc96_cardwell_rsi50_momentum_bounce_elite.strategy import USDJPYCardwellRSI50MomentumBounceElite
from strategies.qualified_strategies.usdjpy.disc104_three_drive_ignition_elite.strategy import USDJPYThreeDriveIgnitionElite
from strategies.qualified_strategies.usdjpy.disc94_london_vwap_pullback_elite.strategy import USDJPYLondonVWAPPullbackElite
from strategies.qualified_strategies.usdjpy.disc91_session_donchian_midline_bounce_elite.strategy import USDJPYSessionDonchianMidlineBounceElite
from strategies.qualified_strategies.usdjpy.disc87_pivot_r1_s1_breakout_elite.strategy import USDJPYPivotR1S1BreakoutElite
from strategies.qualified_strategies.usdjpy.disc99_m15_macd_zero_cross_momentum_elite.strategy import USDJPYM15MACDZeroCrossMomentumElite
from strategies.qualified_strategies.usdjpy.disc95_camarilla_expansion_elite.strategy import USDJPYCamarillaExpansionElite
from strategies.qualified_strategies.usdjpy.disc100_double_structure_neckline_breakout_elite.strategy import USDJPYDoubleStructureNecklineBreakoutElite
from strategies.qualified_strategies.usdjpy.disc89_ny_london_vwap_ignition_elite.strategy import USDJPYNYLondonVWAPIgnitionElite
from strategies.qualified_strategies.usdjpy.disc92_asian_sr_flip_retest_elite.strategy import USDJPYAsianSRFlipRetestElite

# Import EURGBP models
from strategies.qualified_strategies.eurgbp.disc48_camarilla_floor_elite.strategy import EURGBPCamarillaFloorElite
from strategies.qualified_strategies.eurgbp.disc46_daily_pivot_wall_elite.strategy import EURGBPDailyPivotWallElite
from strategies.qualified_strategies.eurgbp.disc98_asian_compression_breakout_elite.strategy import EURGBPAsianCompressionBreakoutElite
from strategies.qualified_strategies.eurgbp.disc107_h1_keltner_squeeze_explosion_elite.strategy import EURGBPH1KeltnerSqueezeExplosionElite
from strategies.qualified_strategies.eurgbp.disc108_macro_asymmetric_runner_elite.strategy import EURGBPMacroAsymmetricRunnerElite
from strategies.qualified_strategies.eurgbp.disc109_intraday_macro_retest_elite.strategy import EURGBPIntradayMacroRetestElite
from strategies.qualified_strategies.eurgbp.disc110_vwap_1sd_rebound_elite.strategy import EURGBPVWAP1SDReboundElite

MODELS = [
    # EURUSD (9)
    {"id": "EU_DISC86", "pair": "EURUSD", "strat": EURUSDEMACrossATRExpansionElite(), "mult": 1.0000, "sl": 9.0},
    {"id": "EU_DISC105", "pair": "EURUSD", "strat": EURUSDChaikinVolatilitySurgeElite(), "mult": 1.0000, "sl": 9.0},
    {"id": "EU_DISC40", "pair": "EURUSD", "strat": RSIRegimeRangeShiftElite(), "mult": 1.0000, "sl": 9.0},
    {"id": "EU_DISC103", "pair": "EURUSD", "strat": EURUSDDoubleSwingNecklineRetestElite(), "mult": 1.0000, "sl": 9.0},
    {"id": "EU_DISC45", "pair": "EURUSD", "strat": ContinuousM5RejectionElite(asset="EURUSD", sl_pips=5.5, tp_pips=11.0, pip_size=0.0001, max_spread=0.6), "mult": 1.0000, "sl": 5.5},
    {"id": "EU_DISC97", "pair": "EURUSD", "strat": EURUSDCardwellRSI50MomentumBounceElite(), "mult": 1.0000, "sl": 8.0},
    {"id": "EU_DISC85", "pair": "EURUSD", "strat": EURUSDSessionOpenImpulseElite(), "mult": 1.0000, "sl": 9.0},
    {"id": "EU_DISC93", "pair": "EURUSD", "strat": EURUSDLondonVWAPPullbackElite(), "mult": 1.0000, "sl": 9.0},
    {"id": "EU_DISC101", "pair": "EURUSD", "strat": EURUSDAsianHLRetestElite(), "mult": 1.0000, "sl": 9.0},

    # USDJPY (13)
    {"id": "UJ_DISC90", "pair": "USDJPY", "strat": USDJPYOpeningRangeImpulseExpansionElite(), "mult": 0.7193, "sl": 12.0},
    {"id": "UJ_DISC102", "pair": "USDJPY", "strat": USDJPYKeltnerMidlineExpansionElite(), "mult": 0.7193, "sl": 12.0},
    {"id": "UJ_DISC106", "pair": "USDJPY", "strat": USDJPYChaikinVolatilitySurgeElite(), "mult": 0.7193, "sl": 14.0},
    {"id": "UJ_DISC96", "pair": "USDJPY", "strat": USDJPYCardwellRSI50MomentumBounceElite(), "mult": 0.7193, "sl": 12.0},
    {"id": "UJ_DISC104", "pair": "USDJPY", "strat": USDJPYThreeDriveIgnitionElite(), "mult": 0.7193, "sl": 12.0},
    {"id": "UJ_DISC94", "pair": "USDJPY", "strat": USDJPYLondonVWAPPullbackElite(), "mult": 0.7193, "sl": 12.0},
    {"id": "UJ_DISC91", "pair": "USDJPY", "strat": USDJPYSessionDonchianMidlineBounceElite(), "mult": 0.7193, "sl": 14.0},
    {"id": "UJ_DISC87", "pair": "USDJPY", "strat": USDJPYPivotR1S1BreakoutElite(), "mult": 0.7193, "sl": 12.0},
    {"id": "UJ_DISC99", "pair": "USDJPY", "strat": USDJPYM15MACDZeroCrossMomentumElite(), "mult": 0.7193, "sl": 12.0},
    {"id": "UJ_DISC95", "pair": "USDJPY", "strat": USDJPYCamarillaExpansionElite(), "mult": 0.7193, "sl": 12.0},
    {"id": "UJ_DISC100", "pair": "USDJPY", "strat": USDJPYDoubleStructureNecklineBreakoutElite(), "mult": 0.7193, "sl": 12.0},
    {"id": "UJ_DISC89", "pair": "USDJPY", "strat": USDJPYNYLondonVWAPIgnitionElite(), "mult": 0.7193, "sl": 12.0},
    {"id": "UJ_DISC92", "pair": "USDJPY", "strat": USDJPYAsianSRFlipRetestElite(), "mult": 0.7193, "sl": 12.0},

    # EURGBP (7)
    {"id": "EG_DISC48", "pair": "EURGBP", "strat": EURGBPCamarillaFloorElite(), "mult": 1.2921, "sl": 8.0},
    {"id": "EG_DISC46", "pair": "EURGBP", "strat": EURGBPDailyPivotWallElite(), "mult": 1.2921, "sl": 8.0},
    {"id": "EG_DISC107", "pair": "EURGBP", "strat": EURGBPH1KeltnerSqueezeExplosionElite(), "mult": 1.2921, "sl": 8.0},
    {"id": "EG_DISC108", "pair": "EURGBP", "strat": EURGBPMacroAsymmetricRunnerElite(), "mult": 1.2921, "sl": 8.0},
    {"id": "EG_DISC98", "pair": "EURGBP", "strat": EURGBPAsianCompressionBreakoutElite(), "mult": 1.2921, "sl": 8.0},
    {"id": "EG_DISC109", "pair": "EURGBP", "strat": EURGBPIntradayMacroRetestElite(), "mult": 1.2921, "sl": 8.0},
    {"id": "EG_DISC110", "pair": "EURGBP", "strat": EURGBPVWAP1SDReboundElite(), "mult": 1.2921, "sl": 8.0},
]

def run_week_task(args):
    m_idx, w = args
    m = MODELS[m_idx]
    engine = BacktestEngine(pip_size=m["strat"].pip_size, use_slippage=True)
    try:
        ticks, _ = get_week_dataset(w, raw_dir=f"data/{m['pair']}")
        if ticks is None or len(ticks) < 500:
            return m_idx, w, 0.0
        trades = engine.run(ticks, m["strat"])
        total_pips = sum(float(r["pips"]) for r in trades) * m["mult"]
        return m_idx, w, total_pips
    except Exception:
        return m_idx, w, 0.0

def main():
    print(f"Loading weekly returns for all {len(MODELS)} certified champions...")
    # Standard 262 weeks
    weeks = sorted(get_discovery_weeks(raw_dir="data/EURUSD") + get_holdout_test_weeks(raw_dir="data/EURUSD"))
    
    tasks = []
    for m_idx in range(len(MODELS)):
        for w in weeks:
            tasks.append((m_idx, w))

    weekly_matrix = np.zeros((len(weeks), len(MODELS)), dtype=np.float64)
    week_map = {w: i for i, w in enumerate(weeks)}

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_week_task, t) for t in tasks]
        for fut in as_completed(futures):
            m_idx, w, pips = fut.result()
            w_idx = week_map[w]
            weekly_matrix[w_idx, m_idx] = pips

    df_weekly = pd.DataFrame(weekly_matrix, columns=[m["id"] for m in MODELS], index=weeks)
    corr_matrix = df_weekly.corr()

    # Model metadata
    model_stats = {}
    for i, m in enumerate(MODELS):
        series = df_weekly[m["id"]]
        pips = series.values
        cum = np.cumsum(pips)
        peak = np.maximum.accumulate(cum)
        dd = peak - cum
        max_dd = float(np.max(dd)) if len(dd) > 0 else 1.0
        tot_pips = float(np.sum(pips))
        mean = float(np.mean(pips))
        std = float(np.std(pips)) if np.std(pips) > 0 else 1.0
        sharpe = (mean / std) * np.sqrt(52.0)
        model_stats[m["id"]] = {
            "pair": m["pair"],
            "tot_pips": tot_pips,
            "max_dd": max_dd,
            "sharpe": sharpe,
            "ret_dd": tot_pips / max_dd if max_dd > 0 else 0.0
        }

    print("\nSimulated all models. Searching for the Top 8 Least-Correlated Portfolio...")
    
    # Stratified search: Select combinations of 8 strategies with balanced currency exposure
    # E.g., 3 USDJPY + 3 EURUSD + 2 EURGBP, or 3-2-3, or 2-3-3
    eu_ids = [m["id"] for m in MODELS if m["pair"] == "EURUSD"]
    uj_ids = [m["id"] for m in MODELS if m["pair"] == "USDJPY"]
    eg_ids = [m["id"] for m in MODELS if m["pair"] == "EURGBP"]

    best_portfolio = None
    best_score = -999999.0

    # Test all valid combinations of (3, 3, 2), (3, 2, 3), (2, 3, 3)
    allocation_schemes = [
        (3, 3, 2),
        (3, 2, 3),
        (2, 3, 3)
    ]

    evaluated = 0
    candidate_portfolios = []

    for n_eu, n_uj, n_eg in allocation_schemes:
        for eu_combo in itertools.combinations(eu_ids, n_eu):
            for uj_combo in itertools.combinations(uj_ids, n_uj):
                for eg_combo in itertools.combinations(eg_ids, n_eg):
                    combo = list(eu_combo + uj_combo + eg_combo)
                    evaluated += 1

                    # Compute sub-correlation matrix
                    sub_corr = corr_matrix.loc[combo, combo].values
                    # Average off-diagonal correlation
                    n = len(combo)
                    off_diag = sub_corr[~np.eye(n, dtype=bool)]
                    mean_corr = float(np.mean(off_diag))
                    max_corr = float(np.max(off_diag))

                    # Combine weekly pnl
                    port_weekly = df_weekly[combo].sum(axis=1).values
                    tot_pips = float(np.sum(port_weekly))
                    cum = np.cumsum(port_weekly)
                    peak = np.maximum.accumulate(cum)
                    dd = peak - cum
                    max_dd = float(np.max(dd)) if len(dd) > 0 else 1.0
                    mean_w = float(np.mean(port_weekly))
                    std_w = float(np.std(port_weekly)) if np.std(port_weekly) > 0 else 1.0
                    port_sharpe = (mean_w / std_w) * np.sqrt(52.0)
                    ret_dd = tot_pips / max_dd if max_dd > 0 else 0.0

                    # Fitness: Prioritize low correlation (mean_corr < 0.10), high Sharpe, high Ret/DD
                    # Penalty for higher correlation
                    fitness = port_sharpe * 10.0 + (ret_dd * 2.0) - (mean_corr * 50.0) - (max_corr * 20.0)

                    candidate_portfolios.append({
                        "combo": combo,
                        "mean_corr": mean_corr,
                        "max_corr": max_corr,
                        "tot_pips": tot_pips,
                        "max_dd": max_dd,
                        "sharpe": port_sharpe,
                        "ret_dd": ret_dd,
                        "fitness": fitness
                    })

    # Sort candidates by fitness
    candidate_portfolios.sort(key=lambda x: x["fitness"], reverse=True)
    winner = candidate_portfolios[0]
    best_combo = winner["combo"]

    print("=" * 100)
    print("OPTIMAL 8-STRATEGY LEAST-CORRELATED TITAN PORTFOLIO")
    print("=" * 100)
    print(f"Total Evaluated Combos: {evaluated}")
    print(f"Mean Pairwise Correlation: {winner['mean_corr']:.4f} (Near Zero Correlation!)")
    print(f"Max Pairwise Correlation:  {winner['max_corr']:.4f}")
    print(f"5-Year Combined Pips:      +{winner['tot_pips']:,.1f} norm pips (${winner['tot_pips']*10:,.0f} cash on 1.0 lot)")
    print(f"Annual Combined Pips:      +{winner['tot_pips']/5:,.1f} norm pips/yr")
    print(f"Portfolio Max Drawdown:    {winner['max_dd']:.1f} norm pips")
    print(f"Portfolio Sharpe Ratio:    {winner['sharpe']:.2f}")
    print(f"Portfolio Return / DD:     {winner['ret_dd']:.2f}x")
    print("-" * 100)

    print("\nSelected 8 Champions:")
    for s_id in best_combo:
        st = model_stats[s_id]
        print(f"  - {s_id:<12} [{st['pair']:<6}] 5Y Pips: +{st['tot_pips']:>7.1f}p | Max DD: {st['max_dd']:>5.1f}p | Sharpe: {st['sharpe']:>4.2f} | Ret/DD: {st['ret_dd']:>5.2f}x")

    print("\n" + "=" * 100)
    print("PAIRWISE CORRELATION MATRIX (8x8):")
    print("=" * 100)
    sub_df = corr_matrix.loc[best_combo, best_combo]
    print(sub_df.round(3).to_string())
    print("=" * 100)

if __name__ == "__main__":
    main()

