from pathlib import Path
import pandas as pd

RESULTS_DIR = Path("results")

records = []
for p in sorted(RESULTS_DIR.glob("*scorecard*.csv")):
    try:
        df = pd.read_csv(p)
        cols = {c.lower(): c for c in df.columns}
        pips_col = cols.get("net_pips") or cols.get("pips") or cols.get("net")
        pf_col = cols.get("profit_factor") or cols.get("pf")
        trades_col = cols.get("trades") or cols.get("total_trades")
        asset_col = cols.get("asset") or cols.get("pair")
        model_col = cols.get("model") or cols.get("strategy") or cols.get("name")
        
        if pips_col and pf_col and model_col:
            for _, r in df.iterrows():
                try:
                    pips = float(r[pips_col])
                    pf = float(r[pf_col])
                    trades = int(r[trades_col]) if trades_col and pd.notna(r[trades_col]) else 0
                    asset = str(r[asset_col]) if asset_col and pd.notna(r[asset_col]) else "UNKNOWN"
                    model = str(r[model_col])
                    if 1.15 <= pf < 1.25 and pips >= 150.0 and trades >= 50:
                        records.append({
                            "source": p.name,
                            "asset": asset,
                            "model": model,
                            "trades": trades,
                            "pips": pips,
                            "pf": pf
                        })
                except Exception:
                    pass
    except Exception:
        pass

df_res = pd.DataFrame(records)
if not df_res.empty:
    df_res = df_res.sort_values(by="pips", ascending=False).drop_duplicates(subset=["asset", "model"])
    print(f"Total 'About to become good' strategies (PF 1.15 - 1.24, Pips >= 150): {len(df_res)}\n")
    print(df_res.to_string(index=False))
else:
    print("None found")

