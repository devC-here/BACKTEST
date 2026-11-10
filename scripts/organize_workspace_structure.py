#!/usr/bin/env python3
"""
================================================================================
  WORKSPACE REORGANIZATION & FOLDERIZATION SCRIPT
================================================================================
Cleans, sorts, and structures the repository into distinct institutional tiers:
1. Root: Cleaned of loose scripts, SOPs, and batch files.
2. docs/: Consolidates all SOPs, VM guides, and historical audit reports.
3. tools/: Consolidates volatility GUI and remote VM batch dispatchers.
4. results/:
   - results/archived_random_split_evaluations/: Houses all historical pre-continuous
     split results, lab scorecards, and old fleet simulations.
   - results/continuous_walkforward_audits/: Clean dedicated home for verified
     continuous chronological walk-forward audits.
5. scratch/:
   - scratch/legacy_archive/: Houses the hundreds of one-off historical scratch files.
   - scratch/: Kept clean for current active research.
================================================================================
"""

import os
import shutil
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

def reorganize():
    print("=" * 80)
    print("STARTING WORKSPACE REORGANIZATION & FOLDERIZATION")
    print("=" * 80)

    # 1. Ensure target directories exist
    docs_dir = ROOT_DIR / "docs"
    docs_dir.mkdir(exist_ok=True)
    
    tools_dir = ROOT_DIR / "tools"
    tools_dir.mkdir(exist_ok=True)
    
    scripts_dir = ROOT_DIR / "scripts"
    scripts_dir.mkdir(exist_ok=True)

    results_dir = ROOT_DIR / "results"
    results_archive_dir = results_dir / "archived_random_split_evaluations"
    results_archive_dir.mkdir(exist_ok=True)
    
    results_chrono_dir = results_dir / "continuous_walkforward_audits"
    results_chrono_dir.mkdir(exist_ok=True)

    scratch_dir = ROOT_DIR / "scratch"
    scratch_archive_dir = scratch_dir / "legacy_archive"
    scratch_archive_dir.mkdir(exist_ok=True)

    # 2. Move root markdown docs into docs/
    root_docs = [
        "AUDIT_AND_BIAS_CORRECTION_REPORT.md",
        "AZURE_VM_CREDENTIALS.md",
        "DATASET_70_30_SPLIT_STANDARD.md",
        "REMOTE_VM_COMPUTATION_HARNESS_GUIDE.md",
        "SOP_3Y_TRAIN_2Y_TEST.md",
        "SOP_NEW_SYMBOL_ALPHA_DISCOVERY.md",
        "SOP_STRATEGY_TESTING_AND_DEPLOYMENT_QUALIFICATION.md",
        "STRATEGY_CATALOG_80_CHAMPIONS.xlsx"
    ]
    for doc in root_docs:
        p = ROOT_DIR / doc
        if p.exists():
            shutil.move(str(p), str(docs_dir / doc))
            print(f"  [DOCS] Moved {doc} -> docs/")

    # 3. Move root tools & batch files into tools/
    root_tools = [
        "dispatch_to_vm.bat",
        "open_remote_vscode.bat",
        "start_volatility_gui.bat",
        "eurusd_volatility_gui.py"
    ]
    for tool in root_tools:
        p = ROOT_DIR / tool
        if p.exists():
            shutil.move(str(p), str(tools_dir / tool))
            print(f"  [TOOLS] Moved {tool} -> tools/")

    # 4. Move loose root scripts into scripts/
    root_scripts = [
        "run_5y_discovered_fleet_audit.py",
        "run_5y_fleet_audit.py",
        "run_symbol_alpha_discovery.py",
        "run_symbol_discovery_labs.py",
        "test_7_star_prop_fleet.py",
        "trade_trajectory_predictor.py"
    ]
    for script in root_scripts:
        p = ROOT_DIR / script
        if p.exists():
            shutil.move(str(p), str(scripts_dir / script))
            print(f"  [SCRIPTS] Moved {script} -> scripts/")

    # 5. Archive old results in results/
    # Preserve current valuable behavioral studies in results/
    preserved_results = {
        "pair_microstructure_behavior_profiles.json",
        "mfe_efficiency_and_harnessing_audit.json",
        "collectible_market_moves_audit.json",
        "mfe_diagnostic_results.json",
        "archived_random_split_evaluations",
        "continuous_walkforward_audits",
        "legacy_archive",
        "optimization"
    }

    for item in results_dir.iterdir():
        if item.name not in preserved_results:
            dest = results_archive_dir / item.name
            try:
                shutil.move(str(item), str(dest))
            except Exception as e:
                pass
    print(f"  [RESULTS] Archived pre-continuous random split result files into: {results_archive_dir.name}/")

    # 6. Archive historical scratch files into scratch/legacy_archive/
    # Preserve only currently relevant active tools in scratch/
    preserved_scratch = {
        "profile_pair_behavior_quant.py",
        "audit_mfe_efficiency_and_harnessing.py",
        "audit_collectible_moves_coverage.py",
        "test_proven_ny_sweep_fade_across_symbols.py",
        "legacy_archive",
        "__pycache__"
    }

    count_scratch_moved = 0
    for item in scratch_dir.iterdir():
        if item.name not in preserved_scratch and item.is_file():
            dest = scratch_archive_dir / item.name
            try:
                shutil.move(str(item), str(dest))
                count_scratch_moved += 1
            except Exception:
                pass
    print(f"  [SCRATCH] Moved {count_scratch_moved} historical scratch scripts into: scratch/legacy_archive/")

    print("\n" + "=" * 80)
    print("WORKSPACE REORGANIZATION & FOLDERIZATION COMPLETE!")
    print("=" * 80)

if __name__ == "__main__":
    reorganize()

