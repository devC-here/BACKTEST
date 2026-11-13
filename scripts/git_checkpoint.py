#!/usr/bin/env python3
"""
================================================================================
  AUTOMATED PROJECT TREE & PROMPT-LEVEL GIT CHECKPOINT GENERATOR
================================================================================
Maintains a living record of project folder architecture, active 71-strategy
fleet roster, and turn-by-turn prompt changes in docs/PROJECT_TREE_AND_CHANGELOG.md,
while automatically creating clean Git commits per prompt/milestone.

Usage:
  python scripts/git_checkpoint.py --prompt "..." --summary "..."
================================================================================
"""

import os
import sys
import json
import argparse
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT_DIR / "docs"
CHANGELOG_PATH = DOCS_DIR / "PROJECT_TREE_AND_CHANGELOG.md"
MANIFEST_PATH = ROOT_DIR / "certified_production_fleet" / "CERTIFIED_FLEET_MANIFEST.json"

IGNORE_DIRS = {
    ".git", ".venv", "venv", "ENV", "env", "__pycache__",
    "node_modules", ".pytest_cache", ".cache", "data",
    "tick_engine/data", "weekly_data", "scratch", "dist", "build"
}

IGNORE_EXTENSIONS = {
    ".pyc", ".pyo", ".pyd", ".so", ".dll", ".npy", ".csv.gz", ".tmp", ".log"
}

def generate_visual_tree(dir_path: Path, max_depth: int = 2, current_depth: int = 0, prefix: str = "") -> List[str]:
    """Generates a clean ASCII tree representation of the directory structure."""
    if current_depth > max_depth:
        return []

    lines = []
    try:
        entries = sorted(list(dir_path.iterdir()), key=lambda x: (not x.is_dir(), x.name.lower()))
    except PermissionError:
        return []

    # Filter ignored entries
    filtered_entries = []
    for e in entries:
        if e.name in IGNORE_DIRS or e.name.startswith("."):
            continue
        if e.is_file() and e.suffix in IGNORE_EXTENSIONS:
            continue
        filtered_entries.append(e)

    total = len(filtered_entries)
    
    # If a directory has many files (e.g. v4.6 or scripts with > 20 files), summarize them cleanly
    if dir_path != ROOT_DIR and total > 20 and all(e.is_file() for e in filtered_entries):
        sample_files = [e.name for e in filtered_entries[:5]]
        lines.append(f"{prefix}├── [{total} files: {', '.join(sample_files)}, ...]")
        return lines

    for i, entry in enumerate(filtered_entries):
        is_last = (i == total - 1)
        connector = "└── " if is_last else "├── "
        
        if entry.is_dir():
            # Count subfiles
            sub_count = len([x for x in entry.iterdir() if x.name not in IGNORE_DIRS and not x.name.startswith(".")])
            lines.append(f"{prefix}{connector}{entry.name}/ ({sub_count} items)")
            if current_depth < max_depth:
                sub_prefix = prefix + ("    " if is_last else "│   ")
                lines.extend(generate_visual_tree(entry, max_depth=max_depth, current_depth=current_depth + 1, prefix=sub_prefix))
        else:
            lines.append(f"{prefix}{connector}{entry.name}")

    return lines


def load_fleet_roster() -> List[Dict[str, Any]]:
    """Loads the certified 71-strategy fleet manifest."""
    if not MANIFEST_PATH.exists():
        return []
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data.get("strategies", [])
    elif isinstance(data, list):
        return data
    return []


def run_cmd(cmd: List[str]) -> Tuple[int, str, str]:
    """Runs a shell command synchronously."""
    res = subprocess.run(cmd, cwd=str(ROOT_DIR), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
    return res.returncode, res.stdout.strip(), res.stderr.strip()


def update_changelog(prompt: str, summary: str, version: str = "v4.6.1", commit_hash: str = "Pending"):
    """Creates or updates the docs/PROJECT_TREE_AND_CHANGELOG.md file."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    
    roster = load_fleet_roster()
    tree_lines = generate_visual_tree(ROOT_DIR, max_depth=2)
    tree_text = "\n".join(tree_lines)
    
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Strategy distribution summary
    pair_counts = {}
    total_holdout_pips = 0.0
    for s in roster:
        sym = s.get("symbol", "UNKNOWN")
        pair_counts[sym] = pair_counts.get(sym, 0) + 1
        hm = s.get("holdout_metrics", {})
        total_holdout_pips += hm.get("net_pips", 0.0)

    distribution_str = ", ".join([f"**{k}**: {v}" for k, v in sorted(pair_counts.items())])

    # Check if file exists to read previous entries
    existing_entries = ""
    if CHANGELOG_PATH.exists():
        content = CHANGELOG_PATH.read_text(encoding="utf-8", errors="replace")
        split_marker = "## 📜 Turn-by-Turn Prompt & Milestone Changelog"
        if split_marker in content:
            existing_entries = content.split(split_marker)[-1].strip()

    new_turn_entry = f"""
### Turn Checkpoint: {timestamp_str} | Commit: `{commit_hash}`
- **Version**: `{version}`
- **User Prompt / Objective**:
  > {prompt}
- **Actions Executed & Technical Changes**:
{summary}
- **Fleet State**: {len(roster)} Certified Strategies | Holdout PnL: **+{total_holdout_pips:,.1f} pips**
- **Pairs**: {distribution_str}

---
"""

    full_document = f"""# 🌳 Project Directory Tree & Turn-by-Turn Changelog
*Automated Living Architecture & Version Control Tracker for BACKTEST Platform*

> [!NOTE]
> This document is automatically maintained by [`scripts/git_checkpoint.py`](file:///d:/PROJECT/TRD/MODELS%20GENERATIONS/BACKTEST/scripts/git_checkpoint.py).
> Every major prompt, optimization, and strategy discovery is committed to Git and logged here with an updated visual directory tree.

---

## 📁 1. Active Project Directory Architecture
```text
BACKTEST/
{tree_text}
```

---

## 🛡️ 2. Certified Production Fleet Overview ({len(roster)} Active Models)
- **Total Certified Strategies**: **{len(roster)}** (100% Dual-Certified & Quarantined Holdout Validated)
- **Currency Pair Distribution**: {distribution_str}
- **Aggregate Holdout Profit (W211..300)**: **+{total_holdout_pips:,.1f} Net Pips**
- **Friction Model**: Realistic ECN Broker Model ($4.00/lot commission + 0.20p adverse entry slippage + raw floating spread)
- **Execution Engine**: Subsequent-tick causal execution ($T_{{\\text{{fill}}}} > T_{{\\text{{signal}}}}$) + 100k prior-week warmup

| Slot | Symbol | Strategy Name | Category | Holdout Pips | Holdout PF | Win Rate | Max DD (pips) |
| :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: |
"""

    for i, s in enumerate(roster, 1):
        slot = i
        sym = s.get("symbol", "")
        name = s.get("name", "")
        cat = s.get("category", "")
        hm = s.get("holdout_metrics", {})
        hpips = hm.get("net_pips", 0.0)
        hpf = hm.get("profit_factor", 0.0)
        hwr = hm.get("win_rate_pct", 0.0)
        hdd = hm.get("max_drawdown_pips", 0.0)
        full_document += f"| {slot:02d} | `{sym}` | `{name}` | {cat} | **+{hpips:.1f}** | **{hpf:.2f}** | {hwr:.1f}% | {hdd:.1f} |\n"

    full_document += f"""

---

## 📜 Turn-by-Turn Prompt & Milestone Changelog

{new_turn_entry}
{existing_entries}
"""

    CHANGELOG_PATH.write_text(full_document, encoding="utf-8")
    print(f"[OK] Successfully updated {CHANGELOG_PATH}")


def main():
    parser = argparse.ArgumentParser(description="Git Checkpoint & Project Tree Tracker")
    parser.add_argument("--prompt", type=str, default="Interactive system development & optimization", help="User prompt or turn objective")
    parser.add_argument("--summary", type=str, default="  - Updated codebase and validated system integrity.", help="Markdown bulleted summary of changes")
    parser.add_argument("--version", type=str, default="v4.6.1", help="Version tag (e.g. v4.6.1)")
    parser.add_argument("--no-commit", action="store_true", help="Only generate docs without git commit")

    args = parser.parse_args()

    print("================================================================================")
    print("  PROJECT TREE & PROMPT-LEVEL GIT CHECKPOINT")
    print("================================================================================")

    # 1. First pass update changelog (with pending hash)
    update_changelog(args.prompt, args.summary, version=args.version, commit_hash="Pending")

    if args.no_commit:
        print("[INFO] --no-commit specified. Skipping git staging and commit.")
        return

    # 2. Stage all files
    print("[INFO] Staging all changes via 'git add -A'...")
    code, _, err = run_cmd(["git", "add", "-A"])
    if code != 0:
        print(f"[ERROR] git add failed: {err}")
        sys.exit(1)

    # 3. Commit
    commit_title = f"feat({args.version}): {args.prompt[:72]}"
    commit_body = f"{args.summary}\n\nAutomated prompt checkpoint by scripts/git_checkpoint.py."
    print(f"[INFO] Creating Git commit: {commit_title}")
    
    code, out, err = run_cmd(["git", "commit", "-m", commit_title, "-m", commit_body])
    if code != 0:
        if "nothing to commit" in out or "nothing to commit" in err:
            print("[INFO] Working tree clean. Nothing new to commit.")
            code, hash_out, _ = run_cmd(["git", "rev-parse", "--short", "HEAD"])
            update_changelog(args.prompt, args.summary, version=args.version, commit_hash=hash_out)
            return
        else:
            print(f"[ERROR] git commit failed: {err or out}")
            sys.exit(1)

    # 4. Get commit hash
    code, hash_out, _ = run_cmd(["git", "rev-parse", "--short", "HEAD"])
    commit_hash = hash_out.strip()
    print(f"[SUCCESS] Git commit created: {commit_hash}")

    # 5. Re-update changelog with the exact commit hash and amend commit so the document has the hash
    update_changelog(args.prompt, args.summary, version=args.version, commit_hash=commit_hash)
    run_cmd(["git", "add", str(CHANGELOG_PATH)])
    run_cmd(["git", "commit", "--amend", "--no-edit"])
    
    code, final_hash, _ = run_cmd(["git", "rev-parse", "--short", "HEAD"])
    print(f"[SUCCESS] Final Checkpoint Commit: {final_hash.strip()}")
    print("================================================================================")


if __name__ == "__main__":
    main()
