#!/usr/bin/env python3
"""
================================================================================
  AUTOMATED SINGLE-FILE TURN & PROMPT RECORDER
================================================================================
Appends a new prompt, response, and changes entry to the root-level
PROMPT_HISTORY_AND_CHANGELOG.md file.

Usage:
  python scripts/record_turn.py --prompt "..." --response "..." --changes "  - File created..."
================================================================================
"""

import re
import argparse
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
LEDGER_PATH = ROOT_DIR / "PROMPT_HISTORY_AND_CHANGELOG.md"


def get_next_turn_number(content: str) -> int:
    turns = re.findall(r"## Turn (\d+):", content)
    if not turns:
        return 1
    return max(int(t) for t in turns) + 1


def record_turn(prompt: str, response: str, changes: str, turn_title: str = None):
    if not LEDGER_PATH.exists():
        print(f"[ERROR] Ledger file not found at {LEDGER_PATH}")
        return

    content = LEDGER_PATH.read_text(encoding="utf-8", errors="replace")
    next_turn = get_next_turn_number(content)
    
    if not turn_title:
        # Generate a brief title from prompt
        clean_p = prompt.strip().replace("\n", " ")
        turn_title = clean_p[:60] + ("..." if len(clean_p) > 60 else "")

    utc_now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    local_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S IST")

    entry = f"""
## Turn {next_turn}: {turn_title}
- **Timestamp**: `{utc_now}` ({local_now})
- **User Prompt**:
  > *"{prompt.strip()}"*

- **Assistant Response**:
  {response.strip()}

- **Changes Done Respectively**:
{changes.strip()}

---
"""

    # Update Quick Navigation Index
    nav_line = f"- [Turn {next_turn}: {turn_title}](#turn-{next_turn}-{re.sub(r'[^a-z0-9]+', '-', turn_title.lower()).strip('-')})\n"
    index_marker = "## 📑 Quick Navigation Index\n\n"
    
    if index_marker in content:
        parts = content.split(index_marker)
        content = parts[0] + index_marker + nav_line + parts[1]

    # Append entry at end of file
    content = content.rstrip() + "\n" + entry
    LEDGER_PATH.write_text(content, encoding="utf-8")
    print(f"[SUCCESS] Appended Turn {next_turn} to {LEDGER_PATH}")


def main():
    parser = argparse.ArgumentParser(description="Record a prompt turn in the root ledger")
    parser.add_argument("--prompt", required=True, help="User prompt text")
    parser.add_argument("--response", required=True, help="Assistant response summary")
    parser.add_argument("--changes", required=True, help="Markdown list of changes done")
    parser.add_argument("--title", default=None, help="Optional title for the turn")

    args = parser.parse_args()
    record_turn(args.prompt, args.response, args.changes, args.title)


if __name__ == "__main__":
    main()
