import json
from pathlib import Path

manifest_file = Path("v 4.5/FLEET_MANIFEST.json")
with open(manifest_file, "r", encoding="utf-8") as f:
    data = json.load(f)

for item in data:
    if 41 <= item.get("slot", 0) <= 48:
        print(f"Slot {item.get('slot'):02d}: {item.get('symbol')} | {item.get('name')} | {item.get('filename')}")

