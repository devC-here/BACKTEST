import json
from pathlib import Path
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent

# Let's inspect scripts to see what other EURGBP scripts exist
p = ROOT_DIR / "scripts"
eurgbp_scripts = list(p.glob("*eurgbp*.py"))
print("EURGBP scripts in scripts/:")
for s in eurgbp_scripts:
    print(f"  {s.name}")

