from pathlib import Path
import re

root = Path("scratch")
labs = sorted(
    root.glob("alpha_discovery_lab_v*.py"),
    key=lambda p: int(re.search(r"\d+", p.stem).group()) if re.search(r"\d+", p.stem) else 0
)

for lab in labs:
    content = lab.read_text(encoding="utf-8", errors="ignore")
    # find docstring
    match = re.search(r'"""(.*?)"""', content, re.DOTALL)
    if match:
        lines = [line.strip() for line in match.group(1).strip().splitlines() if line.strip() and not line.strip().startswith("=")]
        header = " | ".join(lines[:3])
    else:
        header = "No header"
    print(f"{lab.name:<28} : {header[:90]}")

