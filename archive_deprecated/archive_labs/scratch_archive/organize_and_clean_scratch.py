import os
import shutil
from pathlib import Path

ROOT = Path(".")
SCRATCH = ROOT / "scratch"
DISCOVERY_LABS = ROOT / "discovery_labs"
DISCOVERY_LABS.mkdir(parents=True, exist_ok=True)

ARCHIVE_SCRATCH = ROOT / "archive_labs" / "scratch_archive"
ARCHIVE_SCRATCH.mkdir(parents=True, exist_ok=True)

# 1. Identify discovery lab files
lab_files = []
validation_files = []
other_files = []

for f in os.listdir(SCRATCH):
    src = SCRATCH / f
    if not src.is_file():
        continue
    if f.startswith("alpha_discovery_lab_") or f.startswith("alpha_optimization_lab_"):
        lab_files.append(f)
    elif f.startswith("validate_lab_") or f.startswith("audit_lab_"):
        validation_files.append(f)
    else:
        other_files.append(f)

print(f"Moving {len(lab_files)} discovery lab files to discovery_labs/...")
for f in lab_files:
    shutil.move(str(SCRATCH / f), str(DISCOVERY_LABS / f))

print(f"Moving {len(validation_files)} validation files to discovery_labs/...")
for f in validation_files:
    shutil.move(str(SCRATCH / f), str(DISCOVERY_LABS / f))

print(f"Archiving {len(other_files)} other scratch files to archive_labs/scratch_archive/...")
for f in other_files:
    shutil.move(str(SCRATCH / f), str(ARCHIVE_SCRATCH / f))

print("Organization complete.")
print(f"Items in discovery_labs/: {len(os.listdir(DISCOVERY_LABS))}")
print(f"Items in scratch/: {len(os.listdir(SCRATCH))}")
