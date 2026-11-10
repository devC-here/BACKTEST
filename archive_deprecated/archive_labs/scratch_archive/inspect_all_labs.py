import glob
import os
import re

labs = sorted(glob.glob('scratch/alpha_discovery_lab_*.py') + glob.glob('scratch/alpha_optimization_lab_*.py'))
print(f'Total lab scripts found: {len(labs)}')

for f in labs:
    base = os.path.basename(f)
    with open(f, 'r', encoding='utf-8', errors='ignore') as fp:
        text = fp.read()
        classes = re.findall(r'class\s+([A-Za-z0-9_]+)', text)
        strat_classes = [c for c in classes if not c.startswith('Base') and not c.startswith('Fast')]
        
        # docstring
        m = re.search(r'"""([\s\S]*?)"""', text)
        doc = m.group(1).strip() if m else ''
        lines = [l.strip() for l in doc.split('\n') if l.strip() and not l.strip().startswith('=')]
        first_line = lines[0] if lines else 'No description'
        print(f"{base:<45} | Classes: {len(strat_classes):<2} | {first_line[:65]}")

