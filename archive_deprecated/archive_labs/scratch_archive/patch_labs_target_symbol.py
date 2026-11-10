import glob
import re

labs = sorted(glob.glob('scratch/alpha_discovery_lab_*.py') + glob.glob('scratch/alpha_optimization_lab_*.py'))
print(f"Total scripts: {len(labs)}")

count_patched = 0
for f in labs:
    with open(f, 'r', encoding='utf-8') as fp:
        content = fp.read()
    
    # Check if it has assets = [...]
    pattern = r'(assets\s*=\s*\[[\s\S]*?\])'
    m = re.search(pattern, content)
    if m:
        orig = m.group(1)
        # Check if already patched
        if 'DISCOVERY_TARGET_SYMBOL' not in orig:
            repl = f'assets = [os.environ["DISCOVERY_TARGET_SYMBOL"]] if "DISCOVERY_TARGET_SYMBOL" in os.environ else {orig.split("=", 1)[1].strip()}'
            new_content = content.replace(orig, repl, 1)
            with open(f, 'w', encoding='utf-8') as fp:
                fp.write(new_content)
            count_patched += 1
            print(f"Patched assets in: {f}")

print(f"Total patched scripts: {count_patched}")

