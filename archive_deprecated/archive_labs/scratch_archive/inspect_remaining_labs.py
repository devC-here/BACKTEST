import glob
import re

labs = [
    'scratch/alpha_discovery_lab_audnzd_cross_specialists.py',
    'scratch/alpha_discovery_lab_audnzd_expansion.py',
    'scratch/alpha_discovery_lab_audnzd_mean_reversion.py',
    'scratch/alpha_discovery_lab_audnzd_peer_fleet.py',
    'scratch/alpha_discovery_lab_audnzd_v1.py',
    'scratch/alpha_discovery_lab_audnzd_v2.py',
    'scratch/alpha_discovery_lab_audnzd_v3.py',
    'scratch/alpha_discovery_lab_eurgbp_expansion.py',
    'scratch/alpha_discovery_lab_v39.py',
    'scratch/alpha_discovery_lab_v40.py',
    'scratch/alpha_discovery_lab_v41.py',
    'scratch/alpha_discovery_lab_v42.py',
    'scratch/alpha_discovery_lab_v43.py',
    'scratch/alpha_discovery_lab_v44.py',
    'scratch/alpha_discovery_lab_v45.py',
    'scratch/alpha_optimization_lab_v38.py',
]

for f in labs:
    with open(f, 'r', encoding='utf-8') as fp:
        text = fp.read()
    # Check data/ calls
    calls = re.findall(r'data/[A-Z0-9_]+', text)
    print(f"{f:<50} | data calls: {set(calls)}")

