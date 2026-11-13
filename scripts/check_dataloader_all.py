import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from tick_engine.core.dataloader import get_discovery_weeks, get_holdout_test_weeks

symbols = ['AUDCAD', 'AUDNZD', 'EURGBP', 'EURNZD', 'EURUSD', 'GBPCHF', 'GBPJPY', 'NZDCAD', 'USDCAD', 'USDJPY']
for s in symbols:
    d = get_discovery_weeks(s)
    h = get_holdout_test_weeks(s)
    print(f"{s:>7}: Discovery {len(d):>3}w | Holdout {len(h):>2}w")
