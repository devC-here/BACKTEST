#!/usr/bin/env python3
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import lzma
from datetime import datetime, timezone

def make_resilient_session():
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=1.0,
        status_forcelist=[500, 502, 503, 504, 429],
        raise_on_status=False
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=10, pool_maxsize=10)
    session.mount("https://", adapter)
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    })
    return session

session = make_resilient_session()
dt = datetime(2023, 5, 17, 10, 0, tzinfo=timezone.utc)
month_zero = f"{dt.month - 1:02d}"
day_str = f"{dt.day:02d}"
hour_str = f"{dt.hour:02d}"

for sym in ["AUDNZD", "EURGBP", "USATECHIDXUSD"]:
    url = f"https://datafeed.dukascopy.com/datafeed/{sym}/{dt.year}/{month_zero}/{day_str}/{hour_str}h_ticks.bi5"
    print(f"\nFetching {sym} with resilient session: {url}")
    t0 = time.time()
    resp = session.get(url, timeout=20)
    dt_req = time.time() - t0
    print(f"Status: {resp.status_code} | Bytes: {len(resp.content)} | Time: {dt_req:.2f}s")
    if resp.status_code == 200 and len(resp.content) > 0:
        dec = lzma.decompress(resp.content)
        ticks = len(dec) // 20
        print(f"SUCCESS: Decompressed {ticks:,} ticks for {sym}!")
