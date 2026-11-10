#!/usr/bin/env python3
import requests
import lzma
import struct
from datetime import datetime, timezone

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}

def test_fetch(symbol, dt):
    month_zero = f"{dt.month - 1:02d}"
    day_str = f"{dt.day:02d}"
    hour_str = f"{dt.hour:02d}"
    url = f"https://datafeed.dukascopy.com/datafeed/{symbol}/{dt.year}/{month_zero}/{day_str}/{hour_str}h_ticks.bi5"
    print(f"Testing URL: {url}")
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        print(f"Status Code: {r.status_code}, Length: {len(r.content)}")
        if r.status_code == 200 and len(r.content) > 0:
            dec = lzma.decompress(r.content)
            ticks = len(dec) // 20
            print(f"Successfully decompressed: {len(dec)} bytes -> {ticks} ticks")
        elif r.status_code == 404:
            print("404 Not Found (no data or market closed)")
        else:
            print(f"Server response: {r.status_code} {r.reason}")
    except Exception as e:
        print(f"Error: {e}")

# Test Wednesday during liquid hours (should always exist)
dt_test = datetime(2023, 5, 17, 10, 0, tzinfo=timezone.utc)
print("=== AUDNZD ===")
test_fetch("AUDNZD", dt_test)

print("\n=== EURGBP ===")
test_fetch("EURGBP", dt_test)

print("\n=== USATECHIDXUSD ===")
test_fetch("USATECHIDXUSD", dt_test)
