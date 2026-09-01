#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 1 - Download Gurbani Data
Fetches all 1430 Angs from the GurbaniNow API and saves locally.
Run: python scripts/01_download_data.py
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import requests
import json
import os
import time

RAW_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
os.makedirs(RAW_DIR, exist_ok=True)

# SikhiToTheMax API base
API_BASE = "https://api.gurbaninow.com/v2/ang"

def download_ang(ang_number):
    """Download a single Ang (page) from the API."""
    url = f"{API_BASE}/{ang_number}"
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            return resp.json()
        else:
            print(f"  [WARN] Ang {ang_number}: HTTP {resp.status_code}")
            return None
    except Exception as e:
        print(f"  [ERR] Ang {ang_number}: {e}")
        return None

def main():
    print()
    print("=" * 55)
    print("   [>>]  Gurbani Data Downloader - All 1430 Angs")
    print("=" * 55)
    print()

    total_angs = 1430
    success = 0
    failed = []

    for ang in range(1, total_angs + 1):
        out_file = os.path.join(RAW_DIR, f"ang_{ang:04d}.json")

        # Skip already downloaded
        if os.path.exists(out_file):
            success += 1
            if ang % 100 == 0:
                print(f"  [OK] Ang {ang} already exists - skipping")
            continue

        data = download_ang(ang)
        if data:
            with open(out_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            success += 1
            if ang % 50 == 0 or ang <= 5:
                print(f"  [OK] Downloaded Ang {ang}/{total_angs}")
        else:
            failed.append(ang)

        # Polite delay to not hammer the API
        time.sleep(0.3)

    print()
    print(f"  [DONE] Complete: {success}/{total_angs} Angs downloaded")
    if failed:
        print(f"  [WARN] Failed Angs: {failed}")
    print(f"  [DIR]  Saved to: {os.path.abspath(RAW_DIR)}")
    print()

if __name__ == '__main__':
    main()
