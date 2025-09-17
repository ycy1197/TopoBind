#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download PDB files (RCSB) in parallel based on PDB IDs found in alldata.csv.

Input : alldata.csv (must contain a column named 'pdb')
Output: PDB/<PDB_ID>.pdb files

Usage:
    python download_pdbs.py
"""

import os
import time
import requests
import pandas as pd
import concurrent.futures
from tqdm import tqdm

CSV_FILE = "alldata.csv"
OUT_DIR = "PDB"

def get_pdb_ids(csv_file: str):
    """Extract unique PDB IDs from the CSV (column 'pdb')."""
    df = pd.read_csv(csv_file)
    pdb_ids = df["pdb"].unique().tolist()

    # Clean potential chain suffixes (e.g., '1ABC_A' -> '1ABC')
    clean = []
    for pdb_id in pdb_ids:
        core_id = str(pdb_id).split("_")[0].upper()
        if len(core_id) >= 4:
            clean.append(core_id)
    return list(set(clean))

def create_download_dir():
    """Create output directory if it does not exist."""
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
        print("Created PDB directory")
    else:
        print("PDB directory already exists")

def download_pdb(pdb_id: str, retries: int = 3):
    """
    Download a single PDB file with simple retry logic.
    Returns: (pdb_id, success: bool, message: str)
    """
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    output_file = os.path.join(OUT_DIR, f"{pdb_id}.pdb")

    if os.path.exists(output_file):
        return pdb_id, True, "exists"

    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                with open(output_file, "wb") as f:
                    f.write(resp.content)
                return pdb_id, True, "downloaded"
            elif resp.status_code == 404:
                return pdb_id, False, "404 not found"
            else:
                if attempt < retries - 1:
                    time.sleep(1)
                    continue
                return pdb_id, False, f"http {resp.status_code}"
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(1)
                continue
            return pdb_id, False, f"exception: {str(e)}"

def download_pdbs_parallel(pdb_ids, max_workers: int = 20):
    """Parallel downloader with progress bar and failure report."""
    success_cnt = skip_cnt = fail_cnt = 0
    failed = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(download_pdb, pid): pid for pid in pdb_ids}
        with tqdm(total=len(pdb_ids), desc="Downloading PDB files") as pbar:
            for fut in concurrent.futures.as_completed(futures):
                pid, ok, msg = fut.result()
                if ok:
                    if msg == "exists":
                        skip_cnt += 1
                    else:
                        success_cnt += 1
                else:
                    fail_cnt += 1
                    failed.append((pid, msg))
                pbar.update(1)

    print(f"\nDone. success: {success_cnt}, skip: {skip_cnt}, fail: {fail_cnt}")
    if failed:
        print("\nFailed PDB IDs:")
        for pid, reason in failed:
            print(f"  {pid}: {reason}")
        with open("failed_pdb_downloads.txt", "w") as f:
            for pid, reason in failed:
                f.write(f"{pid}: {reason}\n")
        print("Saved failures to 'failed_pdb_downloads.txt'")

    return success_cnt, skip_cnt, fail_cnt

def main():
    pdb_ids = get_pdb_ids(CSV_FILE)
    print(f"Unique PDB IDs extracted: {len(pdb_ids)}")
    create_download_dir()

    max_workers = 20
    print(f"Start downloading with {max_workers} threads ...")
    t0 = time.time()
    download_pdbs_parallel(pdb_ids, max_workers=max_workers)
    print(f"Elapsed: {time.time() - t0:.2f}s")

if __name__ == "__main__":
    main()
