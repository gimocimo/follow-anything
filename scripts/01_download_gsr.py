#!/usr/bin/env python3
"""Download + extract a SoccerNet SN-GSR-2025 split from Hugging Face (gated).

Prereqs: a Hugging Face account with access to the gated dataset
(https://huggingface.co/datasets/SoccerNet/SN-GSR-2025), then `hf auth login`.

Data is per-split zips (train ~9.8GB, valid ~11.2GB, test ~8.9GB, challenge ~5.3GB).
`train` and `valid` ship labels; `train` is the development source and `valid` is
the untouched final-confirmation set (see docs/PROJECT_PLAN.md §4).

    python scripts/01_download_gsr.py --split train
    python scripts/01_download_gsr.py --split valid
"""
import argparse
import zipfile
from pathlib import Path

REPO = "SoccerNet/SN-GSR-2025"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data/soccernet-gsr")
    ap.add_argument("--split", default="train", choices=["train", "valid", "test", "challenge"])
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_name = f"{args.split}.zip"
    print(f"Downloading {REPO}::{zip_name} -> {data_dir}  (gated; requires `hf auth login`) ...")
    zpath = hf_hub_download(repo_id=REPO, filename=zip_name, repo_type="dataset", local_dir=str(data_dir))

    dest = data_dir / args.split
    if dest.is_dir() and any(dest.glob("**/Labels-GameState.json")):
        print(f"already extracted: {dest}")
    else:
        print(f"extracting {zip_name} -> {dest} ...")
        with zipfile.ZipFile(zpath) as zf:
            zf.extractall(dest)

    n = len(list(dest.glob("**/Labels-GameState.json")))
    print(f"done: {n} sequence(s) under {dest}")
    print(f"Next: python scripts/05_gsr_make_splits.py --source-split {args.split}")


if __name__ == "__main__":
    main()
