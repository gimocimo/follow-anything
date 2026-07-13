#!/usr/bin/env python3
"""Download SoccerNet-Tracking data.

Requires SoccerNet NDA access; pass the password they provide (the commonly
distributed one is the default). The data is large, so we default to the
`train` split only — it ships public ground truth, which is all the baseline
needs. Add `test`/`challenge` later for the full benchmark.

    python scripts/01_download_soccernet.py --data-dir data/soccernet --splits train
"""
import argparse
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data-dir", default="data/soccernet")
    ap.add_argument("--password", default="s0cc3rn3t", help="SoccerNet NDA password")
    ap.add_argument(
        "--splits", nargs="+", default=["train"],
        choices=["train", "valid", "test", "challenge"],
    )
    args = ap.parse_args()

    from SoccerNet.Downloader import SoccerNetDownloader

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    dl = SoccerNetDownloader(LocalDirectory=str(data_dir))
    dl.password = args.password
    print(f"Downloading SoccerNet-Tracking splits={args.splits} into {data_dir} ...")
    dl.downloadDataTask(task="tracking", split=args.splits, password=args.password)

    seqinfos = sorted(data_dir.glob("**/seqinfo.ini"))
    print(f"\nDone. Found {len(seqinfos)} sequence(s).")
    if seqinfos:
        sample = seqinfos[0].parent
        print(f"Sample sequence: {sample}")
        for ini in ("seqinfo.ini", "gameinfo.ini"):
            p = sample / ini
            if p.exists():
                print(f"\n--- {ini} (first 800 chars) ---")
                print(p.read_text()[:800])
        print("\nNext: python scripts/03_make_splits.py --data-dir", args.data_dir)


if __name__ == "__main__":
    main()
