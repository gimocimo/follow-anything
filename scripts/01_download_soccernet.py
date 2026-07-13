#!/usr/bin/env python3
"""Download SoccerNet-Tracking (2023) data.

Requires SoccerNet NDA access; pass the password they provide (the commonly
distributed one is the default). The data downloads as zips from SoccerNet's
server and this script extracts them.

The data is large, so we default to the `train` split only — it ships public
ground truth, which is all the baseline needs. Splits:
    train         images + public GT   (what the baseline uses)
    test          images only
    test_labels   private test GT       (needed to evaluate on the official test set)
    challenge     images only

    python scripts/01_download_soccernet.py --data-dir data/soccernet --splits train
"""
import argparse
import zipfile
from pathlib import Path

TASK = "tracking-2023"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data-dir", default="data/soccernet")
    ap.add_argument("--password", default="s0cc3rn3t", help="SoccerNet NDA password")
    ap.add_argument(
        "--splits", nargs="+", default=["train"],
        choices=["train", "test", "test_labels", "challenge"],
    )
    args = ap.parse_args()

    from SoccerNet.Downloader import SoccerNetDownloader

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    dl = SoccerNetDownloader(LocalDirectory=str(data_dir))
    dl.password = args.password
    print(f"Downloading SoccerNet {TASK} splits={args.splits} into {data_dir} ...")
    print("(if you see 'not uploaded on the server yet' that means an HTTP error "
          "— usually a wrong password.)")
    dl.downloadDataTask(task=TASK, split=args.splits, password=args.password)

    # Extract any zips that aren't already unpacked.
    task_dir = data_dir / TASK
    for z in sorted(task_dir.glob("*.zip")):
        target = task_dir / z.stem
        if target.is_dir():
            print(f"  already extracted: {z.name}")
            continue
        print(f"  extracting {z.name} ...")
        with zipfile.ZipFile(z) as zf:
            zf.extractall(task_dir)

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
        print(f"\nNext: python scripts/03_make_splits.py --data-dir {args.data_dir}")
    else:
        print("No sequences found after extraction — check the download messages above.")


if __name__ == "__main__":
    main()
