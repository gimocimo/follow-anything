#!/usr/bin/env python3
"""Build a MATCH-DISJOINT train/val/test split over SoccerNet-Tracking clips.

Clips are grouped by their source match (from gameinfo.ini) so no match's clips
straddle a split boundary — the same rigor we adopted after the microrobot
leakage. If no match id is present in the metadata, it falls back to per-clip
grouping (still leak-safe, since whole clips are the unit) and says so loudly.

    python scripts/03_make_splits.py --data-dir data/soccernet
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pitchvision.data.soccernet import index_sequences
from pitchvision.data.splits import group_disjoint_split, summarize_splits


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data-dir", default="data/soccernet")
    ap.add_argument("--split", default="train", help="SoccerNet split to carve up (train has GT)")
    ap.add_argument("--ratios", type=float, nargs=3, default=[0.7, 0.15, 0.15],
                    metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="outputs/soccernet_splits.json")
    args = ap.parse_args()

    seqs = index_sequences(args.data_dir, split=args.split)
    if not seqs:
        sys.exit(f"No sequences found under {args.data_dir} (split={args.split}). "
                 f"Run scripts/01_download_soccernet.py first.")

    n_groups = len({s.match_id for s in seqs})
    fallback = n_groups == len(seqs)
    n_gt = sum(s.has_gt for s in seqs)
    print(f"{len(seqs)} clips | {n_groups} group(s) | "
          f"{'PER-CLIP fallback (no match id in gameinfo.ini)' if fallback else 'grouped by source match'}")
    print(f"clips with ground truth: {n_gt}/{len(seqs)}")

    # surface gameinfo.ini so we can refine match-grouping if it fell back
    gi = seqs[0].path / "gameinfo.ini"
    if gi.exists():
        print(f"\nsample gameinfo.ini ({seqs[0].name}), first 500 chars:\n{gi.read_text()[:500]}\n")

    splits = group_disjoint_split(
        seqs, group_key=lambda s: s.match_id, ratios=tuple(args.ratios), seed=args.seed
    )
    print("split sizes ->", summarize_splits(splits))

    payload = {
        "meta": {
            "data_dir": args.data_dir, "source_split": args.split,
            "ratios": args.ratios, "seed": args.seed,
            "grouping": "per_clip_fallback" if fallback else "by_match",
        },
        **{
            name: [
                {"name": s.name, "match_id": s.match_id, "length": s.length,
                 "path": str(s.path), "has_gt": s.has_gt}
                for s in items
            ]
            for name, items in splits.items()
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out}")
    print("Next: python scripts/04_baseline_eval.py --split test --max-seqs 5")


if __name__ == "__main__":
    main()
