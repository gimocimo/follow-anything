#!/usr/bin/env python3
"""Match-disjoint train/val/test split over SN-GSR-2025 clips.

Clips are grouped by source game (`info.game_id` in each Labels-GameState.json)
so clips from the same match never straddle a split boundary — genuine
match-level grouping (not a per-clip fallback), exactly the rigor we insisted on
after the microrobot leakage.

    python scripts/05_gsr_make_splits.py --data-dir data/soccernet-gsr
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pitchvision.data.gsr import index_gsr_sequences
from pitchvision.data.splits import group_disjoint_split, summarize_splits


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data-dir", default="data/soccernet-gsr")
    ap.add_argument("--split", default="train", help="GSR split to carve up (train has labels)")
    ap.add_argument("--ratios", type=float, nargs=3, default=[0.7, 0.15, 0.15],
                    metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="outputs/gsr_splits.json")
    args = ap.parse_args()

    seqs = index_gsr_sequences(args.data_dir, split=args.split)
    if not seqs:
        sys.exit(f"No GSR sequences under {args.data_dir} (split={args.split}). "
                 f"Extract train.zip first.")

    n_games = len({s["match_id"] for s in seqs})
    fallback = n_games == len(seqs)
    print(f"{len(seqs)} clips | {n_games} game(s) | "
          f"{'PER-CLIP fallback (no game_id!)' if fallback else 'grouped by game_id'}")
    print(f"total frames: {sum(s['length'] for s in seqs)}")

    splits = group_disjoint_split(
        seqs, group_key=lambda s: s["match_id"], ratios=tuple(args.ratios), seed=args.seed
    )
    print("split sizes ->", summarize_splits(splits))
    for name, items in splits.items():
        print(f"  {name}: {len(items)} clips from games {sorted({s['match_id'] for s in items})}")

    payload = {
        "meta": {"data_dir": args.data_dir, "source_split": args.split,
                 "ratios": args.ratios, "seed": args.seed,
                 "grouping": "per_clip_fallback" if fallback else "by_game_id"},
        **{name: [{"name": s["name"], "match_id": s["match_id"], "length": s["length"],
                   "path": s["path"], "labels": s["labels"]} for s in items]
           for name, items in splits.items()},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out}")
    print("Next: python scripts/06_gsr_baseline_eval.py --split test --max-seqs 3")


if __name__ == "__main__":
    main()
