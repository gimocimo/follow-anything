#!/usr/bin/env python3
"""Build a COMMITTED, reproducible, match-disjoint split over SN-GSR-2025 clips.

Policy: **leave-one-game-out**, deterministic and explicit — no ratios or seeds.
With games sorted by id, test = the first game, val = the second, train = the rest.
For the downloaded `train` games (ids 4, 6, 9) this yields the committed split
**train=game9, val=game6, test=game4**.

Roles (PROJECT_PLAN §4): game 4 (the `train`-source split's "test") is the
**development benchmark** — it has been inspected during development. The
**final-confirmation set** is the official `valid` split (held-out, game-disjoint;
re-scored across model iterations, so NOT a pristine one-shot — PROJECT_PLAN §7); build it with
`--source-split valid` once downloaded.

Fails closed: requires `info.game_id` on every clip (no silent fallback grouping),
and refuses to emit an empty required split (the old 70/15/15 ratio default
produced an empty test set).

    python scripts/05_gsr_make_splits.py --data-dir data/soccernet-gsr
    python scripts/05_gsr_make_splits.py --source-split valid --out outputs/gsr_splits_final.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pitchvision.data.gsr import index_gsr_sequences
from pitchvision.data.splits import leave_one_game_out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data/soccernet-gsr")
    ap.add_argument("--source-split", default="train", choices=["train", "valid", "test"],
                    help="which downloaded GSR split to partition (leave-one-game-out)")
    ap.add_argument("--out", default="outputs/gsr_splits.json")
    args = ap.parse_args()

    seqs = index_gsr_sequences(args.data_dir, split=args.source_split, require_game_id=True)
    if not seqs:
        sys.exit(f"No GSR sequences under {args.data_dir} (split={args.source_split}). Extract the data first.")

    # also require every clip to actually contain frames (not just be listed)
    empty_clips = [s["name"] for s in seqs if not s["length"]]
    if empty_clips:
        sys.exit(f"Refusing to split: {len(empty_clips)} clip(s) have zero frames: {empty_clips[:5]}")

    splits, assignment = leave_one_game_out(seqs, group_key=lambda s: s["game_id"])
    # leave_one_game_out already guarantees non-empty train/val/test (raises otherwise)

    print(f"{len(seqs)} clips | games {sorted({s['game_id'] for s in seqs})} | policy=leave-one-game-out")
    print(f"game -> split: {assignment}")
    for name in ("train", "val", "test"):
        games = sorted({s["game_id"] for s in splits[name]})
        print(f"  {name}: {len(splits[name])} clips from game(s) {games}")

    role = "development" if args.source_split == "train" else "final"
    payload = {
        "meta": {"data_dir": args.data_dir, "source_split": args.source_split,
                 "policy": "leave-one-game-out", "game_assignment": assignment,
                 "role": role, "grouping": "info.game_id (required)"},
        **{name: [{"name": s["name"], "game_id": s["game_id"], "match_id": s["match_id"],
                   "length": s["length"], "path": s["path"], "labels": s["labels"]}
                  for s in items]
           for name, items in splits.items()},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out}  (role={role})")


if __name__ == "__main__":
    main()
