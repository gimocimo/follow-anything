#!/usr/bin/env python3
"""Convert SN-GSR-2025 clips to a YOLO detection dataset (Rung 3, 3c fine-tuning).

**Leave-one-game-out by construction:** pass only the games DISJOINT from the eval
game via --games, so the fine-tuned detector never sees a frame from its evaluation
game. This is the same anti-leakage invariant (PROJECT_PLAN §3) enforced everywhere else.
    dev detector   (eval = train game 4):  --source-split train --games 6,9
    final detector (eval = valid game 2):  --source-split train --games 4,6,9   # valid/game2 unseen

Classes: 0=person (GSR player/GK/referee = cats 1,2,3), 1=ball (cat 4). GSR cat 7
"other" (~0.5%, ambiguous) and non-object cats (pitch/camera) are excluded from
detector training — those objects are still counted in the tracking GT, so dev/final
eval stays apples-to-apples with the COCO baseline (which also doesn't special-case them).

    python scripts/09_gsr_to_yolo.py --source-split train --games 6,9 --out data/yolo_gsr_dev
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pitchvision.data.gsr import _bbox_xywh, _frame_index, index_gsr_sequences

PERSON_CATS = {1, 2, 3}          # player, goalkeeper, referee
BALL_CAT = 4
CLASS_OF = {**{c: 0 for c in PERSON_CATS}, BALL_CAT: 1}
NAMES = {0: "person", 1: "ball"}


def build(labels_json):
    """Parse a Labels-GameState.json -> {frame: {"dim": (W,H), "file": name, "boxes": [(cls,x,y,w,h)]}}."""
    data = json.loads(Path(labels_json).read_text())
    frames, frame_of = {}, {}
    for img in data.get("images", []):
        fr = _frame_index(img)
        if fr is None:
            continue
        frames[fr] = {"dim": (img.get("width"), img.get("height")),
                      "file": img.get("file_name") or f"{fr:06d}.jpg", "boxes": []}
        iid = img.get("image_id") or img.get("id")
        if iid is not None:
            frame_of[iid] = fr
    for ann in data.get("annotations", []):
        cls = CLASS_OF.get(ann.get("category_id"))
        if cls is None:
            continue
        box = _bbox_xywh(ann)
        fr = frame_of.get(ann.get("image_id"))
        if box is not None and fr in frames:
            frames[fr]["boxes"].append((cls, *box))
    return frames


def yolo_lines(dim, boxes):
    """(cls, x,y,w,h) top-left pixels -> normalized 'cls cx cy w h', clipped to [0,1]."""
    W, H = dim
    out = []
    if not W or not H:
        return out
    for cls, x, y, w, h in boxes:
        if w <= 0 or h <= 0:
            continue
        cx, cy = (x + w / 2) / W, (y + h / 2) / H
        nw, nh = w / W, h / H
        cx, cy = min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0)
        nw, nh = min(max(nw, 0.0), 1.0), min(max(nh, 0.0), 1.0)
        out.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data/soccernet-gsr")
    ap.add_argument("--source-split", default="train", choices=["train", "valid"])
    ap.add_argument("--games", required=True,
                    help="comma-sep game_ids to INCLUDE — MUST exclude the eval game (leave-one-game-out)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=5,
                    help="keep every Nth frame (adjacent frames are near-duplicates; 5 => ~150/clip)")
    ap.add_argument("--val-every", type=int, default=10, help="hold out every Nth clip as training val")
    ap.add_argument("--copy", action="store_true", help="copy images instead of symlinking")
    args = ap.parse_args()

    games = {g.strip() for g in args.games.split(",")}
    seqs = [s for s in index_gsr_sequences(args.data_dir, split=args.source_split, require_game_id=True)
            if s["game_id"] in games]
    if not seqs:
        sys.exit(f"no sequences in split={args.source_split} with game_id in {sorted(games)}")

    out = Path(args.out)
    counts = {"train": {"img": 0, "person": 0, "ball": 0},
              "val": {"img": 0, "person": 0, "ball": 0}}
    seen_games = set()
    for ci, s in enumerate(sorted(seqs, key=lambda x: x["name"])):
        seen_games.add(s["game_id"])
        subset = "val" if (ci % args.val_every == 0) else "train"
        (out / "images" / subset).mkdir(parents=True, exist_ok=True)
        (out / "labels" / subset).mkdir(parents=True, exist_ok=True)
        img_dir = Path(s["path"]) / "img1"
        for fr, meta in sorted(build(s["labels"]).items()):
            if fr % args.stride != 0:
                continue
            src = img_dir / Path(meta["file"]).name
            if not src.exists():
                continue
            stem = f"{s['name']}_{fr:06d}"
            dst_img = out / "images" / subset / f"{stem}.jpg"
            if args.copy:
                shutil.copy(src, dst_img)
            else:
                if dst_img.is_symlink() or dst_img.exists():
                    dst_img.unlink()
                dst_img.symlink_to(src.resolve())
            lines = yolo_lines(meta["dim"], meta["boxes"])
            (out / "labels" / subset / f"{stem}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""))
            counts[subset]["img"] += 1
            for ln in lines:
                counts[subset]["ball" if ln.startswith("1 ") else "person"] += 1

    out.mkdir(parents=True, exist_ok=True)
    data_yaml = out / "data.yaml"
    data_yaml.write_text(yaml.safe_dump(
        {"path": str(out.resolve()), "train": "images/train", "val": "images/val", "names": NAMES},
        sort_keys=False))
    print(f"games included: {sorted(seen_games)}  ({len(seqs)} clips)  stride={args.stride}")
    print(f"  train: {counts['train']['img']:6d} imgs | {counts['train']['person']} person | {counts['train']['ball']} ball")
    print(f"  val:   {counts['val']['img']:6d} imgs | {counts['val']['person']} person | {counts['val']['ball']} ball")
    print(f"wrote {data_yaml}")
    if counts["val"]["img"] == 0:
        print("WARNING: empty val set — lower --val-every or add clips")


if __name__ == "__main__":
    main()
