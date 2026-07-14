#!/usr/bin/env python3
"""Rung 2 demo: prompt one player and follow them through a clip with SAM 2.

    python scripts/07_promptable_demo.py --clip SNGS-060 --frames 90
    python scripts/07_promptable_demo.py --clip SNGS-060 --box 900 400 960 560

If no --box/--point is given, the most prominent player from GT on the first
frame is used as the "click". SAM 2 propagates the mask through the segment and
we render a spotlight video (dim background, highlight the tracked player).
"""
import argparse
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pitchvision.data.gsr import gsr_to_mot_rows
from pitchvision.promptable.sam2_track import follow_object


def pick_player_box(labels_json, frame=1):
    """Largest GT box on `frame` = most prominent player (not the ball)."""
    rows = [r for r in gsr_to_mot_rows(labels_json) if r[0] == frame]
    if not rows:
        return None
    fr, tid, x, y, w, h = max(rows, key=lambda r: r[4] * r[5])
    return (x, y, x + w, y + h), tid


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data/soccernet-gsr")
    ap.add_argument("--clip", default="SNGS-060")
    ap.add_argument("--frames", type=int, default=90, help="segment length")
    ap.add_argument("--box", type=float, nargs=4, default=None, metavar=("X1", "Y1", "X2", "Y2"))
    ap.add_argument("--point", type=float, nargs=2, default=None, metavar=("X", "Y"))
    ap.add_argument("--model", default="facebook/sam2.1-hiera-small")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out-dir", default="outputs/promptable")
    args = ap.parse_args()

    clip_dir = next((c for c in Path(args.data_dir).glob(f"**/{args.clip}") if (c / "img1").is_dir()), None)
    if clip_dir is None:
        sys.exit(f"clip {args.clip} not found under {args.data_dir}")
    img1, labels = clip_dir / "img1", clip_dir / "Labels-GameState.json"

    box, point = args.box, args.point
    if box is None and point is None:
        picked = pick_player_box(labels, frame=1)
        if picked is None:
            sys.exit("no GT on frame 1 to auto-pick a player; pass --box or --point")
        box, tid = picked
        print(f"auto-picked player track {tid}, box={tuple(round(v, 1) for v in box)}")

    frames = sorted(img1.glob("*.jpg"), key=lambda p: int(p.stem))[: args.frames]
    tmp = Path(tempfile.mkdtemp(prefix="pv_sam2_"))
    for f in frames:
        os.symlink(f.resolve(), tmp / f.name)
    print(f"running SAM 2 ({args.model}) on {len(frames)} frames of {args.clip} ...")

    try:
        frame_files, masks = follow_object(tmp, box=box, point=point, model_id=args.model, device=args.device)
    except Exception as e:
        print(f"[{args.device}] failed ({type(e).__name__}: {e}) — retrying on CPU ...")
        frame_files, masks = follow_object(tmp, box=box, point=point, model_id=args.model, device="cpu")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_video = out_dir / f"{args.clip}_follow.mp4"
    writer, covered, w, h = None, 0, 0, 0
    for pos, fp in enumerate(frame_files):
        img = cv2.imread(str(fp))
        if writer is None:
            h, w = img.shape[:2]
            writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (w, h))
        m = masks.get(pos)
        if m is not None and m.any():
            covered += 1
            dark = (img * 0.45).astype(np.uint8)
            dark[m] = img[m]
            green = np.zeros_like(img)
            green[m] = (0, 255, 0)
            img = cv2.addWeighted(dark, 1.0, green, 0.35, 0)
            ys, xs = np.where(m)
            cv2.rectangle(img, (int(xs.min()), int(ys.min())), (int(xs.max()), int(ys.max())), (0, 255, 0), 3)
            cv2.putText(img, "tracked player", (int(xs.min()), max(0, int(ys.min()) - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        writer.write(img)
        if pos == len(frame_files) // 2 and w:
            cv2.imwrite(str(out_dir / f"{args.clip}_preview.png"), cv2.resize(img, (w // 2, h // 2)))
    if writer:
        writer.release()

    print(f"done: tracked player present in {covered}/{len(frame_files)} frames")
    print(f"video   -> {out_video}")
    print(f"preview -> {out_dir / (args.clip + '_preview.png')}")


if __name__ == "__main__":
    main()
