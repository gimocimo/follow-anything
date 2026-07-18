#!/usr/bin/env python3
"""Render a showcase demo video: every player tracked simultaneously with persistent IDs.

Runs the SAME production path as the evaluation (one streaming `model.track` call, so IDs
persist), then draws a clean overlay: per-ID colour, ID chip, motion trail, and a header bar.

Defaults to a **held-out** clip (dev game 4 — never trained on), so the demo shows honest
generalisation rather than memorised training footage.

    python scripts/14_make_demo.py --clip-index 0 --max-frames 300
    python scripts/14_make_demo.py --clip data/soccernet-gsr/train/SNGS-060 --out docs/demo_players.mp4
"""
import argparse
import json
import sys
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pitchvision.config import get_device
from pitchvision.pipeline.run_video import _resolve_tracker


def color_for(tid: int):
    """Stable, well-spread colour per track id (BGR)."""
    h = int((tid * 47) % 180)
    px = cv2.cvtColor(np.uint8([[[h, 200, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
    return int(px[0]), int(px[1]), int(px[2])


def draw_header(img, text, sub=None):
    h, w = img.shape[:2]
    bar = img.copy()
    cv2.rectangle(bar, (0, 0), (w, 64 if sub else 44), (18, 18, 18), -1)
    cv2.addWeighted(bar, 0.72, img, 0.28, 0, img)
    cv2.putText(img, text, (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (255, 255, 255), 2, cv2.LINE_AA)
    if sub:
        cv2.putText(img, sub, (18, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (185, 185, 185), 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", default=None, help="path to a GSR sequence dir (contains img1/)")
    ap.add_argument("--splits-file", default="outputs/gsr_splits.json")
    ap.add_argument("--split", default="test", help="which split to pull --clip-index from (test = held-out)")
    ap.add_argument("--clip-index", type=int, default=0)
    ap.add_argument("--weights", default="models/yolo11m_gsr_ft.pt")
    ap.add_argument("--tracker", default="configs/trackers/botsort_newtrk040.yaml")
    ap.add_argument("--classes", default="0,1")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-frames", type=int, default=300, help="~12 s at 25 fps")
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--trail", type=int, default=25, help="motion-trail length in frames (0 = off)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    clip = Path(args.clip) if args.clip else None
    if clip is None:
        data = json.loads(Path(args.splits_file).read_text())
        s = data[args.split][args.clip_index]
        clip = Path(s["path"])
    img_dir = clip / "img1"
    out_path = Path(args.out) if args.out else Path("outputs/demos") / f"{clip.name}_tracked.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO
    model = YOLO(args.weights)
    results = model.track(
        source=str(img_dir), classes=[int(c) for c in args.classes.split(",")],
        conf=args.conf, imgsz=args.imgsz, tracker=_resolve_tracker(args.tracker),
        persist=True, stream=True, device=get_device(args.device), verbose=False,
    )

    writer, trails, n = None, defaultdict(lambda: deque(maxlen=args.trail)), 0
    for r in results:
        if n >= args.max_frames:
            break
        frame = cv2.imread(r.path)
        if frame is None:
            continue
        b = r.boxes
        live = 0
        if b is not None and b.id is not None:
            xyxy = b.xyxy.cpu().numpy()
            ids = b.id.cpu().numpy().astype(int)
            cls = b.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), tid, c in zip(xyxy, ids, cls):
                col = color_for(int(tid))
                is_ball = (c == 1)
                if is_ball:  # ball: a ring, so it reads even at a few pixels
                    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    cv2.circle(frame, (cx, cy), 13, (0, 240, 255), 2, cv2.LINE_AA)
                    cv2.putText(frame, "ball", (cx + 16, cy - 8), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 240, 255), 1, cv2.LINE_AA)
                    continue
                live += 1
                p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
                cv2.rectangle(frame, p1, p2, col, 2, cv2.LINE_AA)
                lab = f"{int(tid)}"
                (tw, th), _ = cv2.getTextSize(lab, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (p1[0], p1[1] - th - 7), (p1[0] + tw + 8, p1[1]), col, -1)
                cv2.putText(frame, lab, (p1[0] + 4, p1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (20, 20, 20), 1, cv2.LINE_AA)
                if args.trail:
                    trails[int(tid)].append((int((x1 + x2) / 2), int(y2)))  # feet point
                    pts = list(trails[int(tid)])
                    for i in range(1, len(pts)):
                        cv2.line(frame, pts[i - 1], pts[i], col, max(1, int(3 * i / len(pts))), cv2.LINE_AA)

        draw_header(frame, f"follow-anything  |  {live} players tracked simultaneously",
                    f"fine-tuned YOLO11m + BoT-SORT  |  {clip.name}  |  held-out match (never trained on)")
        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
        writer.write(frame)
        n += 1
        if n % 50 == 0:
            print(f"  {n} frames ...", flush=True)

    if writer is not None:
        writer.release()
    print(f"\nwrote {out_path}  ({n} frames, {n/args.fps:.1f}s)")


if __name__ == "__main__":
    main()
