#!/usr/bin/env python3
"""Rung-1 smoke test: track players + ball in a football clip with ZERO training.

    python scripts/00_smoke_test.py --video data/sample.mp4

Outputs an annotated video and MOTChallenge-format results in outputs/.
"""
import argparse
import sys
from pathlib import Path

# make src/ importable without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pitchvision.pipeline.run_video import run_video


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", required=True, help="path to a football clip (mp4)")
    ap.add_argument("--weights", default="yolo11n.pt", help="YOLO weights (auto-downloads)")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--tracker", default="botsort.yaml", help="botsort.yaml | bytetrack.yaml")
    ap.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()

    if not Path(args.video).exists():
        sys.exit(f"[error] video not found: {args.video}\n"
                 f"        drop a short football clip there, or pass --video <path>.")

    info = run_video(
        video=args.video,
        weights=args.weights,
        imgsz=args.imgsz,
        conf=args.conf,
        tracker=args.tracker,
        device=args.device,
        out_dir=args.out,
    )
    print(f"[ok] device={info['device']}  frames={info['frames']}")
    print(f"[ok] annotated video: {info['video']}")
    print(f"[ok] MOT results:     {info['mot']}")


if __name__ == "__main__":
    main()
