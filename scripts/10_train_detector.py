#!/usr/bin/env python3
"""Fine-tune YOLO11 on the GSR YOLO dataset built by scripts/09 (Rung 3, 3c).

Run on a CUDA GPU (rented — see handoff/rung-3/runpod-setup.md). Produces best.pt,
which is then evaluated with the SAME sealed tracking eval as the baseline:
    python scripts/06_gsr_baseline_eval.py --weights <best.pt> --classes 0,1 \
        --tracker configs/trackers/botsort_newtrk040.yaml --split test --out-dir outputs/gsr_ft_dev

    python scripts/10_train_detector.py --data data/yolo_gsr_dev/data.yaml \
        --weights yolo11m.pt --epochs 80 --imgsz 1280 --device 0
"""
import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="path to the data.yaml from scripts/09")
    ap.add_argument("--weights", default="yolo11m.pt", help="pretrained COCO weights to start from")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", default="-1", help="-1 = auto-fit to VRAM; or an int / 0<float<1 fraction")
    ap.add_argument("--device", default="0", help="CUDA device index, or 'cpu' / 'mps'")
    ap.add_argument("--project", default="runs/gsr_ft")
    ap.add_argument("--name", default=None)
    ap.add_argument("--patience", type=int, default=20, help="early-stop patience (epochs)")
    args = ap.parse_args()

    from ultralytics import YOLO

    name = args.name or f"{Path(args.weights).stem}_{args.imgsz}"
    batch = float(args.batch) if "." in str(args.batch) else int(args.batch)
    # absolute project dir so ultralytics writes exactly there (a relative project gets
    # nested under ultralytics' own runs_dir, e.g. runs/detect/<project>)
    project = str(Path(args.project).resolve())
    model = YOLO(args.weights)
    model.train(
        data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=batch,
        device=args.device, project=project, name=name, patience=args.patience,
        # broadcast football framing is consistent -> mild geometric aug, no rotation/flip-heavy tricks
        mosaic=1.0, close_mosaic=10, degrees=0.0, fliplr=0.5, scale=0.5, translate=0.1,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, verbose=True,
    )
    # report the ACTUAL save dir from the trainer (don't reconstruct it)
    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    print(f"\nDONE. best weights -> {best}")
    print("Evaluate on the held-out game (same sealed eval as the baseline):")
    print(f"  python scripts/06_gsr_baseline_eval.py --weights {best} --classes 0,1 \\")
    print(f"    --tracker configs/trackers/botsort_newtrk040.yaml --split test --out-dir outputs/gsr_ft_dev")


if __name__ == "__main__":
    main()
