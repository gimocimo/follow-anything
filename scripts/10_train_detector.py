#!/usr/bin/env python3
"""Fine-tune YOLO11 on the GSR YOLO dataset built by scripts/09 (Rung 3, 3c).

Run on a CUDA GPU (rented — see handoff/rung-3/runpod-setup.md). Writes, next to the
weights, a `train_manifest.json` that records the resolved training args, base-weight
hash, versions/env, AND the dataset's `export_sha256` from scripts/09 — so the checkpoint
is cryptographically tied to a *provably* leave-one-game-out training set.

    python scripts/10_train_detector.py --data data/yolo_gsr_dev/data.yaml \
        --weights yolo11m.pt --epochs 60 --imgsz 1280 --device 0
    # resume after an interruption (survives disconnects if launched under nohup/tmux):
    python scripts/10_train_detector.py --resume runs/gsr_ft/m_1280/weights/last.pt
"""
import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path


def _sha256(path):
    p = Path(path)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _versions(names):
    from importlib.metadata import PackageNotFoundError, version
    out = {}
    for n in names:
        try:
            out[n] = version(n)
        except PackageNotFoundError:
            out[n] = None
    return out


def _git_sha():
    try:
        return subprocess.check_output(
            ["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"],
            text=True).strip()
    except Exception:
        return None


def _write_train_manifest(model, args, save_dir):
    """Record everything needed to reproduce + prove this run, incl. the dataset fingerprint."""
    # link to the exact scripts/09 export (its export_sha256 proves the game set)
    export_info = None
    try:
        data_yaml = args.data or json.loads((Path(save_dir) / "args.yaml").read_text()).get("data")
        if data_yaml:
            em = Path(data_yaml).parent / "export_manifest.json"
            if em.exists():
                export_info = json.loads(em.read_text())
    except Exception:
        pass
    resolved = {}
    try:
        resolved = {k: v for k, v in vars(model.trainer.args).items()}
    except Exception:
        pass
    manifest = {
        "base_weights": args.weights, "base_sha256": _sha256(args.weights),
        "data_yaml": args.data,
        "dataset_export": export_info,   # export_sha256 + requested/observed games + per-game frames
        "requested": {"epochs": args.epochs, "imgsz": args.imgsz, "batch": args.batch,
                      "workers": args.workers, "amp": args.amp, "device": args.device,
                      "resume": args.resume},
        "resolved_args": resolved,
        "versions": _versions(["ultralytics", "torch", "torchvision", "numpy", "lap"]),
        "git_sha": _git_sha(), "python": platform.python_version(), "platform": platform.platform(),
    }
    out = Path(save_dir) / "train_manifest.json"
    out.write_text(json.dumps(manifest, indent=2, default=str))
    return out, export_info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="path to the data.yaml from scripts/09 (omit when --resume)")
    ap.add_argument("--weights", default="yolo11m.pt", help="pretrained COCO weights to start from")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", default="-1", help="-1 = auto; or an int / 0<float<1 fraction")
    ap.add_argument("--device", default="0", help="CUDA index, or 'cpu' / 'mps'")
    ap.add_argument("--project", default="runs/gsr_ft")
    ap.add_argument("--name", default=None)
    ap.add_argument("--patience", type=int, default=20, help="early-stop patience (epochs)")
    ap.add_argument("--workers", type=int, default=8,
                    help="dataloader workers; use 0 if training HANGS at 'Starting training' "
                         "(low container /dev/shm deadlocks forked workers on rented pods)")
    ap.add_argument("--amp", default="true", choices=["true", "false"],
                    help="mixed precision; set false if loss/backward is NaN on very new GPUs")
    ap.add_argument("--resume", default=None,
                    help="resume a previous run from its last.pt (reads original args from the checkpoint)")
    args = ap.parse_args()

    from ultralytics import YOLO

    if args.resume:
        model = YOLO(args.resume)
        model.train(resume=True)
    else:
        if not args.data:
            ap.error("--data is required unless --resume is given")
        name = args.name or f"{Path(args.weights).stem}_{args.imgsz}"
        batch = float(args.batch) if "." in str(args.batch) else int(args.batch)
        project = str(Path(args.project).resolve())  # absolute so ultralytics writes exactly there
        model = YOLO(args.weights)
        model.train(
            data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=batch,
            device=args.device, project=project, name=name, patience=args.patience,
            workers=args.workers, amp=(args.amp == "true"),
            # broadcast football framing is consistent -> mild geometric aug
            mosaic=1.0, close_mosaic=10, degrees=0.0, fliplr=0.5, scale=0.5, translate=0.1,
            hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, verbose=True,
        )

    save_dir = Path(model.trainer.save_dir)
    best = save_dir / "weights" / "best.pt"
    man_path, export_info = _write_train_manifest(model, args, save_dir)
    print(f"\nDONE. best weights -> {best}")
    print(f"train manifest -> {man_path}")
    if export_info:
        print(f"  dataset: games {export_info.get('observed_games')} "
              f"(export_sha256 {str(export_info.get('export_sha256'))[:16]}…) — provably leave-one-game-out")
    else:
        print("  WARNING: no export_manifest.json beside data.yaml — dataset provenance NOT captured. "
              "Rebuild the set with the current scripts/09 (it emits one).")
    print("Bring HOME: best.pt + train_manifest.json + the dataset's export_manifest.json.")
    print("Evaluate (same sealed eval as the baseline):")
    print(f"  python scripts/06_gsr_baseline_eval.py --weights {best} --classes 0,1 \\")
    print(f"    --tracker configs/trackers/botsort_newtrk040.yaml --split test --out-dir outputs/gsr_ft_dev")


if __name__ == "__main__":
    main()
