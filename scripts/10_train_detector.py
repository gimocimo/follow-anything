#!/usr/bin/env python3
"""Fine-tune YOLO11 on the GSR YOLO dataset built by scripts/09 (Rung 3, 3c).

**Binds the checkpoint to the exact dataset bytes it consumed.** Before training (and again
after) the on-disk dataset is re-hashed against `export_manifest.json` / `export_inventory.json`;
anything missing, modified, or **extra** aborts the run. An export-time fingerprint alone cannot
detect a frame added *afterwards* — ultralytics globs the whole tree via `data.yaml` — so without
this preflight a checkpoint is not provably leave-one-game-out (PROJECT_PLAN §3).

Writes `train_manifest.json` beside the weights: resolved args, base-weight hash, versions/env,
the dataset `export_sha256`, the pre/post verification reports, and the **output `best.pt` SHA** —
one unbroken chain from dataset bytes to checkpoint to metrics.

    python scripts/10_train_detector.py --data data/yolo_gsr_dev/data.yaml \
        --weights yolo11m.pt --epochs 60 --imgsz 1280 --device 0
    python scripts/10_train_detector.py --resume runs/gsr_ft/m_1280/weights/last.pt
"""
import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pitchvision.data.export_verify import format_report, sha256_file, verify_export


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


def _data_from_resume(ckpt):
    """Recover the data.yaml path from a run's args.yaml. NB: args.yaml is YAML, not JSON —
    parsing it with json.loads silently fails and nulls the dataset provenance."""
    args_yaml = Path(ckpt).resolve().parents[1] / "args.yaml"
    if args_yaml.exists():
        try:
            return (yaml.safe_load(args_yaml.read_text()) or {}).get("data")
        except Exception:
            return None
    return None


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
    ap.add_argument("--resume", default=None, help="resume from a run's last.pt")
    ap.add_argument("--skip-verify", action="store_true",
                    help="train even if the dataset fails verification (the checkpoint will NOT be "
                         "provably leave-one-game-out; recorded as such in the manifest)")
    args = ap.parse_args()

    from ultralytics import YOLO

    data_yaml = args.data or (_data_from_resume(args.resume) if args.resume else None)
    dataset_dir = Path(data_yaml).parent if data_yaml else None

    # ---- PREFLIGHT: dataset on disk must be byte-identical to its manifest ----
    pre = None
    if dataset_dir:
        pre = verify_export(dataset_dir)
        print(f"[preflight] {format_report(pre)}", flush=True)
        if not pre["ok"] and not args.skip_verify:
            sys.exit("ABORTING: the dataset does not match its export manifest (above). Rebuild it with "
                     "`scripts/09_gsr_to_yolo.py --overwrite`, or pass --skip-verify to train anyway "
                     "(the checkpoint will NOT be provably leave-one-game-out).")
    else:
        print("[preflight] WARNING: no data.yaml resolved — dataset provenance NOT captured.", flush=True)

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

    # ---- POSTFLIGHT: catch anything that changed *during* the run ----
    post = verify_export(dataset_dir) if dataset_dir else None
    if post:
        print(f"[postflight] {format_report(post)}", flush=True)

    export_info = None
    if dataset_dir and (dataset_dir / "export_manifest.json").exists():
        export_info = json.loads((dataset_dir / "export_manifest.json").read_text())
    resolved = {}
    try:
        resolved = {k: v for k, v in vars(model.trainer.args).items()}
    except Exception:
        pass

    manifest = {
        "base_weights": args.weights, "base_sha256": sha256_file(args.weights) if Path(args.weights).exists() else None,
        "data_yaml": data_yaml,
        "dataset_export": export_info,
        "dataset_verified_pre": pre, "dataset_verified_post": post,
        "provably_leave_one_game_out": bool(pre and pre.get("ok") and post and post.get("ok")),
        "output_best_sha256": sha256_file(best) if best.exists() else None,
        "requested": {"epochs": args.epochs, "imgsz": args.imgsz, "batch": args.batch,
                      "workers": args.workers, "amp": args.amp, "device": args.device,
                      "resume": args.resume, "skip_verify": args.skip_verify},
        "resolved_args": resolved,
        "versions": _versions(["ultralytics", "torch", "torchvision", "numpy", "lap"]),
        "git_sha": _git_sha(), "python": platform.python_version(), "platform": platform.platform(),
    }
    man_path = save_dir / "train_manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2, default=str))

    print(f"\nDONE. best weights -> {best}")
    print(f"train manifest -> {man_path}")
    print(f"  provably leave-one-game-out: {manifest['provably_leave_one_game_out']}"
          + (f"  (games {export_info.get('observed_games')}, export_sha256 "
             f"{str(export_info.get('export_sha256'))[:16]}…)" if export_info else ""))
    print(f"  best.pt sha256: {str(manifest['output_best_sha256'])[:16]}…")
    print("Bring HOME: best.pt + train_manifest.json + the dataset's export_manifest.json + export_inventory.json.")
    print("Evaluate (same sealed eval as the baseline):")
    print(f"  python scripts/06_gsr_baseline_eval.py --weights {best} --classes 0,1 \\")
    print(f"    --tracker configs/trackers/botsort_newtrk040.yaml --split test --out-dir outputs/gsr_ft_dev")


if __name__ == "__main__":
    main()
