#!/usr/bin/env python3
"""Fine-tune YOLO11 on the GSR YOLO dataset built by scripts/09 (Rung 3, 3c).

**Binds the checkpoint to the exact dataset bytes it consumed**, at three levels:

  1. **Directory** — every file is re-hashed against `export_manifest`/`export_inventory`
     before training and again after; missing / modified / **extra** anything aborts.
  2. **Loader config** — `data.yaml` is authenticated: its resolved `path`/`train`/`val` must
     point at the manifested directory. (Verifying only the directory is insufficient: the
     YAML handed to ultralytics can aim `train:` at any other folder while the directory
     itself still verifies clean.)
  3. **Resolved file list** — an `on_train_start` guard compares **ultralytics' own**
     `train_loader.dataset.im_files` against the inventory. This is the authoritative check:
     whatever the YAML says, this is what the loader actually enumerated, so YAML redirection,
     nested subdirectories and exotic image formats cannot slip past it.

Writes `train_manifest.json` beside the weights: resolved args, base-weight hash, `data.yaml`
hash, versions/env, the dataset `export_sha256`, pre/post verification, the loader-guard
result, and the output `best.pt` SHA — one unbroken chain from dataset bytes to checkpoint.

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
    """Recover data.yaml from a run's args.yaml. NB: args.yaml is YAML, not JSON — parsing it
    with json.loads silently fails and nulls the dataset provenance."""
    args_yaml = Path(ckpt).resolve().parents[1] / "args.yaml"
    if not args_yaml.exists():
        return None
    try:
        return (yaml.safe_load(args_yaml.read_text()) or {}).get("data")
    except Exception:
        return None


def check_data_yaml(data_yaml, dataset_dir):
    """The loader reads data.yaml, not the directory — so pin it to the manifested directory."""
    problems = []
    try:
        cfg = yaml.safe_load(Path(data_yaml).read_text()) or {}
    except Exception as e:
        return [f"unreadable data.yaml: {e}"]
    dd = Path(dataset_dir).resolve()
    root = Path(cfg.get("path") or Path(data_yaml).parent).resolve()
    if root != dd:
        problems.append(f"data.yaml path={root} != manifested dataset dir {dd}")
    for split in ("train", "val"):
        raw = cfg.get(split)
        if not raw:
            problems.append(f"data.yaml has no '{split}' entry")
            continue
        for item in ([raw] if isinstance(raw, str) else list(raw)):
            p = (root / item).resolve()
            if not (p == dd or dd in p.parents):
                problems.append(f"data.yaml {split}={p} escapes the manifested dir {dd}")
    return problems


def make_loader_guard(dataset_dir, state):
    """on_train_start: compare ultralytics' RESOLVED file list against the inventory.

    The authoritative check — it inspects what the loader actually enumerated, so it cannot
    be fooled by a redirected data.yaml, a nested subdirectory, or an unexpected image format.
    """
    inv = json.loads((Path(dataset_dir) / "export_inventory.json").read_text())
    expected = {str((Path(dataset_dir) / "images" / e["subset"] / f"{e['stem']}.jpg").resolve())
                for e in inv}

    def guard(trainer):
        actual = set()
        for name in ("train_loader", "test_loader"):
            ds = getattr(getattr(trainer, name, None), "dataset", None)
            for f in (getattr(ds, "im_files", None) or []):
                actual.add(str(Path(f).resolve()))
        if actual != expected:
            extra = sorted(actual - expected)[:5]
            missing = sorted(expected - actual)[:5]
            state["loader_guard"] = {"ok": False, "n_actual": len(actual), "n_expected": len(expected),
                                     "extra": extra, "missing": missing}
            raise RuntimeError(
                f"LOADER MISMATCH — refusing to train on unverified data. ultralytics resolved "
                f"{len(actual)} files; the manifest authorises {len(expected)}. "
                f"extra={extra} missing={missing}")
        state["loader_guard"] = {"ok": True, "n_files": len(actual)}
        print(f"[loader-guard] ultralytics resolved exactly the {len(actual)} manifested files ✅",
              flush=True)
    return guard


def make_stopfile_guard(stop_path, state):
    """Graceful early exit that PRESERVES the proof. If `stop_path` appears mid-run, finish the
    current epoch, run final validation, save best.pt, and let postflight write the manifest — so
    a run stopped to cap GPU spend is STILL provably leave-one-game-out. Registered on
    'on_fit_epoch_end', which fires AFTER ultralytics' own early-stopper sets trainer.stop, so this
    can only ADD a stop, never clear one. Single-GPU only (same constraint as the loader guard)."""
    sp = Path(stop_path)

    def guard(trainer):
        if sp.exists():
            print(f"[stop-file] {sp} present -> stopping after this epoch; best.pt, final "
                  "validation and train_manifest.json will still be written.", flush=True)
            trainer.stop = True
            state["stopfile_triggered"] = True
    return guard


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="path to the data.yaml from scripts/09 (omit when --resume)")
    ap.add_argument("--weights", default="yolo11m.pt")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", default="-1", help="-1 = auto; or an int / 0<float<1 fraction")
    ap.add_argument("--device", default="0", help="CUDA index, or 'cpu' / 'mps'")
    ap.add_argument("--project", default="runs/gsr_ft")
    ap.add_argument("--name", default=None)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--stop-file", default="STOP",
                    help="graceful-stop sentinel: `touch STOP` and the run ends after the current "
                         "epoch WITH best.pt + a valid train_manifest (proof preserved). Caps GPU spend safely.")
    ap.add_argument("--workers", type=int, default=8,
                    help="dataloader workers; use 0 if training HANGS at 'Starting training'")
    ap.add_argument("--amp", default="true", choices=["true", "false"])
    ap.add_argument("--resume", default=None, help="resume from a run's last.pt")
    ap.add_argument("--skip-verify", action="store_true",
                    help="train even if verification fails (the checkpoint will NOT be provably "
                         "leave-one-game-out; recorded as such in the manifest)")
    args = ap.parse_args()

    if Path(args.stop_file).exists():
        sys.exit(f"ABORTING: stop-file '{args.stop_file}' already exists and would halt training at "
                 f"epoch 1. Remove it first:  rm {args.stop_file}")

    from ultralytics import YOLO

    # ---- resolve the dataset; resume must use the CHECKPOINT's dataset, not a passed one ----
    if args.resume:
        ck_data = _data_from_resume(args.resume)
        if not ck_data and not args.skip_verify:
            sys.exit("ABORTING: cannot resolve the dataset from the checkpoint's args.yaml — "
                     "resume provenance is unverifiable.")
        if args.data and ck_data and Path(args.data).resolve() != Path(ck_data).resolve():
            sys.exit(f"ABORTING: --data ({args.data}) differs from the checkpoint's dataset "
                     f"({ck_data}). ultralytics restores the checkpoint's dataset on resume, so "
                     f"verifying yours would prove nothing.")
        data_yaml = ck_data
    else:
        if not args.data:
            ap.error("--data is required unless --resume is given")
        data_yaml = args.data
    dataset_dir = Path(data_yaml).parent if data_yaml else None

    # ---- PREFLIGHT: directory bytes + data.yaml target ----
    pre, yaml_problems = None, []
    if dataset_dir:
        pre = verify_export(dataset_dir)
        print(f"[preflight] {format_report(pre)}", flush=True)
        yaml_problems = check_data_yaml(data_yaml, dataset_dir)
        for p in yaml_problems:
            print(f"[preflight] data.yaml PROBLEM: {p}", flush=True)
        if (not pre["ok"] or yaml_problems) and not args.skip_verify:
            sys.exit("ABORTING: dataset or data.yaml failed verification (above). Rebuild with "
                     "`scripts/09_gsr_to_yolo.py --overwrite`, or pass --skip-verify to train "
                     "anyway (the checkpoint will NOT be provably leave-one-game-out).")
    else:
        print("[preflight] WARNING: no data.yaml resolved — dataset provenance NOT captured.", flush=True)

    state = {"loader_guard": None, "stopfile_triggered": False}
    model = YOLO(args.resume) if args.resume else YOLO(args.weights)
    if dataset_dir and not args.skip_verify:
        model.add_callback("on_train_start", make_loader_guard(dataset_dir, state))
    model.add_callback("on_fit_epoch_end", make_stopfile_guard(args.stop_file, state))

    if args.resume:
        model.train(resume=True)
    else:
        name = args.name or f"{Path(args.weights).stem}_{args.imgsz}"
        batch = float(args.batch) if "." in str(args.batch) else int(args.batch)
        model.train(
            data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=batch,
            device=args.device, project=str(Path(args.project).resolve()), name=name,
            patience=args.patience, workers=args.workers, amp=(args.amp == "true"),
            mosaic=1.0, close_mosaic=10, degrees=0.0, fliplr=0.5, scale=0.5, translate=0.1,
            hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, verbose=True,
        )

    save_dir = Path(model.trainer.save_dir)
    best = save_dir / "weights" / "best.pt"

    # ---- POSTFLIGHT: fail closed if anything drifted during the run ----
    post = verify_export(dataset_dir) if dataset_dir else None
    if post:
        print(f"[postflight] {format_report(post)}", flush=True)

    export_info = None
    if dataset_dir and (dataset_dir / "export_manifest.json").exists():
        export_info = json.loads((dataset_dir / "export_manifest.json").read_text())
    try:
        resolved = {k: v for k, v in vars(model.trainer.args).items()}
    except Exception:
        resolved = {}

    guard = state["loader_guard"]
    provable = bool(pre and pre.get("ok") and post and post.get("ok")
                    and not yaml_problems and guard and guard.get("ok") and not args.skip_verify)
    manifest = {
        "base_weights": args.weights, "base_sha256": sha256_file(args.weights) if Path(args.weights).exists() else None,
        "data_yaml": data_yaml, "data_yaml_sha256": sha256_file(data_yaml) if data_yaml and Path(data_yaml).exists() else None,
        "data_yaml_problems": yaml_problems,
        "dataset_export": export_info,
        "dataset_verified_pre": pre, "dataset_verified_post": post,
        "loader_guard": guard,
        "stopped_via_stopfile": state.get("stopfile_triggered", False),
        "provably_leave_one_game_out": provable,
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
    print(f"  provably leave-one-game-out: {provable}")
    print(f"  best.pt sha256: {str(manifest['output_best_sha256'])[:16]}…")
    print("Bring HOME: best.pt + train_manifest.json + data.yaml + export_manifest.json + export_inventory.json")

    if post and not post.get("ok") and not args.skip_verify:
        sys.exit("POSTFLIGHT FAILED: the dataset changed during training — this checkpoint is NOT "
                 "provably leave-one-game-out (see report above).")


if __name__ == "__main__":
    main()
