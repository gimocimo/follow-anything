#!/usr/bin/env python3
"""Zero-training baseline on SN-GSR-2025: YOLO11n + BoT-SORT over held-out clips,
scored with bbox HOTA / MOTA / IDF1 against the GSR image-box ground truth.

Category policy: GT includes ALL object categories (players/GK/referee/ball +
"other"); per-class counts are recorded. Writes a **provenance manifest** (git SHA,
package versions, device, weight hash, args, category counts) next to the metrics,
so the number is reproducible from committed inputs.

NB: SoccerNet's official GSR metric is GS-HOTA over pitch coordinates (rung-6
flagship). This is the standard bbox tracking baseline.

    python scripts/06_gsr_baseline_eval.py --split test --max-seqs 3   # quick
    python scripts/06_gsr_baseline_eval.py --split test                # full split
"""
import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pitchvision.config import get_device
from pitchvision.data.gsr import gsr_category_counts, gsr_to_mot_rows
from pitchvision.eval.mot_eval import evaluate, write_gt, write_tracker
from pitchvision.pipeline.run_video import run_image_folder

TRACKER = "yolo11n_botsort"


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
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
    proj = Path(__file__).resolve().parents[1]
    try:
        return subprocess.check_output(["git", "-C", str(proj), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def build_manifest(args, device, weights, splits_meta, category_counts):
    from pitchvision.pipeline.run_video import _resolve_tracker
    wp = Path(weights)
    resolved_tracker = _resolve_tracker(args.tracker)
    return {
        "git_sha": _git_sha(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": device,
        "packages": _versions(["torch", "torchvision", "ultralytics", "numpy",
                                "opencv-python", "scipy", "trackeval", "lap"]),
        "detector": {"weights": str(wp), "sha256": _sha256(wp) if wp.exists() else None,
                     "imgsz": args.imgsz, "conf": args.conf,
                     "classes": "COCO person(0) + sports_ball(32)"},
        "tracker": {"requested": args.tracker, "resolved": resolved_tracker,
                    "sha256": _sha256(Path(resolved_tracker)) if Path(resolved_tracker).exists() else None},
        "category_policy": "all object categories (player/GK/referee/ball + other); see src/pitchvision/data/gsr.py",
        "gt_category_counts": category_counts,
        "split": {"file": str(args.splits_file), "name": args.split, "meta": splits_meta},
        "args": vars(args),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits-file", default="outputs/gsr_splits.json")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--weights", default="yolo11n.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--tracker", default="botsort.yaml")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-seqs", type=int, default=None, help="limit #sequences for a quick check")
    ap.add_argument("--out-dir", default="outputs/gsr_baseline")
    args = ap.parse_args()

    data = json.loads(Path(args.splits_file).read_text())
    seqs = data[args.split]
    if args.max_seqs:
        seqs = seqs[: args.max_seqs]
    if not seqs:
        sys.exit(f"No sequences in split '{args.split}' of {args.splits_file}")

    role = data.get("meta", {}).get("role", "?")
    subset = f"  [SUBSET: first {len(seqs)}]" if args.max_seqs else ""
    print(f"GSR baseline on split='{args.split}' (role={role}){subset} — {len(seqs)} sequence(s)\n")

    device = get_device(args.device)
    pred_dir = Path(args.out_dir) / "trackers"
    gt_tmp = Path(tempfile.mkdtemp(prefix="pv_gsrgt_"))
    cat_counts: Counter = Counter()
    try:
        seq_lengths = {}
        for k, s in enumerate(seqs, 1):
            name, seq_path = s["name"], Path(s["path"])
            print(f"  [{k}/{len(seqs)}] tracking {name} ...", flush=True)
            rows, nframes = run_image_folder(
                seq_path / "img1", weights=args.weights, conf=args.conf,
                imgsz=args.imgsz, tracker=args.tracker, device=args.device,
            )
            write_tracker(pred_dir, TRACKER, name, rows)
            write_gt(gt_tmp, name, gsr_to_mot_rows(s["labels"], strict=True))
            for cn, cv in gsr_category_counts(s["labels"]).items():
                cat_counts[cn] += cv
            seq_lengths[name] = max(nframes, s.get("length") or nframes)
        res = evaluate(str(gt_tmp), str(pred_dir), seq_lengths, trackers_to_eval=[TRACKER])
    finally:
        shutil.rmtree(gt_tmp, ignore_errors=True)

    m = res[TRACKER]
    print("\n==== GSR BASELINE (YOLO11n + BoT-SORT, zero training, bbox-HOTA) ====")
    print(f"  HOTA {m['HOTA']:.4f} | MOTA {m['MOTA']:.4f} | IDF1 {m['IDF1']:.4f}")
    print(f"  DetA {m['DetA']:.4f} | AssA {m['AssA']:.4f} | IDSW {m['IDSW']}")
    print(f"  GT category counts: {dict(cat_counts)}")

    manifest = build_manifest(args, device, args.weights, data.get("meta", {}), dict(cat_counts))
    out = Path(args.out_dir) / "baseline_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"split": args.split, "role": role, "n_seqs": len(seqs),
         "subset": bool(args.max_seqs), "metrics": m, "provenance": manifest}, indent=2))
    print(f"  saved -> {out}  (with provenance manifest)")


if __name__ == "__main__":
    main()
