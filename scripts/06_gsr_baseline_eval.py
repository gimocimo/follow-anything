#!/usr/bin/env python3
"""Zero-training baseline on SN-GSR-2025: YOLO11n + BoT-SORT over the clips,
scored with bbox HOTA / MOTA / IDF1 against the GSR image-box ground truth.

NB: SoccerNet's *official* GSR metric is GS-HOTA over pitch coordinates (the
flagship, rung 6). This is the standard bbox tracking baseline (rungs 1-3) — the
number every later rung must beat.

    python scripts/06_gsr_baseline_eval.py --split test --max-seqs 3   # quick
    python scripts/06_gsr_baseline_eval.py --split test                # full split
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pitchvision.pipeline.run_video import run_image_folder
from pitchvision.eval.mot_eval import evaluate, write_gt, write_tracker
from pitchvision.data.gsr import gsr_to_mot_rows

TRACKER = "yolo11n_botsort"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--splits-file", default="outputs/gsr_splits.json")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--weights", default="yolo11n.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--tracker", default="botsort.yaml")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-seqs", type=int, default=None,
                    help="limit #sequences for a quick check (default: all)")
    ap.add_argument("--out-dir", default="outputs/gsr_baseline")
    args = ap.parse_args()

    data = json.loads(Path(args.splits_file).read_text())
    seqs = data[args.split]
    if args.max_seqs:
        seqs = seqs[: args.max_seqs]
    if not seqs:
        sys.exit(f"No sequences in split '{args.split}' of {args.splits_file}")

    subset = f"  [SUBSET: first {len(seqs)}]" if args.max_seqs else ""
    print(f"GSR baseline on split='{args.split}'{subset} — {len(seqs)} sequence(s)\n")

    pred_dir = Path(args.out_dir) / "trackers"
    gt_tmp = Path(tempfile.mkdtemp(prefix="pv_gsrgt_"))
    seq_lengths = {}
    for k, s in enumerate(seqs, 1):
        name, seq_path = s["name"], Path(s["path"])
        print(f"  [{k}/{len(seqs)}] tracking {name} ...", flush=True)
        rows, nframes = run_image_folder(
            seq_path / "img1", weights=args.weights, conf=args.conf,
            imgsz=args.imgsz, tracker=args.tracker, device=args.device,
        )
        write_tracker(pred_dir, TRACKER, name, rows)
        write_gt(gt_tmp, name, gsr_to_mot_rows(s["labels"]))
        seq_lengths[name] = max(nframes, s["length"] or nframes)

    res = evaluate(str(gt_tmp), str(pred_dir), seq_lengths, trackers_to_eval=[TRACKER])
    m = res[TRACKER]
    print("\n==== GSR BASELINE (YOLO11n + BoT-SORT, zero training, bbox-HOTA) ====")
    print(f"  HOTA {m['HOTA']:.4f} | MOTA {m['MOTA']:.4f} | IDF1 {m['IDF1']:.4f}")
    print(f"  DetA {m['DetA']:.4f} | AssA {m['AssA']:.4f} | IDSW {m['IDSW']}")

    out = Path(args.out_dir) / "baseline_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"split": args.split, "n_seqs": len(seqs), "subset": bool(args.max_seqs), "metrics": m},
        indent=2,
    ))
    print(f"  saved -> {out}")


if __name__ == "__main__":
    main()
