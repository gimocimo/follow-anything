#!/usr/bin/env python3
"""Per-class tracking eval (Rung 3): score the BALL (tiny, fast, single-instance) and
the PERSON class separately, so "ball measured" in the Rung-3 gate is explicit and we
can see where association is hard.

Runs the same detector + tracker as scripts/06, but keeps the detector class per
detection, then scores each class group with TrackEval independently (GT filtered to the
matching GSR categories). The 'all' group reproduces the headline number as a sanity check.

    python scripts/11_perclass_eval.py --weights models/yolo11m_gsr_ft.pt --classes 0,1 \
      --tracker configs/trackers/botsort_newtrk040.yaml --split test --out-dir outputs/gsr_ft_dev
"""
import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pitchvision.config import get_device
from pitchvision.data.gsr import gsr_to_mot_rows
from pitchvision.eval.mot_eval import evaluate, write_gt, write_tracker
from pitchvision.pipeline.run_video import _resolve_tracker

# detector class id -> GT categories. Fine-tuned GSR model: 0=person, 1=ball.
GROUPS = {
    "all":    {"det_cls": {0, 1}, "gt_cats": None},        # None -> all object cats (== scripts/06)
    "person": {"det_cls": {0},    "gt_cats": {1, 2, 3}},   # player / GK / referee
    "ball":   {"det_cls": {1},    "gt_cats": {4}},
}


def track_with_class(img_dir, weights, classes, conf, imgsz, tracker, device):
    """Like run_image_folder, but each row carries the detector class:
    (frame, id, x, y, w, h, conf, cls)."""
    from ultralytics import YOLO
    model = YOLO(weights)
    dev = get_device(device)
    rows = []
    results = model.track(source=str(img_dir), classes=list(classes), conf=conf, imgsz=imgsz,
                          tracker=_resolve_tracker(tracker), persist=True, stream=True,
                          device=dev, verbose=False)
    for r in results:
        digits = "".join(c for c in Path(r.path).stem if c.isdigit())
        if not digits:
            continue
        fr = int(digits)
        b = r.boxes
        if b is None or b.id is None:
            continue
        xywh = b.xywh.cpu().numpy()
        ids = b.id.cpu().numpy()
        confs = b.conf.cpu().numpy()
        cls = b.cls.cpu().numpy()
        for (cx, cy, bw, bh), tid, cf, cl in zip(xywh, ids, confs, cls):
            rows.append((fr, int(tid), cx - bw / 2, cy - bh / 2, bw, bh, float(cf), int(cl)))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits-file", default="outputs/gsr_splits.json")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--weights", default="models/yolo11m_gsr_ft.pt")
    ap.add_argument("--classes", default="0,1")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--tracker", default="configs/trackers/botsort_newtrk040.yaml")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-seqs", type=int, default=None)
    ap.add_argument("--out-dir", default="outputs/gsr_ft_dev")
    args = ap.parse_args()

    data = json.loads(Path(args.splits_file).read_text())
    seqs = data[args.split]
    if args.max_seqs:
        seqs = seqs[: args.max_seqs]
    classes = tuple(int(c) for c in args.classes.split(","))

    # 1) inference once per sequence (capturing class), reused for every group
    all_tracks, seq_lengths = {}, {}
    for k, s in enumerate(seqs, 1):
        name, seq_path = s["name"], Path(s["path"])
        print(f"  [{k}/{len(seqs)}] tracking {name} ...", flush=True)
        rows = track_with_class(seq_path / "img1", args.weights, classes, args.conf,
                                args.imgsz, args.tracker, args.device)
        all_tracks[name] = rows
        seq_lengths[name] = s.get("length") or (max((r[0] for r in rows), default=0))

    # 2) score each class group with the shared TrackEval path
    out = {}
    for g, spec in GROUPS.items():
        gt_dir = Path(tempfile.mkdtemp(prefix=f"pv_gt_{g}_"))
        pred_dir = Path(tempfile.mkdtemp(prefix=f"pv_pred_{g}_"))
        try:
            for s in seqs:
                name = s["name"]
                trows = [(r[0], r[1], r[2], r[3], r[4], r[5], r[6])
                         for r in all_tracks[name] if r[7] in spec["det_cls"]]
                write_tracker(pred_dir, "ft", name, trows)
                write_gt(gt_dir, name, gsr_to_mot_rows(s["labels"], keep_categories=spec["gt_cats"]))
            m = evaluate(str(gt_dir), str(pred_dir), seq_lengths, trackers_to_eval=["ft"])["ft"]
            out[g] = {k: m[k] for k in ("HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW")}
        finally:
            shutil.rmtree(gt_dir, ignore_errors=True)
            shutil.rmtree(pred_dir, ignore_errors=True)

    print(f"\n==== Rung-3 per-class ({Path(args.weights).name}, split={args.split}) ====")
    print(f"{'class':8} {'HOTA':>8} {'DetA':>8} {'AssA':>8} {'MOTA':>8} {'IDF1':>8} {'IDSW':>7}")
    for g in ("all", "person", "ball"):
        m = out[g]
        print(f"{g:8} {m['HOTA']:8.4f} {m['DetA']:8.4f} {m['AssA']:8.4f} "
              f"{m['MOTA']:8.4f} {m['IDF1']:8.4f} {m['IDSW']:7d}")

    dst = Path(args.out_dir) / "perclass_metrics.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps({"split": args.split, "weights": args.weights, "per_class": out}, indent=2))
    print(f"saved -> {dst}")


if __name__ == "__main__":
    main()
