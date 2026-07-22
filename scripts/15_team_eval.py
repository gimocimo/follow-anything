#!/usr/bin/env python3
"""Measure team-assignment accuracy against SN-GSR-2025's `attributes.team` labels.

Team assignment was previously eyeballed. GSR annotates the true team, so it can be scored.
Cluster labels are arbitrary, so both permutations are tried and the better one kept; only
players with a GT team are scored (referees/goalkeepers have no team).

    python scripts/15_team_eval.py --max-seqs 4
"""
import argparse, json, sys
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pitchvision.config import get_device
from pitchvision.demo.teams import (assign_teams, gt_team_boxes, sample_frame, score_assignment,
                                    true_team_of_tracks)
from pitchvision.pipeline.run_video import _resolve_tracker


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits-file", default="outputs/gsr_splits.json")
    ap.add_argument("--split", default="test")
    ap.add_argument("--weights", default="models/yolo11m_gsr_ft.pt")
    ap.add_argument("--tracker", default="configs/trackers/botsort_newtrk040.yaml")
    ap.add_argument("--classes", default="0,1")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-seqs", type=int, default=4)
    ap.add_argument("--max-frames", type=int, default=250)
    ap.add_argument("--ks", default="3", help="comma-separated k values to sweep")
    ap.add_argument("--out", default="outputs/team_eval.json")
    args = ap.parse_args()

    from ultralytics import YOLO
    seqs = json.loads(Path(args.splits_file).read_text())[args.split][: args.max_seqs]
    model = YOLO(args.weights)
    ks = [int(x) for x in args.ks.split(",")]
    per_clip = []
    # A/B: naive (all torso samples, unweighted) vs occlusion-aware + confidence-weighted.
    agg = {kv: {"naive": [0.0, 0], "improved": [0.0, 0]} for kv in ks}

    for s in seqs:
        samples, boxes, areas = defaultdict(list), defaultdict(list), defaultdict(list)
        res = model.track(source=str(Path(s["path"]) / "img1"),
                          classes=[int(c) for c in args.classes.split(",")], conf=args.conf,
                          imgsz=args.imgsz, tracker=_resolve_tracker(args.tracker), persist=True,
                          stream=True, device=get_device(args.device), verbose=False)
        for n, r in enumerate(res):
            if n >= args.max_frames:
                break
            b = r.boxes
            if b is None or b.id is None:
                continue
            frame = cv2.imread(r.path)
            digits = "".join(c for c in Path(r.path).stem if c.isdigit())
            fr = int(digits) if digits else n + 1
            dets = []
            for (x1, y1, x2, y2), tid, cl, cf in zip(
                    b.xyxy.cpu().numpy(), b.id.cpu().numpy().astype(int),
                    b.cls.cpu().numpy().astype(int), b.conf.cpu().numpy()):
                if cl != 0:
                    continue
                box = (int(x1), int(y1), int(x2), int(y2))
                boxes[int(tid)].append((fr, (float(x1), float(y1), float(x2 - x1), float(y2 - y1))))
                areas[int(tid)].append((x2 - x1) * (y2 - y1))
                dets.append((int(tid), float(cf), box))
            if frame is not None:
                for tid, lab, w, occ in sample_frame(frame, dets):
                    samples[tid].append((lab, w, occ))
        weights = {t: float(sum(v) / len(v)) for t, v in areas.items()}   # area-weight the ACCURACY
        truth = true_team_of_tracks(boxes, gt_team_boxes(s["labels"]))
        row = {"clip": s["name"]}
        for kv in ks:
            naive, _ = assign_teams(samples, k=kv, occlusion_aware=False, conf_weighted=False)
            impr, conf = assign_teams(samples, k=kv)
            sc_n = score_assignment(naive, truth, weights=weights)
            sc_i = score_assignment(impr, truth, weights=weights)
            mc = round(float(np.mean([c["confidence"] for c in conf.values()])), 3) if conf else None
            row[f"k{kv}"] = {"naive": sc_n, "improved": sc_i, "mean_confidence": mc}
            for key, sc in (("naive", sc_n), ("improved", sc_i)):
                if sc["accuracy"] is not None:
                    agg[kv][key][0] += sc["accuracy"] * sc["n"]
                    agg[kv][key][1] += sc["n"]
        per_clip.append(row)
        print("  " + s["name"] + "  " + "  ".join(
            (f"k{kv}: {row[f'k{kv}']['naive']['accuracy']:.3f}->{row[f'k{kv}']['improved']['accuracy']:.3f}"
             if row[f'k{kv}']['improved']['accuracy'] is not None else f"k{kv}:n/a") for kv in ks), flush=True)

    print("\n==== team assignment vs GSR labels (weighted accuracy: naive -> occlusion-aware + conf-weighted) ====")
    best = None
    for kv in ks:
        (nw, nn), (iw, ino) = agg[kv]["naive"], agg[kv]["improved"]
        nv, iv = (nw / nn if nn else float("nan")), (iw / ino if ino else float("nan"))
        print(f"  k={kv}:  {nv:.4f} -> {iv:.4f}  ({iv - nv:+.4f})  over {ino} player tracks", flush=True)
        if ino:
            best = iv if best is None else max(best, iv)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"overall_accuracy": best, "per_clip": per_clip}, indent=2))
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
