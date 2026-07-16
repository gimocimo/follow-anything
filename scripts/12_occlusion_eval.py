#!/usr/bin/env python3
"""Occlusion analysis (Rung 3): does the tracker RECOVER a player's identity after an
occlusion? A dev-only proxy (no tuning on the sealed set) that isolates occlusion, which
global AssA/IDSW only blend together.

Method: for each GT person identity we find *visibility gaps* — frames where its box
disappears for 1..maxgap frames then returns (occlusion or a brief off-screen). We match
tracker IDs to GT IDs per frame by IoU >= iou_thr, then for each gap ask: did the tracker
keep the SAME track ID before and after the gap? recovery_rate = recovered / total gaps,
stratified by gap length (short gaps are easier). Persons only (players/GK/referee) —
the occlusion-relevant class (the ball's problem is detection, not occlusion recovery).

    python scripts/12_occlusion_eval.py --weights models/yolo11m_gsr_ft.pt --classes 0,1 \
      --tracker configs/trackers/botsort_newtrk040.yaml --split test --out-dir outputs/gsr_ft_dev
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pitchvision.data.gsr import gsr_to_mot_rows
from pitchvision.pipeline.run_video import run_image_folder

PERSON_CATS = {1, 2, 3}


def iou(a, b):
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = a[2] * a[3] + b[2] * b[3] - inter
    return inter / ua if ua > 0 else 0.0


def match_per_frame(gt_rows, tr_rows, iou_thr):
    """frame -> {gt_id: track_id} via greedy IoU (highest-IoU first, one-to-one)."""
    tr_by_fr, gt_by_fr = defaultdict(list), defaultdict(list)
    for (f, tid, x, y, w, h, *_) in tr_rows:
        tr_by_fr[f].append((tid, (x, y, w, h)))
    for (f, gid, x, y, w, h, *_) in gt_rows:
        gt_by_fr[f].append((gid, (x, y, w, h)))
    assign = defaultdict(dict)
    for f, gts in gt_by_fr.items():
        cands = []
        for gid, gb in gts:
            for tid, tb in tr_by_fr.get(f, []):
                v = iou(gb, tb)
                if v >= iou_thr:
                    cands.append((v, gid, tid))
        cands.sort(reverse=True)
        ug, ut = set(), set()
        for v, gid, tid in cands:
            if gid not in ug and tid not in ut:
                assign[f][gid] = tid
                ug.add(gid); ut.add(tid)
    return assign


def analyze(gt_rows, tr_rows, maxgap, iou_thr):
    assign = match_per_frame(gt_rows, tr_rows, iou_thr)
    frames_of = defaultdict(set)
    for (f, gid, *_) in gt_rows:
        frames_of[gid].add(f)
    by_len = defaultdict(lambda: [0, 0])   # gap_len -> [recovered, total] (direct occlusion recovery)
    frag = []                              # distinct tracker IDs per GT person (1 = cleanly tracked)
    for gid, frs in frames_of.items():
        frs = sorted(frs)
        ids = {assign.get(f, {}).get(gid) for f in frs}
        ids.discard(None)
        if ids:
            frag.append(len(ids))
        for i in range(len(frs) - 1):
            gap = frs[i + 1] - frs[i] - 1
            if 1 <= gap <= maxgap:
                before = assign.get(frs[i], {}).get(gid)
                after = assign.get(frs[i + 1], {}).get(gid)
                ok = (before is not None and before == after)
                by_len[gap][1] += 1
                by_len[gap][0] += 1 if ok else 0
    return by_len, frag


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
    ap.add_argument("--maxgap", type=int, default=30, help="max gap length (frames) counted as occlusion (~1s)")
    ap.add_argument("--iou-thr", type=float, default=0.5)
    ap.add_argument("--out-dir", default="outputs/gsr_ft_dev")
    args = ap.parse_args()

    data = json.loads(Path(args.splits_file).read_text())
    seqs = data[args.split]
    if args.max_seqs:
        seqs = seqs[: args.max_seqs]
    classes = tuple(int(c) for c in args.classes.split(","))

    agg = defaultdict(lambda: [0, 0])
    all_frag = []
    for k, s in enumerate(seqs, 1):
        print(f"  [{k}/{len(seqs)}] {s['name']} ...", flush=True)
        tr_rows, _ = run_image_folder(Path(s["path"]) / "img1", weights=args.weights, classes=classes,
                                      conf=args.conf, imgsz=args.imgsz, tracker=args.tracker, device=args.device)
        gt_rows = gsr_to_mot_rows(s["labels"], keep_categories=PERSON_CATS, strict=True)
        by_len, frag = analyze(gt_rows, tr_rows, args.maxgap, args.iou_thr)
        for L, (r, t) in by_len.items():
            agg[L][0] += r
            agg[L][1] += t
        all_frag += frag

    tot_r = sum(v[0] for v in agg.values())
    tot_t = sum(v[1] for v in agg.values())
    buckets = {"1-3": [0, 0], "4-10": [0, 0], "11-30": [0, 0]}
    for L, (r, t) in agg.items():
        b = "1-3" if L <= 3 else ("4-10" if L <= 10 else "11-30")
        buckets[b][0] += r
        buckets[b][1] += t

    rate = tot_r / tot_t if tot_t else 0.0
    nf = len(all_frag)
    mean_frag = sum(all_frag) / nf if nf else 0.0
    single = sum(1 for x in all_frag if x == 1)
    print(f"\n==== Rung-3 occlusion analysis (dev persons, {Path(args.weights).name}) ====")
    print(f"  ID fragmentation: mean {mean_frag:.2f} tracker-IDs per GT person | "
          f"cleanly tracked (1 ID): {single}/{nf} ({100*single/nf if nf else 0:.0f}%)")
    print(f"  recovery across visibility gaps (1..{args.maxgap} fr): {tot_r}/{tot_t} = {rate:.3f}")
    for b, (r, t) in buckets.items():
        if t:
            print(f"    gap {b:>5} fr: {r/t:.3f}  ({r}/{t})")
    dst = Path(args.out_dir) / "occlusion_metrics.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps({
        "fragmentation": {"mean_ids_per_gt": mean_frag, "clean_single_id": single, "n_gt_tracks": nf},
        "gap_recovery": {"rate": rate, "gaps": tot_t, "recovered": tot_r, "maxgap": args.maxgap,
                         "iou_thr": args.iou_thr,
                         "buckets": {b: {"recovered": v[0], "total": v[1]} for b, v in buckets.items()}},
    }, indent=2))
    print(f"saved -> {dst}")


if __name__ == "__main__":
    main()
