#!/usr/bin/env python3
"""On-screen occlusion benchmark (Rung 3): when a player is occluded **by another player**,
does the tracker keep their identity?

Supersedes an earlier statistic that counted GT "visibility gaps". An adversarial audit showed
25/27 of those gaps were **frame-edge exits** (not occlusions) and 26/27 of the apparent
failures were **detection** failures — so the number measured edge detection coverage and was
mislabelled as occlusion recovery. It has been withdrawn.

This version:
  * **Occlusion episode** = a run of frames where a GT person is overlapped by ANOTHER GT
    person with IoU >= `--occ-thr`, entered *and* exited while that player is **interior**
    (their box never touches the frame border) — i.e. a genuine on-screen occlusion.
  * Reports **endpoint coverage** (did the tracker detect the player on both sides at all?)
    and **conditional same-ID retention** (of episodes where both endpoints WERE detected,
    how often was identity kept?) as SEPARATE numbers — detection failure is never reported
    as an identity failure.
  * Reports frame-edge exit/re-entry events separately, correctly labelled.
  * Keeps **ID fragmentation** (distinct tracker IDs per GT person track-instance).
  * Stamps full provenance (weights/tracker sha256, split, args, versions, git SHA).

    python scripts/12_occlusion_eval.py --weights models/yolo11m_gsr_ft.pt --classes 0,1 \
      --tracker configs/trackers/botsort_newtrk040.yaml --split test --out-dir outputs/gsr_ft_dev
"""
import argparse
import hashlib
import json
import platform
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pitchvision.data.gsr import gsr_to_mot_rows
from pitchvision.pipeline.run_video import _resolve_tracker, run_image_folder

PERSON_CATS = {1, 2, 3}


def iou(a, b):
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = a[2] * a[3] + b[2] * b[3] - inter
    return inter / ua if ua > 0 else 0.0


def _sha256(path):
    p = Path(path)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _git_sha():
    try:
        return subprocess.check_output(
            ["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def frame_dims(labels_json):
    data = json.loads(Path(labels_json).read_text())
    for img in data.get("images", []):
        w, h = img.get("width"), img.get("height")
        if w and h:
            return float(w), float(h)
    return None


def match_per_frame(gt_rows, tr_rows, iou_thr):
    """frame -> {gt_id: track_id} via greedy highest-IoU one-to-one matching."""
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
    return assign, gt_by_fr


def analyze(gt_rows, tr_rows, dims, occ_thr, iou_thr, edge_px, max_len):
    assign, gt_by_fr = match_per_frame(gt_rows, tr_rows, iou_thr)
    W, H = dims if dims else (None, None)

    def interior(box):
        if W is None:
            return True
        x, y, w, h = box
        return (x > edge_px and y > edge_px and (x + w) < (W - edge_px) and (y + h) < (H - edge_px))

    box_of, overlap = {}, {}
    for f, gts in gt_by_fr.items():
        for gid, gb in gts:
            box_of[(f, gid)] = gb
            m = 0.0
            for ogid, ob in gts:
                if ogid != gid:
                    m = max(m, iou(gb, ob))
            overlap[(f, gid)] = m

    frames_of = defaultdict(list)
    for (f, gid, *_) in gt_rows:
        frames_of[gid].append(f)

    episodes = {"n": 0, "both_endpoints": 0, "retained": 0, "switched": 0, "one_or_no_endpoint": 0}
    frag = []
    edge_gaps, interior_gaps = 0, 0

    for gid, frs in frames_of.items():
        frs = sorted(set(frs))
        ids = {assign.get(f, {}).get(gid) for f in frs}
        ids.discard(None)
        if ids:
            frag.append(len(ids))

        # --- visibility gaps, split honestly into edge vs interior (reported separately) ---
        for i in range(len(frs) - 1):
            gap = frs[i + 1] - frs[i] - 1
            if 1 <= gap <= max_len:
                a, b = box_of.get((frs[i], gid)), box_of.get((frs[i + 1], gid))
                if a and b and interior(a) and interior(b):
                    interior_gaps += 1
                else:
                    edge_gaps += 1

        # --- on-screen occlusion episodes: occluded by ANOTHER player, entered/exited interior ---
        i = 0
        while i < len(frs):
            if overlap.get((frs[i], gid), 0.0) < occ_thr:
                i += 1
                continue
            j = i
            while j + 1 < len(frs) and overlap.get((frs[j + 1], gid), 0.0) >= occ_thr:
                j += 1
            entry_i, exit_i = i - 1, j + 1          # frames either side of the episode
            if entry_i >= 0 and exit_i < len(frs) and (frs[j] - frs[i] + 1) <= max_len:
                fe, fx = frs[entry_i], frs[exit_i]
                be, bx = box_of.get((fe, gid)), box_of.get((fx, gid))
                if be and bx and interior(be) and interior(bx):
                    episodes["n"] += 1
                    tid_e = assign.get(fe, {}).get(gid)
                    tid_x = assign.get(fx, {}).get(gid)
                    if tid_e is not None and tid_x is not None:
                        episodes["both_endpoints"] += 1
                        if tid_e == tid_x:
                            episodes["retained"] += 1
                        else:
                            episodes["switched"] += 1
                    else:
                        episodes["one_or_no_endpoint"] += 1
            i = j + 1

    return episodes, frag, edge_gaps, interior_gaps


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
    ap.add_argument("--occ-thr", type=float, default=0.3, help="IoU with another player that counts as occlusion")
    ap.add_argument("--iou-thr", type=float, default=0.5, help="GT<->track matching IoU")
    ap.add_argument("--edge-px", type=float, default=8.0, help="border margin; boxes nearer than this are 'edge'")
    ap.add_argument("--max-len", type=int, default=30, help="max episode/gap length in frames (~1s)")
    ap.add_argument("--out-dir", default="outputs/gsr_ft_dev")
    args = ap.parse_args()

    data = json.loads(Path(args.splits_file).read_text())
    seqs = data[args.split]
    if args.max_seqs:
        seqs = seqs[: args.max_seqs]
    classes = tuple(int(c) for c in args.classes.split(","))

    tot = {"n": 0, "both_endpoints": 0, "retained": 0, "switched": 0, "one_or_no_endpoint": 0}
    all_frag, edge_g, int_g = [], 0, 0
    for k, s in enumerate(seqs, 1):
        print(f"  [{k}/{len(seqs)}] {s['name']} ...", flush=True)
        tr_rows, _ = run_image_folder(Path(s["path"]) / "img1", weights=args.weights, classes=classes,
                                      conf=args.conf, imgsz=args.imgsz, tracker=args.tracker, device=args.device)
        gt_rows = gsr_to_mot_rows(s["labels"], keep_categories=PERSON_CATS, strict=True)
        ep, frag, eg, ig = analyze(gt_rows, tr_rows, frame_dims(s["labels"]),
                                   args.occ_thr, args.iou_thr, args.edge_px, args.max_len)
        for key in tot:
            tot[key] += ep[key]
        all_frag += frag
        edge_g += eg
        int_g += ig

    nf = len(all_frag)
    mean_frag = sum(all_frag) / nf if nf else 0.0
    single = sum(1 for x in all_frag if x == 1)
    cov = tot["both_endpoints"] / tot["n"] if tot["n"] else 0.0
    ret = tot["retained"] / tot["both_endpoints"] if tot["both_endpoints"] else None

    print(f"\n==== Rung-3 on-screen occlusion (dev persons, {Path(args.weights).name}) ====")
    print(f"  occlusion episodes (IoU>={args.occ_thr} with another player, interior entry+exit): {tot['n']}")
    print(f"    endpoint coverage (tracker saw the player both sides): {tot['both_endpoints']}/{tot['n']} = {cov:.3f}")
    print(f"    CONDITIONAL same-ID retention (of those): "
          + (f"{tot['retained']}/{tot['both_endpoints']} = {ret:.3f}" if ret is not None else "n/a")
          + f"   (switched {tot['switched']})")
    print(f"    detection-limited episodes (one/no endpoint detected): {tot['one_or_no_endpoint']}")
    print(f"  ID fragmentation: mean {mean_frag:.2f} tracker-IDs per GT person track-instance | "
          f"single-ID {single}/{nf} ({100*single/nf if nf else 0:.0f}%)")
    print(f"  [separately] visibility gaps — interior {int_g}, frame-edge {edge_g} "
          f"(edge gaps are exits/re-entries, NOT occlusions)")

    out = {
        "occlusion_episodes": {**tot, "endpoint_coverage": cov, "conditional_same_id_retention": ret},
        "fragmentation": {"mean_ids_per_gt": mean_frag, "single_id": single, "n_gt_track_instances": nf},
        "visibility_gaps": {"interior": int_g, "frame_edge": edge_g,
                            "note": "edge gaps are exits/re-entries, not occlusions"},
        "params": {"occ_thr": args.occ_thr, "iou_thr": args.iou_thr, "edge_px": args.edge_px,
                   "max_len": args.max_len},
        "provenance": {
            "git_sha": _git_sha(), "python": platform.python_version(), "platform": platform.platform(),
            "split": {"file": args.splits_file, "name": args.split, "n_seqs": len(seqs)},
            "detector": {"weights": args.weights, "sha256": _sha256(args.weights),
                         "imgsz": args.imgsz, "conf": args.conf, "classes": args.classes},
            "tracker": {"requested": args.tracker, "resolved": _resolve_tracker(args.tracker),
                        "sha256": _sha256(_resolve_tracker(args.tracker))},
            "args": vars(args),
        },
    }
    dst = Path(args.out_dir) / "occlusion_metrics.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=2))
    print(f"saved -> {dst}")


if __name__ == "__main__":
    main()
