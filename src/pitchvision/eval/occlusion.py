"""Occlusion analysis primitives (Rung 3) — separated from the CLI so they can be unit-tested.

Our own §3 invariant says metrics are validated to known values *before* being trusted. Two
occlusion statistics were published before that happened, and both were wrong in ways an audit
had to catch:

  * "≈4% occlusion recovery" — 25 of its 27 "gaps" were frame-edge exits, not occlusions.
  * "91% keeps identity through occlusion" — an *endpoint* comparison that silently dropped the
    122 longest (hardest) episodes, ignored identity breaks *inside* the episode, and counted
    each mutual overlap twice.

This module therefore reports four distinct quantities and never collapses them into one
headline:

  endpoint_coverage      — was the player detected on both sides of the episode at all?
  endpoint_same_id       — of those, was the ID the same before and after? (recovery)
  detection_continuity   — was the player detected in EVERY frame of the episode?
  id_stability           — detected throughout AND never changed ID (the strict measure)

Results are stratified by episode duration; nothing is excluded. The unit is an
**episode of one person being overlapped**, so a mutual overlap contributes two episodes.
"""
from __future__ import annotations

from collections import defaultdict

BUCKETS = (("1-10", 1, 10), ("11-30", 11, 30), ("31+", 31, 10**9))


def iou(a, b):
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = a[2] * a[3] + b[2] * b[3] - inter
    return inter / ua if ua > 0 else 0.0


def match_per_frame(gt_rows, tr_rows, iou_thr=0.5):
    """frame -> {gt_id: track_id}, greedy highest-IoU one-to-one."""
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
                ug.add(gid), ut.add(tid)
    return assign, gt_by_fr


def find_episodes(gt_rows, dims, occ_thr=0.3, edge_px=8.0):
    """Episodes of one GT person being overlapped by ANOTHER, entered and exited interior.

    Returns a list of {gid, frames, length}: `frames` spans entry .. exit inclusive (the
    GT-present frames), so the endpoints are the last clean frame before and first clean
    frame after the overlap.
    """
    _, gt_by_fr = match_per_frame(gt_rows, [], iou_thr=2.0)   # only need the GT index
    W, H = (dims or (None, None))

    def interior(b):
        if W is None:
            return True
        return b[0] > edge_px and b[1] > edge_px and (b[0] + b[2]) < (W - edge_px) and (b[1] + b[3]) < (H - edge_px)

    box_of, overlap = {}, {}
    for f, gts in gt_by_fr.items():
        for gid, gb in gts:
            box_of[(f, gid)] = gb
            overlap[(f, gid)] = max((iou(gb, ob) for ogid, ob in gts if ogid != gid), default=0.0)

    frames_of = defaultdict(list)
    for (f, gid, *_) in gt_rows:
        frames_of[gid].append(f)

    episodes = []
    for gid, frs in frames_of.items():
        frs = sorted(set(frs))
        i = 0
        while i < len(frs):
            if overlap.get((frs[i], gid), 0.0) < occ_thr:
                i += 1
                continue
            j = i
            while j + 1 < len(frs) and overlap.get((frs[j + 1], gid), 0.0) >= occ_thr:
                j += 1
            if i - 1 >= 0 and j + 1 < len(frs):
                fe, fx = frs[i - 1], frs[j + 1]
                be, bx = box_of.get((fe, gid)), box_of.get((fx, gid))
                if be and bx and interior(be) and interior(bx):
                    episodes.append({"gid": gid, "frames": frs[i - 1:j + 2],
                                     "length": frs[j] - frs[i] + 1})
            i = j + 1
    return episodes


def score_episodes(episodes, assign):
    """Per-episode outcomes — four independent quantities, never merged."""
    out = []
    for ep in episodes:
        ids = [assign.get(f, {}).get(ep["gid"]) for f in ep["frames"]]
        both = ids[0] is not None and ids[-1] is not None
        out.append({
            "length": ep["length"],
            "both_endpoints": both,
            "endpoint_same_id": bool(both and ids[0] == ids[-1]),
            "detected_throughout": all(v is not None for v in ids),
            "id_stable_throughout": all(v is not None for v in ids) and len(set(ids)) == 1,
        })
    return out


def summarise(scored):
    """Aggregate overall and per duration bucket. Rates are None when the denominator is 0."""
    def agg(rows):
        n = len(rows)
        both = sum(r["both_endpoints"] for r in rows)
        return {
            "episodes": n,
            "endpoint_coverage": (both / n) if n else None,
            "endpoint_same_id": (sum(r["endpoint_same_id"] for r in rows) / both) if both else None,
            "detection_continuity": (sum(r["detected_throughout"] for r in rows) / n) if n else None,
            "id_stability": (sum(r["id_stable_throughout"] for r in rows) / n) if n else None,
        }
    return {"overall": agg(scored),
            "by_duration": {name: agg([r for r in scored if lo <= r["length"] <= hi])
                            for name, lo, hi in BUCKETS}}
