"""Per-match team assignment from kit colour — and the ground truth to score it against.

Team identity is **match-specific**: "left"/"right" is not a visual property a detector can
learn, because both squads change kit every game. So team assignment is inherently a per-match
*clustering* problem. What a detector CAN learn is **role** (player / goalkeeper / referee),
which does generalise — and excluding officials from the clustering removes precisely the
outliers that make naive k-means split {both teams} vs {referee}.

SN-GSR-2025 annotates `attributes.team` (left/right) and `attributes.role`, so this module also
exposes the ground truth needed to *measure* assignment accuracy rather than eyeball it.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def _torso_rect(box):
    """The torso sub-rectangle sampled for kit colour (skips head, shorts and grass)."""
    x1, y1, x2, y2 = box
    h, w = y2 - y1, x2 - x1
    return (x1 + 0.25 * w, y1 + 0.15 * h, x1 + 0.75 * w, y1 + 0.45 * h)


def torso_color(frame, box, min_h=26):
    """Median Lab colour of the torso patch (skips head, shorts and grass)."""
    import cv2  # lazy: keep the pure clustering/geometry importable without OpenCV (e.g. in CI)
    x1, y1, x2, y2 = box
    if (y2 - y1) < min_h or (x2 - x1) < 8:
        return None
    tx1, ty1, tx2, ty2 = (int(v) for v in _torso_rect(box))
    patch = frame[max(0, ty1):max(0, ty2), max(0, tx1):max(0, tx2)]
    if patch.size == 0:
        return None
    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    return np.median(lab, axis=0)


def torso_occlusion(box, others):
    """Fraction of THIS box's torso rectangle covered by the nearest OTHER box — the reason a
    torso-colour sample gets contaminated by a different shirt. 0 = clear view, ~1 = fully
    overlapped by another player."""
    tx1, ty1, tx2, ty2 = _torso_rect(box)
    ta = max(0.0, tx2 - tx1) * max(0.0, ty2 - ty1)
    if ta <= 0:
        return 1.0
    worst = 0.0
    for o in others:
        inter = (max(0.0, min(tx2, o[2]) - max(tx1, o[0]))
                 * max(0.0, min(ty2, o[3]) - max(ty1, o[1])))
        worst = max(worst, inter)
    return worst / ta


def sample_frame(frame, dets, occ_thr=0.2):
    """Sample torso colours for one frame's PLAYER detections.

    `dets`: [(track_id, conf, box)]. Yields (track_id, lab, weight, occluded): `weight` is the
    detection confidence, and `occluded` flags a torso another player's box intrudes on (so its
    colour is the WRONG shirt). Sampling every player with its neighbours known is what lets the
    caller keep only clean views per track — the dominant fix for A<->B team flips."""
    boxes = [d[2] for d in dets]
    out = []
    for i, (tid, conf, box) in enumerate(dets):
        lab = torso_color(frame, box)
        if lab is None:
            continue
        occ = torso_occlusion(box, boxes[:i] + boxes[i + 1:])
        out.append((tid, lab, float(conf), occ >= occ_thr))
    return out


def _kmeans(X, init, iters=40):
    C = np.asarray(init, dtype=np.float32).copy()
    lab = np.zeros(len(X), dtype=int)
    for _ in range(iters):
        lab = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1).argmin(1)
        for c in range(len(C)):
            if (lab == c).any():
                C[c] = X[lab == c].mean(0)
    return lab, C, float(((X - C[lab]) ** 2).sum())


def best_kmeans(X, k, restarts=15, seed=0):
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(restarts):
        lab, C, inertia = _kmeans(X, X[rng.choice(len(X), k, replace=False)])
        if best is None or inertia < best[2]:
            best = (lab, C, inertia)
    return best[0], best[1]


def _track_mean(samples, min_samples, occlusion_aware, conf_weighted):
    """Per-track weighted-mean torso colour + provenance. Prefers un-occluded views, falling back
    to all views only when too few are clean."""
    clean = [(lab, w) for lab, w, occ in samples if not occ]
    used_clean = occlusion_aware and len(clean) >= min_samples
    used = clean if used_clean else [(lab, w) for lab, w, _ in samples]
    if len(used) < min_samples:
        return None
    labs = np.stack([lab for lab, _ in used])
    ws = (np.array([max(w, 1e-6) for _, w in used], np.float32) if conf_weighted
          else np.ones(len(used), np.float32))
    mean = (labs * ws[:, None]).sum(0) / ws.sum()
    return mean, {"n_clean": len(clean), "n_total": len(samples), "used_clean": used_clean}


def assign_teams(track_samples, min_samples=3, k=3, occlusion_aware=True, conf_weighted=True):
    """track_id -> (0 | 1 | None(other)), plus a per-track confidence dict.

    Clusters the PER-TRACK mean torso colour in chroma space (Lab a*/b* only — luminance mostly
    encodes lighting, which would split players by how sunlit they are). Uses k=3 and keeps the two
    largest clusters as the squads, so a referee (usually further from both teams than they are
    from each other) forms its own group instead of hijacking one.

    Two robustness levers, each measurable via scripts/15_team_eval.py against GSR team labels:
    **occlusion-aware** sampling keeps only frames where no other player intrudes on the torso (the
    dominant cause of A<->B flips — an overlapping opponent paints the wrong shirt into the patch),
    and **confidence-weighted** averaging lets crisp, high-confidence detections outvote marginal
    few-pixel ones. Pass both False to reproduce the naive baseline.

    `track_samples`: {track_id: [(lab, weight, occluded), ...]} as produced by `sample_frame`.
    Returns (team_of, conf_of). conf_of[tid] = {margin, confidence, n_clean, n_total, occl_frac}:
    `margin` in [-1, 1] is (d_other - d_self) / (d_other + d_self) to the two squad centroids (how
    much closer this track sits to its assigned kit than to the other), and `confidence` folds in
    how many clean samples backed it — so low-margin / thinly-observed tracks are flagged, not
    trusted.
    """
    means, meta = {}, {}
    for t, samples in track_samples.items():
        r = _track_mean(samples, min_samples, occlusion_aware, conf_weighted)
        if r is not None:
            means[t], meta[t] = r
    if len(means) < 6:
        return {}, {}
    tids = list(means)
    X = np.stack([means[t] for t in tids])[:, 1:].astype(np.float32)   # a*, b*
    k = min(k, len(X))
    lab, C = best_kmeans(X, k)
    sizes = [(lab == c).sum() for c in range(k)]
    teams = list(np.argsort(sizes)[::-1][:2])
    remap = {int(c): (0 if c == teams[0] else 1) for c in teams}
    cen = {0: C[teams[0]], 1: C[teams[1]]}
    team_of, conf_of = {}, {}
    for i, t in enumerate(tids):
        team = remap.get(int(lab[i]))
        if team in (0, 1):
            d_self = float(np.linalg.norm(X[i] - cen[team]))
            d_other = float(np.linalg.norm(X[i] - cen[1 - team]))
            margin = (d_other - d_self) / (d_other + d_self + 1e-6)
        else:
            margin = 0.0
        backing = meta[t]["n_clean"] if meta[t]["used_clean"] else 0.5 * meta[t]["n_total"]
        conf = max(0.0, margin) * min(1.0, backing / (2 * min_samples))
        team_of[t] = team
        conf_of[t] = {"margin": round(float(margin), 3), "confidence": round(float(conf), 3),
                      "n_clean": meta[t]["n_clean"], "n_total": meta[t]["n_total"],
                      "occl_frac": round(1 - meta[t]["n_clean"] / max(1, meta[t]["n_total"]), 3)}
    return team_of, conf_of


# ---------------- ground truth (SN-GSR-2025 attributes) ----------------

def gt_team_boxes(labels_json):
    """frame -> [(team, role, (x, y, w, h))] for annotated people."""
    data = json.loads(Path(labels_json).read_text())
    frame_of = {}
    for img in data.get("images", []):
        iid = img.get("image_id") or img.get("id")
        digits = "".join(ch for ch in Path(img.get("file_name", "")).stem if ch.isdigit())
        if iid is not None and digits:
            frame_of[iid] = int(digits)
    out = defaultdict(list)
    for a in data.get("annotations", []):
        at = a.get("attributes") or {}
        b = a.get("bbox_image") or {}
        fr = frame_of.get(a.get("image_id"))
        if fr is None or not b or at.get("role") not in ("player", "goalkeeper", "referee"):
            continue
        out[fr].append((at.get("team"), at.get("role"),
                        (float(b["x"]), float(b["y"]), float(b["w"]), float(b["h"]))))
    return out


def _iou(a, b):
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = a[2] * a[3] + b[2] * b[3] - inter
    return inter / ua if ua > 0 else 0.0


def true_team_of_tracks(track_boxes, gt_by_frame, iou_thr=0.5):
    """track_id -> (majority GT team, majority GT role) over the frames it was matched."""
    votes = defaultdict(Counter)
    roles = defaultdict(Counter)
    for tid, frames in track_boxes.items():
        for fr, box in frames:
            best, bt, br = iou_thr, None, None
            for team, role, gb in gt_by_frame.get(fr, []):
                v = _iou(box, gb)
                if v >= best:
                    best, bt, br = v, team, role
            if bt is not None:
                votes[tid][bt] += 1
            if br is not None:
                roles[tid][br] += 1
    return {tid: (votes[tid].most_common(1)[0][0] if votes[tid] else None,
                  roles[tid].most_common(1)[0][0] if roles[tid] else None)
            for tid in track_boxes}


def score_assignment(pred, truth, weights=None):
    """Cluster labels are arbitrary, so score both label permutations and keep the best.

    Only PLAYERS with a GT team are scored. Returns accuracy plus the counts behind it.
    """
    items = [(t, p, truth[t][0]) for t, p in pred.items()
             if t in truth and truth[t][0] in ("left", "right") and truth[t][1] == "player"]
    if not items:
        return {"accuracy": None, "n": 0}
    def acc(mapping):
        w = tot = 0.0
        for t, p, gt in items:
            k = (weights or {}).get(t, 1.0)
            tot += k
            if p is not None and mapping[p] == gt:
                w += k
        return w / tot if tot else 0.0
    a = acc({0: "left", 1: "right"})
    b = acc({0: "right", 1: "left"})
    unassigned = sum(1 for _, p, _ in items if p is None)
    return {"accuracy": max(a, b), "n": len(items), "unassigned": unassigned,
            "mapping": "0=left" if a >= b else "0=right"}
