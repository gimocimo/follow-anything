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

import cv2
import numpy as np


def torso_color(frame, box, min_h=26):
    """Median Lab colour of the torso patch (skips head, shorts and grass)."""
    x1, y1, x2, y2 = box
    h, w = y2 - y1, x2 - x1
    if h < min_h or w < 8:
        return None
    ty1, ty2 = int(y1 + 0.15 * h), int(y1 + 0.45 * h)
    tx1, tx2 = int(x1 + 0.25 * w), int(x1 + 0.75 * w)
    patch = frame[max(0, ty1):max(0, ty2), max(0, tx1):max(0, tx2)]
    if patch.size == 0:
        return None
    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    return np.median(lab, axis=0)


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


def assign_teams(track_colors, min_samples=3, k=3, weights=None):
    """track_id -> 0 | 1 | None(other).

    Clusters the PER-TRACK mean torso colour in chroma space (Lab a*/b* only — luminance
    mostly encodes lighting, which would split players by how sunlit they are). Uses k=3 and
    keeps the two largest clusters as the squads, so a referee (often further from both teams
    than they are from each other) forms its own group instead of hijacking one.

    `weights` optionally maps track_id -> confidence weight (e.g. mean box area), so distant,
    few-pixel tracks influence the centroids less than close, well-observed ones.
    """
    means = {t: np.mean(v, axis=0) for t, v in track_colors.items() if len(v) >= min_samples}
    if len(means) < 6:
        return {}
    tids = list(means)
    X = np.stack([means[t] for t in tids])[:, 1:].astype(np.float32)   # a*, b*
    k = min(k, len(X))

    # NB: fitting centroids only on well-observed tracks was tried and MEASURED — it left k=2
    # unchanged (0.916) and degraded k=3 (0.916 -> 0.857), so it is deliberately not used.
    # The colour heuristic appears to be near its ceiling; the principled fix is to exclude
    # officials using the detector's predicted role (see the 4-class training recipe).
    lab, C = best_kmeans(X, k)
    if weights:                       # weight reliable tracks more when refining centroids
        for c in range(k):
            m = lab == c
            if m.any():
                w = np.array([max(weights.get(tids[i], 1.0), 1e-6) for i in np.where(m)[0]], dtype=np.float32)
                C[c] = (X[m] * w[:, None]).sum(0) / w.sum()
        lab = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1).argmin(1)
    sizes = [(lab == c).sum() for c in range(k)]
    teams = list(np.argsort(sizes)[::-1][:2])
    remap = {int(c): (0 if c == teams[0] else 1) for c in teams}
    return {t: remap.get(int(lab[i])) for i, t in enumerate(tids)}


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
