"""
Read SoccerNet Game-State-Reconstruction (SN-GSR-2025) labels and convert them
to MOTChallenge tracking rows.

Each sequence directory contains:
    img1/000001.jpg ...
    Labels-GameState.json

Labels-GameState.json is a COCO-like dict:
    images:      [{image_id, file_name, height, width, ...}]
    annotations: [{image_id, track_id, category_id, bbox_image:{x,y,w,h,...}, ...}]
    categories:  [{id, name}]   # 1 player, 2 goalkeeper, 3 referee, 4 ball, 5 pitch, 6 camera, 7 other

CATEGORY POLICY (decided 2026-07-13, logged in PROJECT_PLAN §7): the tracking
baseline includes EVERY annotation that has a track_id AND an image bbox — i.e.
categories 1-4 (player / GK / referee / ball) AND category 7 ("other" people such
as staff). Non-object rows (pitch, camera) carry no track_id/bbox and are excluded.
This keeps one class-agnostic score matching our un-classed tracker output; use
`gsr_category_counts` for per-class diagnostics, or `keep_categories={1,2,3,4}` to
restrict. The published baseline (HOTA 0.481) is over ALL object categories.

We score standard (bbox-IoU) HOTA. SoccerNet's official GSR metric is GS-HOTA over
*pitch* coordinates — the flagship (rung 6) target, not this bbox baseline.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional

_SPLITS = ("train", "valid", "test", "challenge")


def _frame_index(img: dict) -> Optional[int]:
    fn = img.get("file_name") or img.get("filename") or ""
    digits = "".join(c for c in Path(fn).stem if c.isdigit())
    return int(digits) if digits else None


def _bbox_xywh(ann: dict):
    b = ann.get("bbox_image") or ann.get("bbox")
    if isinstance(b, dict):
        if all(k in b for k in ("x", "y", "w", "h")):
            return float(b["x"]), float(b["y"]), float(b["w"]), float(b["h"])
        if all(k in b for k in ("x_center", "y_center", "w", "h")):
            w, h = float(b["w"]), float(b["h"])
            return float(b["x_center"]) - w / 2, float(b["y_center"]) - h / 2, w, h
    elif isinstance(b, (list, tuple)) and len(b) >= 4:
        return float(b[0]), float(b[1]), float(b[2]), float(b[3])
    return None


def gsr_to_mot_rows(labels_json, keep_categories: Optional[Iterable[int]] = None,
                    strict: bool = False) -> list:
    """Convert a Labels-GameState.json to MOT rows [(frame, id, x, y, w, h)].

    Includes every annotation with a track id and an image bbox. By default ALL
    object categories are kept (see CATEGORY POLICY above); pass
    `keep_categories={1,2,3,4}` to restrict to player/GK/referee/ball.

    With `strict=True` (used for gate-producing GT) it fails closed instead of
    silently skipping: an object annotation (track_id + bbox) whose image_id has
    no frame, a non-finite / non-positive box, or a duplicate (frame, id) raises.
    """
    import math

    data = json.loads(Path(labels_json).read_text())
    frame_of = {}
    for img in data.get("images", []):
        iid = img.get("image_id") or img.get("id")
        fr = _frame_index(img)
        if iid is not None and fr is not None:
            frame_of[iid] = fr

    keep = set(keep_categories) if keep_categories is not None else None
    rows, seen = [], set()
    for ann in data.get("annotations", []):
        if keep is not None and ann.get("category_id") not in keep:
            continue
        tid = ann.get("track_id")
        box = _bbox_xywh(ann)
        if tid is None or box is None:
            continue  # not a boxed + tracked object row
        fr = frame_of.get(ann.get("image_id"))
        if fr is None:
            if strict:
                raise ValueError(f"{labels_json}: ann {ann.get('id')} (track {tid}) "
                                 f"has no frame for image_id {ann.get('image_id')}")
            continue
        if strict:
            if not all(map(math.isfinite, box)) or box[2] <= 0 or box[3] <= 0:
                raise ValueError(f"{labels_json}: invalid box {box} (frame {fr}, id {tid})")
            key = (fr, int(tid))
            if key in seen:
                raise ValueError(f"{labels_json}: duplicate (frame,id)={key}")
            seen.add(key)
        rows.append((fr, int(tid), *box))
    rows.sort(key=lambda r: (r[0], r[1]))
    return rows


def gsr_category_counts(labels_json) -> dict:
    """Count boxed + tracked annotations per category name (per-class diagnostics)."""
    data = json.loads(Path(labels_json).read_text())
    id_to_name = {c["id"]: c.get("name", str(c["id"])) for c in data.get("categories", [])}
    counts: Counter = Counter()
    for ann in data.get("annotations", []):
        if ann.get("track_id") is not None and _bbox_xywh(ann) is not None:
            counts[ann.get("category_id")] += 1
    return {id_to_name.get(cid, str(cid)): n
            for cid, n in sorted(counts.items(), key=lambda x: (x[0] if x[0] is not None else 0))}


def _read_game_id(labels_json: Path) -> Optional[str]:
    try:
        info = json.loads(labels_json.read_text()).get("info", {}) or {}
    except Exception:
        return None
    gid = info.get("game_id")
    return str(gid) if gid not in (None, "") else None


def index_gsr_sequences(root, split: Optional[str] = None, require_game_id: bool = False) -> list:
    """Find GSR sequences (directories containing Labels-GameState.json).

    Each record carries `game_id` (from `info.game_id`, else None) and `match_id`
    (= game_id, falling back to the sequence name only when game_id is absent).
    With `require_game_id=True`, raises if any in-scope sequence lacks a game_id —
    so a gate-producing split can never silently claim game-disjointness on a
    fallback grouping.
    """
    root = Path(root)
    seqs, missing = [], []
    for lbl in sorted(root.glob("**/Labels-GameState.json")):
        d = lbl.parent
        sp = next((p for p in d.parts if p in _SPLITS), "unknown")
        if split and sp != split:
            continue
        game_id = _read_game_id(lbl)
        if game_id is None:
            missing.append(str(lbl))
        n_frames = len(list((d / "img1").glob("*.jpg")))
        seqs.append({
            "name": d.name, "split": sp, "path": str(d), "labels": str(lbl),
            "length": n_frames, "game_id": game_id, "match_id": game_id or d.name,
        })
    if require_game_id and missing:
        raise ValueError(
            f"{len(missing)} sequence(s) missing info.game_id — cannot guarantee a game-disjoint "
            f"split. Offenders:\n  " + "\n  ".join(missing[:20]))
    return seqs
