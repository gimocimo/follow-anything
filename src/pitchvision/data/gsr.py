"""
Read SoccerNet Game-State-Reconstruction (SN-GSR-2025) labels and convert them
to MOTChallenge tracking rows.

Each sequence directory contains:
    img1/000001.jpg ...
    Labels-GameState.json

Labels-GameState.json is a COCO-like dict:
    images:      [{image_id, file_name, height, width, ...}]
    annotations: [{image_id, track_id, category_id, bbox_image:{x,y,w,h,...}, ...}]
    categories:  [{id, name}]   # player, goalkeeper, referee, ball, ...

For a *tracking* baseline we use the IMAGE bboxes + track ids and score standard
(bbox-IoU) HOTA. NB: SoccerNet's official GSR metric is GS-HOTA over *pitch*
coordinates — that's the flagship (rung 6) target, not this bbox baseline.

These parsers are defensive about exact key names; verify against a real file
once the data is downloaded (see scripts/05_inspect_gsr.py).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

_SPLITS = ("train", "valid", "test", "challenge")


def _frame_index(img: dict) -> Optional[int]:
    fn = img.get("file_name") or img.get("filename") or ""
    digits = "".join(ch for ch in Path(fn).stem if ch.isdigit())
    return int(digits) if digits else None


def _bbox_xywh(ann: dict):
    """Return top-left (x, y, w, h) from a GSR annotation, or None."""
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


def gsr_to_mot_rows(labels_json: Union[str, Path]) -> list:
    """Convert a Labels-GameState.json to MOT rows [(frame, id, x, y, w, h)].

    Includes every annotation with a track id and an image bbox (players,
    goalkeepers, referees, ball — lumped, matching our un-classed tracker).
    """
    data = json.loads(Path(labels_json).read_text())
    frame_of = {}
    for img in data.get("images", []):
        iid = img.get("image_id") or img.get("id")
        fr = _frame_index(img)
        if iid is not None and fr is not None:
            frame_of[iid] = fr

    rows = []
    for ann in data.get("annotations", []):
        fr = frame_of.get(ann.get("image_id"))
        tid = ann.get("track_id")
        box = _bbox_xywh(ann)
        if fr is None or tid is None or box is None:
            continue
        rows.append((fr, int(tid), *box))
    rows.sort(key=lambda r: (r[0], r[1]))
    return rows


def index_gsr_sequences(root: Union[str, Path], split: Optional[str] = None) -> list:
    """Find GSR sequences (directories containing Labels-GameState.json)."""
    root = Path(root)
    seqs = []
    for lbl in sorted(root.glob("**/Labels-GameState.json")):
        d = lbl.parent
        sp = next((p for p in d.parts if p in _SPLITS), "unknown")
        if split and sp != split:
            continue
        try:
            info = json.loads(lbl.read_text()).get("info", {}) or {}
        except Exception:
            info = {}
        match_id = str(info.get("game_id") or info.get("game")
                       or info.get("clip") or info.get("id") or d.name)
        n_frames = len(list((d / "img1").glob("*.jpg")))
        seqs.append({
            "name": d.name, "split": sp, "path": str(d), "labels": str(lbl),
            "length": n_frames, "match_id": match_id,
        })
    return seqs
