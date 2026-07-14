"""
Thin end-to-end slice (Rung 1): video -> detect (YOLO) -> track (ByteTrack /
BoT-SORT) -> annotated video + MOTChallenge-format results.

Uses COCO-pretrained YOLO (person + sports ball), so it runs with ZERO training
and is the project's first demo + baseline.
"""
from __future__ import annotations

from pathlib import Path

import cv2

from ..config import get_device


def _resolve_tracker(tracker: str) -> str:
    """Prefer the repo's vendored, version-pinned tracker config over the one
    bundled with ultralytics, so tracker behaviour is reproducible across
    ultralytics versions."""
    if tracker in ("botsort.yaml", "bytetrack.yaml"):
        vendored = Path(__file__).resolve().parents[3] / "configs" / "trackers" / Path(tracker).name
        if vendored.exists():
            return str(vendored)
    return tracker


def run_video(
    video: str,
    weights: str = "yolo11n.pt",
    classes=(0, 32),          # COCO: person=0, sports ball=32
    conf: float = 0.25,
    imgsz: int = 1280,
    tracker: str = "botsort.yaml",
    device: str = "auto",
    out_dir: str = "outputs",
) -> dict:
    from ultralytics import YOLO  # imported here so the package imports without it

    video = str(video)
    stem = Path(video).stem
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    out_video = out / f"{stem}_tracked.mp4"
    out_mot = out / f"{stem}_mot.txt"

    dev = get_device(device)
    model = YOLO(weights)

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()

    writer = None
    n_frames = 0

    with open(out_mot, "w") as mot:
        results = model.track(
            source=video,
            classes=list(classes),
            conf=conf,
            imgsz=imgsz,
            tracker=_resolve_tracker(tracker),
            persist=True,
            stream=True,
            device=dev,
            verbose=False,
        )
        for frame_idx, r in enumerate(results, start=1):
            frame = r.plot()  # BGR annotated frame
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(
                    str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
                )
            writer.write(frame)
            n_frames += 1

            b = r.boxes
            if b is not None and b.id is not None:
                xywh = b.xywh.cpu().numpy()
                ids = b.id.cpu().numpy()
                confs = b.conf.cpu().numpy()
                for (cx, cy, bw, bh), tid, cf in zip(xywh, ids, confs):
                    x, y = cx - bw / 2, cy - bh / 2
                    # MOTChallenge format: frame,id,x,y,w,h,conf,-1,-1,-1
                    # TODO(Phase 2): keep a separate ball track file (class split).
                    mot.write(
                        f"{frame_idx},{int(tid)},{x:.2f},{y:.2f},"
                        f"{bw:.2f},{bh:.2f},{cf:.4f},-1,-1,-1\n"
                    )

    if writer is not None:
        writer.release()

    return {
        "device": dev,
        "frames": n_frames,
        "video": str(out_video),
        "mot": str(out_mot),
    }


def run_image_folder(
    img_dir,
    weights: str = "yolo11n.pt",
    classes=(0, 32),
    conf: float = 0.25,
    imgsz: int = 1280,
    tracker: str = "botsort.yaml",
    device: str = "auto",
):
    """Track over an ordered image sequence (e.g. a SoccerNet `img1/` folder).

    Returns ``(rows, n_frames)`` where each row is
    ``(frame, id, x, y, w, h, conf)`` (1-indexed frames, top-left xywh).

    Uses ONE streaming ``model.track`` call over the whole folder so track IDs
    persist across the sequence. (Calling ``track`` per single frame resets the
    tracker even with ``persist=True`` — IDs get reused every frame and AssA
    collapses.) Frame numbers are read from each result's image filename.
    """
    from ultralytics import YOLO

    img_dir = Path(img_dir)
    frames = sorted(img_dir.glob("*.jpg")) or sorted(img_dir.glob("*.png"))
    if not frames:
        raise FileNotFoundError(f"no image frames (*.jpg/*.png) in {img_dir}")

    dev = get_device(device)
    model = YOLO(weights)
    rows = []
    results = model.track(
        source=str(img_dir),
        classes=list(classes),
        conf=conf,
        imgsz=imgsz,
        tracker=tracker,
        persist=True,
        stream=True,
        device=dev,
        verbose=False,
    )
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
        for (cx, cy, bw, bh), tid, cf in zip(xywh, ids, confs):
            rows.append((fr, int(tid), cx - bw / 2, cy - bh / 2, bw, bh, float(cf)))
    return rows, len(frames)
