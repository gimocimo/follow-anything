#!/usr/bin/env python3
"""Render a showcase demo video: every player tracked simultaneously, coloured by team.

Runs the SAME production path as the evaluation (one streaming `model.track` call, so IDs
persist), then renders a clean overlay: **team colour**, track-ID chip, and a header bar.

Team assignment is *predicted*, not taken from ground truth: we sample each player's torso
colour, average it **per track**, and k-means the per-track means into two kits. Averaging
per track (rather than per frame) is what makes it stable — a player briefly shadowed or
half-occluded still lands in the right team — and it is only possible because the tracker
gives persistent identities. Colours far from both kit centroids (goalkeepers, referees)
are drawn separately rather than forced into a team.

Defaults to a **held-out** clip (dev game 4 — never trained on), so the demo shows honest
generalisation rather than memorised training footage.

    python scripts/14_make_demo.py --clip-index 0 --max-frames 300
    python scripts/14_make_demo.py --clip-index 2 --out docs/demo_teams.mp4 --trail 12
"""
import argparse
import json
import sys
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pitchvision.config import get_device
from pitchvision.pipeline.run_video import _resolve_tracker

TEAM_BGR = [(64, 64, 232), (232, 150, 64)]      # kit A (red), kit B (blue)
OTHER_BGR = (60, 220, 240)                       # goalkeeper / referee / uncertain
BALL_BGR = (0, 240, 255)


def torso_color(frame, box, min_h=26):
    """Median Lab colour of the torso patch (skips head, shorts, grass)."""
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
    k = len(C)
    lab = np.zeros(len(X), dtype=int)
    for _ in range(iters):
        lab = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1).argmin(1)
        for c in range(k):
            if (lab == c).any():
                C[c] = X[lab == c].mean(0)
    return lab, C, float(((X - C[lab]) ** 2).sum())


def _best_kmeans(X, k, restarts=15, seed=0):
    """k-means with multiple random seedings, keeping the lowest-inertia solution."""
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(restarts):
        init = X[rng.choice(len(X), k, replace=False)]
        lab, C, inertia = _kmeans(X, init)
        if best is None or inertia < best[2]:
            best = (lab, C, inertia)
    return best[0], best[1]


def assign_teams(track_colors, min_samples=3, k=3):
    """track_id -> 0 | 1 | None(other=GK/referee).

    Clusters PER-TRACK mean torso colour in CHROMA space (Lab a*/b* only — luminance mostly
    encodes lighting/shadow and would split players by how sunlit they are, not by kit).

    Uses **k=3, then takes the two largest clusters as the teams**. k=2 is the wrong model
    for football: a referee's kit is often *further* from both teams than the teams are from
    each other, so 2-means correctly-but-uselessly splits {both teams} vs {referee}. Three
    groups (team, team, officials) matches reality; anything outside the two biggest is drawn
    as "other".
    """
    means = {t: np.mean(v, axis=0) for t, v in track_colors.items() if len(v) >= min_samples}
    if len(means) < 6:
        return {}
    tids = list(means)
    X = np.stack([means[t] for t in tids])[:, 1:].astype(np.float32)   # a*, b* only
    k = min(k, len(X))
    lab, C = _best_kmeans(X, k)
    sizes = [(lab == c).sum() for c in range(k)]
    teams = list(np.argsort(sizes)[::-1][:2])          # two largest clusters = the two squads
    remap = {int(c): (0 if c == teams[0] else 1) for c in teams}
    return {t: remap.get(int(lab[i])) for i, t in enumerate(tids)}


def draw_header(img, title, sub):
    w = img.shape[1]
    bar = img.copy()
    cv2.rectangle(bar, (0, 0), (w, 66), (18, 18, 18), -1)
    cv2.addWeighted(bar, 0.72, img, 0.28, 0, img)
    cv2.putText(img, title, (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, sub, (18, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (185, 185, 185), 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", default=None, help="path to a GSR sequence dir (contains img1/)")
    ap.add_argument("--splits-file", default="outputs/gsr_splits.json")
    ap.add_argument("--split", default="test", help="split to pull --clip-index from (test = held-out)")
    ap.add_argument("--clip-index", type=int, default=0)
    ap.add_argument("--weights", default="models/yolo11m_gsr_ft.pt")
    ap.add_argument("--tracker", default="configs/trackers/botsort_newtrk040.yaml")
    ap.add_argument("--classes", default="0,1")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-frames", type=int, default=300, help="~12 s at 25 fps")
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--trail", type=int, default=0, help="motion-trail length in frames (0 = off)")
    ap.add_argument("--scale", type=float, default=1.0, help="output scale, e.g. 0.667 for 1280-wide")
    ap.add_argument("--no-teams", action="store_true", help="colour by track id instead of team")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    clip = Path(args.clip) if args.clip else None
    if clip is None:
        data = json.loads(Path(args.splits_file).read_text())
        clip = Path(data[args.split][args.clip_index]["path"])
    img_dir = clip / "img1"
    out_path = Path(args.out) if args.out else Path("outputs/demos") / f"{clip.name}_tracked.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- pass 1: track once, cache detections + per-track torso colours ----
    from ultralytics import YOLO
    model = YOLO(args.weights)
    results = model.track(
        source=str(img_dir), classes=[int(c) for c in args.classes.split(",")],
        conf=args.conf, imgsz=args.imgsz, tracker=_resolve_tracker(args.tracker),
        persist=True, stream=True, device=get_device(args.device), verbose=False,
    )
    cached, track_colors, n = [], defaultdict(list), 0
    for r in results:
        if n >= args.max_frames:
            break
        b = r.boxes
        dets = []
        if b is not None and b.id is not None:
            frame = cv2.imread(r.path)
            xyxy = b.xyxy.cpu().numpy()
            ids = b.id.cpu().numpy().astype(int)
            cls = b.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), tid, c in zip(xyxy, ids, cls):
                box = (int(x1), int(y1), int(x2), int(y2))
                dets.append((int(tid), int(c), box))
                if c == 0 and frame is not None:
                    col = torso_color(frame, box)
                    if col is not None:
                        track_colors[int(tid)].append(col)
        cached.append((r.path, dets))
        n += 1
        if n % 100 == 0:
            print(f"  tracked {n} frames ...", flush=True)

    team_of = {} if args.no_teams else assign_teams(track_colors)
    n_a = sum(1 for v in team_of.values() if v == 0)
    n_b = sum(1 for v in team_of.values() if v == 1)
    print(f"  teams from kit colour: A={n_a} tracks, B={n_b}, other/GK/ref={sum(1 for v in team_of.values() if v is None)}")

    # ---- pass 2: render (no inference; just re-read + draw) ----
    writer, trails = None, defaultdict(lambda: deque(maxlen=max(1, args.trail)))
    for path, dets in cached:
        frame = cv2.imread(path)
        if frame is None:
            continue
        live = 0
        for tid, c, (x1, y1, x2, y2) in dets:
            if c == 1:
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.circle(frame, (cx, cy), 13, BALL_BGR, 2, cv2.LINE_AA)
                cv2.putText(frame, "ball", (cx + 16, cy - 8), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, BALL_BGR, 1, cv2.LINE_AA)
                continue
            live += 1
            t = team_of.get(tid, None)
            col = TEAM_BGR[t] if t in (0, 1) else OTHER_BGR
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2, cv2.LINE_AA)
            lab = str(tid)
            (tw, th), _ = cv2.getTextSize(lab, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 7), (x1 + tw + 8, y1), col, -1)
            cv2.putText(frame, lab, (x1 + 4, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (20, 20, 20), 1, cv2.LINE_AA)
            if args.trail:
                trails[tid].append(((x1 + x2) // 2, y2))
                pts = list(trails[tid])
                for i in range(1, len(pts)):
                    cv2.line(frame, pts[i - 1], pts[i], col, max(1, int(3 * i / len(pts))), cv2.LINE_AA)

        sub = ("fine-tuned YOLO11m + BoT-SORT  |  teams auto-assigned from kit colour  |  "
               f"{clip.name} — held-out match (never trained on)")
        draw_header(frame, f"follow-anything  |  {live} players tracked simultaneously", sub)
        if args.scale != 1.0:
            frame = cv2.resize(frame, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_AREA)
        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
        writer.write(frame)

    if writer is not None:
        writer.release()
    print(f"\nwrote {out_path}  ({len(cached)} frames, {len(cached)/args.fps:.1f}s)")


if __name__ == "__main__":
    main()
