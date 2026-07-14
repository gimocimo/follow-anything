"""Regression test: distinguish a genuinely persistent tracker from one that
recycles numeric IDs every frame (the Rung-1 per-frame-reset bug).

Recycled IDs look fine by *count* (the buggy path produced ~30 distinct ids over
750 frames — same order as the true object count), so counting can't catch it. We
test MOTION CONTINUITY instead: for the same id across consecutive frames, the box
centre must not teleport. A `_recycled_control` negative case proves the metric
actually flags the failure mode. Runs in CI (pure text parsing, no torch).
"""
import random
from pathlib import Path

GOLDEN = Path(__file__).parent / "data" / "golden_track_SNGS-060_f1-60.txt"
DISP_THRESH = 80.0  # px; players move a few px/frame in a 1920-wide broadcast shot


def _load(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        p = line.split(",")
        x, y, w, h = map(float, p[2:6])
        rows.append((int(p[0]), int(p[1]), x + w / 2, y + h / 2))
    return rows


def persistence_score(rows):
    """Fraction of consecutive same-id frame transitions whose centre moves < DISP_THRESH."""
    by_id = {}
    for fr, i, cx, cy in rows:
        by_id.setdefault(i, []).append((fr, cx, cy))
    good = total = 0
    for track in by_id.values():
        track.sort()
        for (f0, x0, y0), (f1, x1, y1) in zip(track, track[1:]):
            if f1 - f0 != 1:
                continue
            total += 1
            if ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5 < DISP_THRESH:
                good += 1
    return good / total if total else 0.0


def _recycled_control(n_frames=60, n_obj=20, w=1920, h=1080):
    """Synthetic 'recycled ids' MOT: each frame reassigns ids 1..n_obj to random
    positions (what the per-frame-reset bug produced). Deterministic PRNG."""
    rng = random.Random(0)
    rows = []
    for fr in range(1, n_frames + 1):
        for i in range(1, n_obj + 1):
            rows.append((fr, i, rng.uniform(0, w), rng.uniform(0, h)))
    return rows


def test_golden_tracker_persists():
    assert GOLDEN.exists(), f"missing golden file {GOLDEN}"
    score = persistence_score(_load(GOLDEN))
    assert score > 0.8, f"a genuine tracker should keep ids across frames; score={score:.3f}"


def test_recycled_ids_are_flagged():
    # the metric MUST catch the per-frame-reset failure mode, else it's useless
    score = persistence_score(_recycled_control())
    assert score < 0.5, f"recycled-id control should score low; got {score:.3f}"
