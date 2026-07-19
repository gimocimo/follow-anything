"""Unit tests for the occlusion metric — validated to KNOWN values on synthetic scenes.

Two occlusion statistics were published before this existed and both were wrong. §3 of the
plan requires metrics to be unit-tested before they are trusted; these are those tests.

Scene construction: person 1 sits still; person 2 slides in, overlaps for a few frames, then
leaves. That creates one *episode per person* (a mutual overlap counts twice — a property we
assert explicitly, because the earlier headline silently double-counted it).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pitchvision.eval.occlusion import find_episodes, match_per_frame, score_episodes, summarise

DIMS = (1000.0, 1000.0)
P1 = (100.0, 100.0, 20.0, 50.0)     # stationary
NEAR = (105.0, 100.0, 20.0, 50.0)   # IoU 0.6 with P1  -> counts as occlusion
FAR = (500.0, 100.0, 20.0, 50.0)    # IoU 0            -> clean


def _gt(overlap_frames, total=7, p1=P1):
    rows = []
    for f in range(1, total + 1):
        rows.append((f, 1, *p1))
        rows.append((f, 2, *(NEAR if f in overlap_frames else FAR)))
    return rows


def _tr(id_by_frame, total=7, p1=P1):
    """Tracker rows for person 1 only; id_by_frame maps frame -> track id (None = missed)."""
    rows = []
    for f in range(1, total + 1):
        tid = id_by_frame.get(f, 101)
        if tid is not None:
            rows.append((f, tid, *p1, 0.9))
    return rows


def _score(gt, tr):
    eps = find_episodes(gt, DIMS)
    assign, _ = match_per_frame(gt, tr, iou_thr=0.5)
    return eps, score_episodes([e for e in eps if e["gid"] == 1], assign)


def test_mutual_overlap_yields_one_episode_per_person():
    """A single mutual overlap is TWO episodes (one per person) — the unit is per-person."""
    eps = find_episodes(_gt({3, 4, 5}), DIMS)
    assert len(eps) == 2
    assert {e["gid"] for e in eps} == {1, 2}
    assert all(e["length"] == 3 for e in eps)          # overlap spans frames 3,4,5


def test_perfect_tracking_scores_all_ones():
    _, scored = _score(_gt({3, 4, 5}), _tr({}))
    s = summarise(scored)["overall"]
    assert s == {"episodes": 1, "endpoint_coverage": 1.0, "endpoint_same_id": 1.0,
                 "detection_continuity": 1.0, "id_stability": 1.0}


def test_internal_id_break_is_hidden_by_endpoints_but_caught_by_stability():
    """The exact flaw in the withdrawn headline: ID breaks mid-episode and returns, so an
    endpoint comparison calls it a success while identity was in fact lost."""
    _, scored = _score(_gt({3, 4, 5}), _tr({4: 999}))
    s = summarise(scored)["overall"]
    assert s["endpoint_same_id"] == 1.0        # endpoints agree ...
    assert s["id_stability"] == 0.0            # ... but identity was NOT held throughout
    assert s["detection_continuity"] == 1.0


def test_permanent_id_switch_fails_endpoints_too():
    _, scored = _score(_gt({3, 4, 5}), _tr({4: 999, 5: 999, 6: 999}))
    s = summarise(scored)["overall"]
    assert s["endpoint_same_id"] == 0.0
    assert s["id_stability"] == 0.0


def test_missing_endpoint_is_a_detection_failure_not_an_identity_failure():
    """A player the tracker never saw on exit must not be counted as an identity error —
    conflating the two is what produced the withdrawn '≈4%' figure."""
    _, scored = _score(_gt({3, 4, 5}), _tr({6: None}))
    s = summarise(scored)["overall"]
    assert s["endpoint_coverage"] == 0.0
    assert s["endpoint_same_id"] is None       # no denominator -> no rate, not a zero
    assert s["detection_continuity"] == 0.0


def test_interior_gap_lowers_detection_continuity_only():
    _, scored = _score(_gt({3, 4, 5}), _tr({4: None}))
    s = summarise(scored)["overall"]
    assert s["endpoint_coverage"] == 1.0 and s["endpoint_same_id"] == 1.0
    assert s["detection_continuity"] == 0.0
    assert s["id_stability"] == 0.0


def test_edge_pinned_player_is_excluded():
    """A player pinned against the frame border is an exit case, not an occlusion, so their
    episode is dropped. The *other* player enters and leaves from clean interior positions,
    so theirs legitimately counts — exclusion is per-person, not per-encounter."""
    edge = (2.0, 100.0, 20.0, 50.0)            # hard against the left border
    gt = []
    for f in range(1, 8):
        gt.append((f, 1, *edge))
        gt.append((f, 2, *((7.0, 100.0, 20.0, 50.0) if f in {3, 4, 5} else FAR)))
    eps = find_episodes(gt, DIMS)
    assert all(e["gid"] != 1 for e in eps), "edge-pinned player's episode must be excluded"
    assert [e["gid"] for e in eps] == [2]


def test_duration_stratification_keeps_long_episodes():
    """Long overlaps must be reported, not silently dropped (the earlier metric excluded
    every episode >30 frames — precisely the hardest ones)."""
    long_overlap = set(range(3, 44))            # 41 frames
    eps = find_episodes(_gt(long_overlap, total=45), DIMS)
    mine = [e for e in eps if e["gid"] == 1]
    assert mine and mine[0]["length"] == 41
    assign, _ = match_per_frame(_gt(long_overlap, total=45), _tr({}, total=45), iou_thr=0.5)
    buckets = summarise(score_episodes(mine, assign))["by_duration"]
    assert buckets["31+"]["episodes"] == 1
    assert buckets["1-10"]["episodes"] == 0
