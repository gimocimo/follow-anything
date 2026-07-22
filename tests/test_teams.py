"""Unit tests for team assignment — the occlusion-aware, confidence-weighted kit clustering.

These lock the two levers that address the A<->B flips seen in the demo: occlusion-aware sampling
(don't average in a torso an opponent is standing in front of) and confidence weighting (let crisp
detections outvote marginal ones), plus the per-track confidence/margin the demo surfaces. Pure
numpy/geometry — no model or data needed; the end-to-end accuracy vs GSR labels is measured
separately by scripts/15_team_eval.py.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pitchvision.demo import teams


def _s(a, b, n=5, w=0.9, occ=False):
    """n torso samples at chroma (a*, b*) with detection weight w and occlusion flag occ."""
    return [(np.array([128.0, float(a), float(b)], np.float32), w, occ) for _ in range(n)]


# ---------- geometry ----------

def test_torso_occlusion_fraction():
    box = (0, 0, 100, 200)                       # torso rect x25..75 (50) x y30..90 (60) = 3000
    assert teams.torso_occlusion(box, []) == 0.0
    assert teams.torso_occlusion(box, [(300, 300, 400, 400)]) == 0.0       # disjoint
    frac = teams.torso_occlusion(box, [(60, 0, 200, 200)])                 # covers x60..75 = 900
    assert abs(frac - 0.3) < 1e-6


def test_sample_frame_flags_occluded():
    frame = np.full((200, 300, 3), 128, np.uint8)
    dets = [(1, 0.9, (0, 0, 100, 200)), (2, 0.8, (60, 0, 160, 200))]       # overlapping torsos
    out = teams.sample_frame(frame, dets)
    assert len(out) == 2
    occ = {tid: o for tid, _, _, o in out}
    assert occ[1] and occ[2]                                              # each intrudes on the other


# ---------- clustering ----------

def test_two_teams_split_with_confidence():
    ts = {i: _s(210, 128) for i in range(4)}
    ts.update({i: _s(50, 128) for i in range(4, 8)})
    team, conf = assign_teams_ok(ts)
    a = {team[i] for i in range(4)}
    b = {team[i] for i in range(4, 8)}
    assert a in ({0}, {1}) and b in ({0}, {1}) and a != b                 # two distinct squads
    assert all(conf[i]["confidence"] > 0.3 for i in range(8))            # confident, well-separated


def test_referee_cluster_is_not_a_team():
    ts = {i: _s(210, 128) for i in range(5)}
    ts.update({i: _s(50, 128) for i in range(5, 10)})
    ts[10], ts[11] = _s(128, 210), _s(128, 205)                          # a distinct 3rd direction
    team, _ = teams.assign_teams(ts, k=3)
    assert team[10] is None and team[11] is None                        # smallest cluster != a team
    assert {team[i] for i in range(10)} == {0, 1}


def test_occlusion_preference_uses_clean_views():
    ts = {i: _s(210, 128) for i in range(4)}
    ts.update({i: _s(50, 128) for i in range(4, 8)})
    ts[8] = _s(210, 128, n=4, occ=False) + _s(50, 128, n=8, occ=True)     # clean A, occluded B
    team, conf = assign_teams_ok(ts)
    assert conf[8]["n_clean"] == 4 and conf[8]["occl_frac"] > 0.6
    assert team[8] == team[0]                                            # grouped with the clean-A tracks
    naive, _ = teams.assign_teams(ts, occlusion_aware=False, conf_weighted=False)
    assert naive[8] != naive[0]                                        # naive averages all -> NOT team A


def test_confidence_weighting_pulls_mean():
    hi = _s(210, 128, n=4, w=0.95)
    lo = _s(50, 128, n=4, w=0.05)
    mean_w, _ = teams._track_mean(hi + lo, 3, True, True)
    mean_u, _ = teams._track_mean(hi + lo, 3, True, False)
    assert mean_w[1] > mean_u[1] + 40                                   # weighted mean near the crisp a*=210


def test_low_margin_track_is_flagged():
    """A track sitting on the boundary between the two kits must report a low margin."""
    ts = {i: _s(210, 128) for i in range(4)}
    ts.update({i: _s(50, 128) for i in range(4, 8)})
    ts[8] = _s(130, 128)                                                # midway between the two kits
    _, conf = assign_teams_ok(ts)
    assert conf[8]["margin"] < 0.15                                     # ambiguous -> flagged in the demo
    assert conf[0]["margin"] > 0.5                                      # a clear member is not


def assign_teams_ok(ts):
    team, conf = teams.assign_teams(ts)
    assert team and conf, "expected a non-empty assignment"
    return team, conf
