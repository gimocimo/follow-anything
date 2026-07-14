#!/usr/bin/env python3
"""Self-test for the TrackEval harness: synthetic sequences with KNOWN, EXACT
metrics. Proves HOTA/DetA/AssA/MOTA/IDF1/IDSW are wired correctly before we trust
them on real data. Every metric in every case is asserted **exactly** (a prior
version only checked `< 0.999`, which a HOTA↔AssA swap would have passed).

Cases (2 objects, 20 frames, non-overlapping boxes — except `iou_partial`):
  - perfect     : predictions == GT                 -> all 1.0, IDSW 0
  - idswitch    : object 1's id flips at frame 11    -> HOTA √0.75, AssA 0.75, MOTA 0.975, IDSW 1
  - halfrecall  : object 2 dropped entirely          -> HOTA √0.5, DetA 0.5, MOTA 0.5, IDF1 2/3
  - iou_partial : 1 object, every pred box shifted so IoU == 2/3 -> matched at 13 of HOTA's
                  19 IoU thresholds (0.05..0.65) -> HOTA=DetA=AssA=13/19, MOTA 1.0

Runnable as a script or under pytest (`pytest scripts/02_eval_selftest.py`).
"""
import math
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pitchvision.eval.mot_eval import evaluate, write_gt, write_tracker

N = 20
TOL = 1e-4


def _gt_rows():
    rows = []
    for fr in range(1, N + 1):
        rows.append((fr, 1, 100 + fr * 2, 100, 50, 100))
        rows.append((fr, 2, 400, 100 + fr * 2, 50, 100))
    return rows


def _perfect():
    return [(fr, i, x, y, w, h, 0.9) for (fr, i, x, y, w, h) in _gt_rows()]


def _idswitch():
    out = []
    for (fr, i, x, y, w, h) in _gt_rows():
        if i == 1 and fr >= 11:
            i = 99
        out.append((fr, i, x, y, w, h, 0.9))
    return out


def _halfrecall():
    return [(fr, i, x, y, w, h, 0.9) for (fr, i, x, y, w, h) in _gt_rows() if i == 1]


# iou_partial: 10x10 boxes; shift x by 2 -> intersection 8*10=80, union 200-80=120, IoU=2/3.
# 2/3≈0.667 lies safely between HOTA thresholds 0.65 and 0.70 (no float-boundary ambiguity),
# so it is matched at 13 of the 19 thresholds (0.05..0.65) -> HOTA=DetA=AssA=13/19.
def _iou_gt():
    return [(fr, 1, 100, 100, 10, 10) for fr in range(1, N + 1)]


def _iou_pred():
    return [(fr, 1, 102, 100, 10, 10, 0.9) for fr in range(1, N + 1)]


CASES = {
    "perfect":     (_gt_rows(), _perfect()),
    "idswitch":    (_gt_rows(), _idswitch()),
    "halfrecall":  (_gt_rows(), _halfrecall()),
    "iou_partial": (_iou_gt(), _iou_pred()),
}

EXPECTED = {
    "perfect":     dict(HOTA=1.0,             DetA=1.0,     AssA=1.0,     MOTA=1.0,   IDF1=1.0,   IDSW=0),
    "idswitch":    dict(HOTA=math.sqrt(0.75), DetA=1.0,     AssA=0.75,    MOTA=0.975, IDF1=0.75,  IDSW=1),
    "halfrecall":  dict(HOTA=math.sqrt(0.5),  DetA=0.5,     AssA=1.0,     MOTA=0.5,   IDF1=2 / 3, IDSW=0),
    "iou_partial": dict(HOTA=13 / 19,         DetA=13 / 19, AssA=13 / 19, MOTA=1.0,   IDF1=1.0,   IDSW=0),
}


def _run(gt_rows, pred_rows):
    tmp = Path(tempfile.mkdtemp(prefix="pv_selftest_"))
    try:
        write_gt(tmp / "gt", "seq1", gt_rows)
        write_tracker(tmp / "trk", "t", "seq1", pred_rows)
        return evaluate(str(tmp / "gt"), str(tmp / "trk"), {"seq1": N}, trackers_to_eval=["t"])["t"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def check_case(name):
    m = _run(*CASES[name])
    for k, want in EXPECTED[name].items():
        got = m[k]
        if k == "IDSW":
            assert got == want, f"{name}.{k}: got {got}, want {want}"
        else:
            assert abs(got - want) < TOL, f"{name}.{k}: got {got:.6f}, want {want:.6f}"
    return m


def test_perfect():     check_case("perfect")
def test_idswitch():    check_case("idswitch")
def test_halfrecall():  check_case("halfrecall")
def test_iou_partial(): check_case("iou_partial")


def main():
    print(f"\n{'case':<13}{'HOTA':>9}{'DetA':>8}{'AssA':>8}{'MOTA':>8}{'IDF1':>8}{'IDSW':>6}")
    ok = True
    for name in CASES:
        try:
            m = check_case(name)
            print(f"{name:<13}{m['HOTA']:>9.4f}{m['DetA']:>8.4f}{m['AssA']:>8.4f}"
                  f"{m['MOTA']:>8.4f}{m['IDF1']:>8.4f}{m['IDSW']:>6d}  PASS")
        except AssertionError as e:
            ok = False
            print(f"{name:<13}  FAIL -> {e}")
    print("\nRESULT:", "ALL EXACT ✅" if ok else "FAILURES ❌")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
