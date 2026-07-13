#!/usr/bin/env python3
"""Self-test for the TrackEval harness using synthetic sequences with KNOWN
metrics. Proves HOTA / MOTA / IDF1 are wired correctly before we trust them on
real football data.

Ground truth: 2 objects, 20 frames, non-overlapping boxes.
  - perfect     : predictions identical to GT      -> HOTA=MOTA=IDF1=1, IDSW=0
  - idswitch    : object 1's id flips at frame 11  -> IDSW=1, MOTA=1-1/40=0.975
  - halfrecall  : object 2 dropped entirely        -> MOTA=1-20/40=0.5, IDF1=2/3
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pitchvision.eval.mot_eval import evaluate, write_gt, write_tracker

N = 20


def gt_rows():
    rows = []
    for fr in range(1, N + 1):
        rows.append((fr, 1, 100 + fr * 2, 100, 50, 100))  # object 1
        rows.append((fr, 2, 400, 100 + fr * 2, 50, 100))  # object 2
    return rows


def perfect_rows():
    return [(fr, i, x, y, w, h, 0.9) for (fr, i, x, y, w, h) in gt_rows()]


def idswitch_rows():
    out = []
    for (fr, i, x, y, w, h) in gt_rows():
        if i == 1 and fr >= 11:
            i = 99
        out.append((fr, i, x, y, w, h, 0.9))
    return out


def halfrecall_rows():
    return [(fr, i, x, y, w, h, 0.9) for (fr, i, x, y, w, h) in gt_rows() if i == 1]


def main():
    tmp = tempfile.mkdtemp(prefix="pv_eval_")
    gtf, trf = Path(tmp) / "gt", Path(tmp) / "trackers"
    write_gt(gtf, "seq1", gt_rows())
    write_tracker(trf, "perfect", "seq1", perfect_rows())
    write_tracker(trf, "idswitch", "seq1", idswitch_rows())
    write_tracker(trf, "halfrecall", "seq1", halfrecall_rows())

    res = evaluate(str(gtf), str(trf), {"seq1": N},
                   trackers_to_eval=["perfect", "idswitch", "halfrecall"])

    print(f"\n{'tracker':<12}{'HOTA':>8}{'MOTA':>8}{'IDF1':>8}{'IDSW':>6}")
    for t in ("perfect", "idswitch", "halfrecall"):
        m = res[t]
        print(f"{t:<12}{m['HOTA']:8.4f}{m['MOTA']:8.4f}{m['IDF1']:8.4f}{m['IDSW']:6d}")

    ok = True

    def check(name, cond):
        nonlocal ok
        ok = ok and cond
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    p, s, h = res["perfect"], res["idswitch"], res["halfrecall"]
    print()
    check("perfect  HOTA == 1", abs(p["HOTA"] - 1) < 1e-6)
    check("perfect  MOTA == 1", abs(p["MOTA"] - 1) < 1e-6)
    check("perfect  IDF1 == 1", abs(p["IDF1"] - 1) < 1e-6)
    check("perfect  IDSW == 0", p["IDSW"] == 0)
    check("idswitch IDSW == 1", s["IDSW"] == 1)
    check("idswitch MOTA == 0.975", abs(s["MOTA"] - 0.975) < 1e-6)
    check("idswitch IDF1 < 1", s["IDF1"] < 0.999)
    check("idswitch HOTA < 1", s["HOTA"] < 0.999)
    check("halfrec  MOTA == 0.5", abs(h["MOTA"] - 0.5) < 1e-6)
    check("halfrec  IDF1 == 2/3", abs(h["IDF1"] - 2 / 3) < 1e-3)

    print("\nRESULT:", "ALL PASS ✅" if ok else "FAILURES ❌")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
