"""
Tracking evaluation via TrackEval (HOTA / MOTA / IDF1).

We drive TrackEval's MotChallenge2DBox evaluator in a "flat folder" mode
(`SKIP_SPLIT_FOL=True` + `SEQ_INFO=...`) so no seqmaps / seqinfo.ini files are
needed. Ground truth uses the MOTChallenge gt format with class=1 (pedestrian);
tracker files use the standard `frame,id,x,y,w,h,conf,-1,-1,-1` rows that
`run_video` already emits, so predictions slot straight in.

HOTA is the primary metric — it balances detection and association quality,
which is exactly what a permanence-focused tracker must be judged on.

Folder layout expected by `evaluate`:
    <gt_folder>/<seq>/gt/gt.txt
    <trackers_folder>/<tracker>/data/<seq>.txt
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence, Union

import numpy as np

Row = Sequence[float]


def _restore_numpy_aliases() -> None:
    """TrackEval (even current HEAD) still uses ``np.float`` / ``np.int``, which
    NumPy 2 removed. They were only aliases for the Python builtins, so we
    restore them as such — this lets TrackEval run unmodified under NumPy 2
    without changing any numerical behaviour."""
    # Only float/int are needed by TrackEval; the other removed aliases
    # (object/str/bool) would trigger their own NumPy FutureWarnings if touched.
    for name, builtin in (("float", float), ("int", int)):
        if not hasattr(np, name):
            setattr(np, name, builtin)


_restore_numpy_aliases()


def write_gt(gt_folder: Union[str, Path], seq: str, rows: Iterable[Row]) -> Path:
    """Write ground-truth rows (frame,id,x,y,w,h) in MOTChallenge gt format.

    conf/class/visibility are set to 1 (class 1 == pedestrian, the generic
    person class SoccerNet / SportsMOT also use with TrackEval).
    """
    out = Path(gt_folder) / seq / "gt"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "gt.txt"
    with open(path, "w") as f:
        for fr, i, x, y, w, h in rows:
            if fr != int(fr) or i != int(i):
                raise ValueError(f"non-integral frame/id in GT: ({fr}, {i})")
            f.write(f"{int(fr)},{int(i)},{x},{y},{w},{h},1,1,1\n")
    return path


def write_tracker(
    trackers_folder: Union[str, Path], tracker: str, seq: str, rows: Iterable[Row]
) -> Path:
    """Write predictions (frame,id,x,y,w,h[,conf]) in MOTChallenge format."""
    out = Path(trackers_folder) / tracker / "data"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{seq}.txt"
    with open(path, "w") as f:
        for r in rows:
            fr, i, x, y, w, h = r[:6]
            if fr != int(fr) or i != int(i):
                raise ValueError(f"non-integral frame/id in predictions: ({fr}, {i})")
            conf = r[6] if len(r) > 6 else 1.0
            f.write(f"{int(fr)},{int(i)},{x},{y},{w},{h},{conf},-1,-1,-1\n")
    return path


def evaluate(
    gt_folder: Union[str, Path],
    trackers_folder: Union[str, Path],
    seq_lengths: Mapping[str, int],
    trackers_to_eval: Optional[Iterable[str]] = None,
    do_preproc: bool = False,
) -> dict:
    """Run TrackEval and return {tracker: {HOTA, DetA, AssA, MOTA, IDF1, IDSW}}."""
    import trackeval

    eval_config = {
        **trackeval.Evaluator.get_default_eval_config(),
        "USE_PARALLEL": False,
        "LOG_ON_ERROR": None,  # don't write a fixed error_log.txt (read-only dirs mask the real error)
        "PRINT_CONFIG": False,
        "PRINT_RESULTS": False,
        "TIME_PROGRESS": False,
        "DISPLAY_LESS_PROGRESS": True,
        "OUTPUT_SUMMARY": False,
        "OUTPUT_DETAILED": False,
        "PLOT_CURVES": False,
    }
    dataset_config = {
        **trackeval.datasets.MotChallenge2DBox.get_default_dataset_config(),
        "GT_FOLDER": str(gt_folder),
        "TRACKERS_FOLDER": str(trackers_folder),
        "TRACKERS_TO_EVAL": list(trackers_to_eval) if trackers_to_eval else None,
        "CLASSES_TO_EVAL": ["pedestrian"],
        "BENCHMARK": "pitchvision",
        "SPLIT_TO_EVAL": "all",
        "SKIP_SPLIT_FOL": True,
        "DO_PREPROC": do_preproc,
        "PRINT_CONFIG": False,
        "SEQ_INFO": {k: int(v) for k, v in seq_lengths.items()},
    }

    evaluator = trackeval.Evaluator(eval_config)
    dataset = trackeval.datasets.MotChallenge2DBox(dataset_config)
    metrics = [
        trackeval.metrics.HOTA({"PRINT_CONFIG": False}),
        trackeval.metrics.CLEAR({"PRINT_CONFIG": False}),
        trackeval.metrics.Identity({"PRINT_CONFIG": False}),
    ]

    output_res, _ = evaluator.evaluate([dataset], metrics)
    per_tracker = output_res["MotChallenge2DBox"]

    results = {}
    for tracker, res in per_tracker.items():
        c = res["COMBINED_SEQ"]["pedestrian"]
        results[tracker] = {
            "HOTA": float(np.mean(c["HOTA"]["HOTA"])),
            "DetA": float(np.mean(c["HOTA"]["DetA"])),
            "AssA": float(np.mean(c["HOTA"]["AssA"])),
            "MOTA": float(c["CLEAR"]["MOTA"]),
            "IDF1": float(c["Identity"]["IDF1"]),
            "IDSW": int(c["CLEAR"]["IDSW"]),
        }
    return results
