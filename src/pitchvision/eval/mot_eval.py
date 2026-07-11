"""
Tracking evaluation (HOTA / MOTA / IDF1).

Plan (Phase 0): vendor TrackEval (https://github.com/JonathonLuiten/TrackEval)
and score predictions in MOTChallenge format against ground truth.

HOTA is the *primary* metric: it balances detection quality and association
quality, which is exactly what a permanence-focused tracker must be judged on
(MOTA over-weights detection; IDF1 over-weights association).
"""
from __future__ import annotations


def evaluate_mot(gt_dir: str, results_dir: str, metrics=("HOTA", "MOTA", "IDF1")) -> dict:
    """Evaluate MOTChallenge-format results against ground truth.

    Deliberately raises until TrackEval is wired up — we never report fake
    numbers. Integration lands in Phase 0.
    """
    raise NotImplementedError(
        "TrackEval integration lands in Phase 0. "
        "See https://github.com/JonathonLuiten/TrackEval"
    )
