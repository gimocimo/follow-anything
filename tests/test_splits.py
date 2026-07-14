"""Unit tests for the splitting invariants (the 'rigor centrepiece')."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pitchvision.data.splits import group_disjoint_split, leave_one_game_out


def _clips():
    # 3 games with the real GSR-train sizes (20 / 19 / 18 clips)
    items = []
    for g, n in (("9", 20), ("6", 19), ("4", 18)):
        items += [{"game": g, "id": f"{g}-{i}"} for i in range(n)]
    return items


def test_leave_one_game_out_sizes_and_disjoint():
    splits, assign = leave_one_game_out(_clips(), group_key=lambda s: s["game"])
    # deterministic: sorted games ['4','6','9'] -> test=4, val=6, train=9
    assert assign == {"4": "test", "6": "val", "9": "train"}
    assert (len(splits["test"]), len(splits["val"]), len(splits["train"])) == (18, 19, 20)
    seen = {}
    for name, items in splits.items():
        for s in items:
            assert seen.get(s["game"], name) == name  # no game in two splits
            seen[s["game"]] = name


def test_leave_one_game_out_needs_three_games():
    with pytest.raises(ValueError):
        leave_one_game_out([{"game": "a"}, {"game": "b"}], group_key=lambda s: s["game"])


def test_group_disjoint_fails_closed_on_empty_split():
    # 3 nearly-equal groups at 70/15/15 leaves test empty -> must RAISE now, not return 39/18/0
    with pytest.raises(ValueError):
        group_disjoint_split(_clips(), group_key=lambda s: s["game"], ratios=(0.7, 0.15, 0.15))
