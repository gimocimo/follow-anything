"""
Group-disjoint dataset splitting.

The whole point: a naive frame-level random split of *video* data leaks
near-duplicate frames — frame t and frame t+1 are almost identical — across
train and test, which massively inflates metrics. (This is exactly the trap
that made a prior microrobot project score ~99.75%: 100% of test poses were
seen in training and 24% of test frames were within 3 indices of a train
frame.) Here we split by GROUP (e.g. `match_id`) so no match ever appears in
more than one split.
"""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Callable, Hashable, Sequence, TypeVar

T = TypeVar("T")

SPLIT_NAMES = ("train", "val", "test")


def group_disjoint_split(
    items: Sequence[T],
    group_key: Callable[[T], Hashable],
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
) -> dict[str, list[T]]:
    """Partition `items` into train/val/test so each group lands entirely in one
    split.

    Greedy allocation: process groups largest-first and assign each to the split
    with the largest current deficit relative to its target share. Deterministic
    given `seed`. Verifies no leakage before returning.
    """
    assert abs(sum(ratios) - 1.0) < 1e-6, "ratios must sum to 1"

    groups: dict[Hashable, list[T]] = defaultdict(list)
    for it in items:
        groups[group_key(it)].append(it)

    # deterministic order: shuffle for tie-breaking, then stable sort by size desc
    ordered = list(groups.items())
    random.Random(seed).shuffle(ordered)
    ordered.sort(key=lambda kv: len(kv[1]), reverse=True)

    total = len(items)
    targets = {n: r * total for n, r in zip(SPLIT_NAMES, ratios)}
    counts = {n: 0 for n in SPLIT_NAMES}
    out: dict[str, list[T]] = {n: [] for n in SPLIT_NAMES}

    for _, members in ordered:
        name = max(SPLIT_NAMES, key=lambda n: targets[n] - counts[n])
        out[name].extend(members)
        counts[name] += len(members)

    empty = [n for n in SPLIT_NAMES if not out[n]]
    if empty:
        raise ValueError(
            f"group_disjoint_split produced empty split(s) {empty} from {len(groups)} "
            f"groups at ratios {ratios} — too few groups for ratio-based splitting. "
            f"Use leave_one_game_out() for a small, fixed group count.")
    assert_no_group_leakage(out, group_key)
    return out


def leave_one_game_out(
    items: Sequence[T], group_key: Callable[[T], Hashable]
) -> tuple[dict[str, list[T]], dict]:
    """Deterministic, explicit split for a small, fixed group count: with the
    groups sorted, test = the first, val = the second, train = the rest. No ratios
    or seeds. Requires >= 3 groups; guarantees non-empty train/val/test. Returns
    ``(splits, assignment)`` where ``assignment`` maps group -> split name."""
    groups: dict[Hashable, list[T]] = defaultdict(list)
    for it in items:
        groups[group_key(it)].append(it)
    keys = sorted(groups)
    if len(keys) < 3:
        raise ValueError(f"leave_one_game_out needs >= 3 groups; found {len(keys)}: {keys}")
    assign = {keys[0]: "test", keys[1]: "val", **{k: "train" for k in keys[2:]}}
    out: dict[str, list[T]] = {n: [] for n in SPLIT_NAMES}
    for g, members in groups.items():
        out[assign[g]].extend(members)
    if any(not out[n] for n in SPLIT_NAMES):
        raise ValueError(f"empty split from assignment {assign}")
    assert_no_group_leakage(out, group_key)
    return out, assign


def assert_no_group_leakage(
    splits: dict[str, list[T]], group_key: Callable[[T], Hashable]
) -> None:
    """Raise AssertionError if any group appears in more than one split."""
    seen: dict[Hashable, str] = {}
    for name, items in splits.items():
        for it in items:
            g = group_key(it)
            if g in seen and seen[g] != name:
                raise AssertionError(
                    f"Group leakage: {g!r} appears in both {seen[g]!r} and {name!r}"
                )
            seen[g] = name


def summarize_splits(splits: dict[str, list[T]]) -> str:
    total = sum(len(v) for v in splits.values()) or 1
    return " | ".join(
        f"{n}: {len(splits[n])} ({100 * len(splits[n]) / total:.1f}%)" for n in splits
    )
