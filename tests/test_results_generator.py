"""Mutation tests for the results generator — it must FAIL CLOSED.

Adversarial review found two fail-open bugs here: absent hashes compared `None == None` and
were reported as a *satisfied* link, and the final verdict trusted a flag stored inside the
very manifest being audited. Each test mutates the evidence and asserts the claim collapses.
A proof artifact that reports success on missing evidence is worse than no artifact at all.
"""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("make_results", ROOT / "scripts" / "13_make_results.py")
mr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mr)

SHA_A, SHA_B, SHA_C = "a" * 64, "b" * 64, "c" * 64


def _artifacts():
    em = {"observed_games": ["3", "5", "6", "9"], "requested_games": ["3", "5", "6", "9"],
          "export_sha256": SHA_A}
    tm = {"dataset_export": {"export_sha256": SHA_A},
          "dataset_verified_pre": {"ok": True}, "dataset_verified_post": {"ok": True},
          "data_yaml_problems": [], "loader_guard": {"ok": True},
          "output_best_sha256": SHA_B}
    dev = {"provenance": {"detector": {"sha256": SHA_B}}}
    fin = {"provenance": {"detector": {"sha256": SHA_B}}}
    return tm, em, dev, fin


def test_clean_evidence_passes():
    links, _ = mr.leak_free_links(*_artifacts())
    assert all(links.values()), links


def test_missing_export_hash_fails_closed():
    """`None == None` must never read as 'linked'."""
    tm, em, dev, fin = _artifacts()
    em["export_sha256"] = None
    tm["dataset_export"]["export_sha256"] = None
    links, _ = mr.leak_free_links(tm, em, dev, fin)
    assert links["train_manifest_links_export"] is False


def test_missing_checkpoint_hashes_fail_closed():
    tm, em, dev, fin = _artifacts()
    tm["output_best_sha256"] = None
    dev["provenance"]["detector"]["sha256"] = None
    fin["provenance"]["detector"]["sha256"] = None
    links, _ = mr.leak_free_links(tm, em, dev, fin)
    assert links["train_manifest_binds_output_checkpoint"] is False
    assert links["dev_and_final_same_checkpoint"] is False


def test_malformed_hash_fails_closed():
    tm, em, dev, fin = _artifacts()
    tm["output_best_sha256"] = "not-a-hash"
    links, _ = mr.leak_free_links(tm, em, dev, fin)
    assert links["train_manifest_binds_output_checkpoint"] is False


def test_stale_game_set_fails():
    """An export that does not match the intended recipe (e.g. the old {6,9}) must not pass —
    otherwise a superseded dataset could quietly back a current claim."""
    tm, em, dev, fin = _artifacts()
    em["observed_games"] = ["6", "9"]
    em["requested_games"] = ["6", "9"]
    links, _ = mr.leak_free_links(tm, em, dev, fin)
    assert links["export_games_are_exactly_train_games"] is False


def test_contaminated_game_set_fails():
    """An export containing the dev eval game can never be 'provably leave-one-game-out'."""
    tm, em, dev, fin = _artifacts()
    em["observed_games"] = ["3", "4", "5", "6", "9"]
    em["requested_games"] = ["3", "4", "5", "6", "9"]
    links, _ = mr.leak_free_links(tm, em, dev, fin)
    assert links["eval_games_excluded_from_training"] is False
    assert not all(links.values())


def test_unverified_dataset_fails():
    tm, em, dev, fin = _artifacts()
    tm["dataset_verified_pre"] = {"ok": False}
    links, _ = mr.leak_free_links(tm, em, dev, fin)
    assert links["dataset_verified_before_training"] is False


def test_absent_loader_guard_fails():
    """A checkpoint trained before the loader guard existed must not claim the strong result."""
    tm, em, dev, fin = _artifacts()
    tm["loader_guard"] = None
    links, _ = mr.leak_free_links(tm, em, dev, fin)
    assert links["loader_resolved_exactly_manifest"] is False


def test_data_yaml_problem_fails():
    tm, em, dev, fin = _artifacts()
    tm["data_yaml_problems"] = ["train=/elsewhere escapes the manifested dir"]
    links, _ = mr.leak_free_links(tm, em, dev, fin)
    assert links["data_yaml_authenticated"] is False


def test_dev_and_final_different_checkpoints_fail():
    tm, em, dev, fin = _artifacts()
    fin["provenance"]["detector"]["sha256"] = SHA_C
    links, _ = mr.leak_free_links(tm, em, dev, fin)
    assert links["dev_and_final_same_checkpoint"] is False


def test_gates_are_read_not_hardcoded(tmp_path):
    """Gate thresholds must come from results/rung1_baseline.json."""
    p = tmp_path / "rung1.json"
    p.write_text('{"dev":{"metrics":{"HOTA":0.111}},"final":{"metrics":{"HOTA":0.222}}}')
    assert mr.load_gates(p) == (0.111, 0.222)
