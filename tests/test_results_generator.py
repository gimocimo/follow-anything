"""Mutation tests for the results generator — it must FAIL CLOSED.

Adversarial review (2026-07) got all-green out of contradictory evidence: pre/post records with
unrelated hashes and the wrong games, a loader guard with `ok:true` but no counts, a missing
data.yaml SHA, and a FINAL manifest describing a different game / tracker / imgsz — because the
links only checked shallow booleans and `build()` hardcoded the final game and copied the tracker
from dev. Each mutation below now collapses the claim. A proof that reports success on
contradictory evidence is worse than no proof at all.
"""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("make_results", ROOT / "scripts" / "13_make_results.py")
mr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mr)

SHA_A, SHA_B, SHA_C, SHA_D, SHA_T = ("a" * 64, "b" * 64, "c" * 64, "d" * 64, "e" * 64)


def _prov(assignment):
    return {"detector": {"sha256": SHA_B, "imgsz": 1536, "classes": "0,1,2,3", "conf": 0.25},
            "tracker": {"sha256": SHA_T, "resolved": "configs/trackers/botsort_newtrk040.yaml"},
            "split": {"meta": {"game_assignment": assignment}},
            "packages": {}, "device": "cuda", "python": "3.12.3", "git_sha": "abc"}


def _artifacts():
    """A fully consistent evidence set: every leak-free link holds."""
    em = {"observed_games": ["3", "5", "6", "9"], "requested_games": ["3", "5", "6", "9"],
          "export_sha256": SHA_A}
    verify = {"ok": True, "export_sha256": SHA_A, "games": ["3", "5", "6", "9"], "n_expected": 45075}
    tm = {"dataset_export": {"export_sha256": SHA_A},
          "dataset_verified_pre": dict(verify), "dataset_verified_post": dict(verify),
          "data_yaml_problems": [], "data_yaml_sha256": SHA_D,
          "loader_guard": {"ok": True, "n_files": 45075, "train_matches_manifest": True,
                           "val_matches_manifest": True, "labels_authenticated": True},
          "output_best_sha256": SHA_B}
    dev = {"n_seqs": 18, "metrics": {"HOTA": 0.63},
           "provenance": _prov({"4": "test", "6": "val", "9": "train"})}
    fin = {"n_seqs": 18, "metrics": {"HOTA": 0.61},
           "provenance": _prov({"2": "test", "3": "val", "5": "train"})}
    return tm, em, dev, fin


def test_clean_evidence_passes():
    links, _ = mr.leak_free_links(*_artifacts())
    assert all(links.values()), links


# ---------- hashes fail closed ----------

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


def test_malformed_hash_both_sides_equal_but_invalid_fails_closed():
    """Codex: the old malformed-hash test mutated only ONE side, so 'both strings and equal' would
    still pass tests while dropping the 64-hex check. Both sides equal but not a hash must FAIL."""
    tm, em, dev, fin = _artifacts()
    tm["output_best_sha256"] = "not-a-hash"
    dev["provenance"]["detector"]["sha256"] = "not-a-hash"   # equal to the other side, still invalid
    links, _ = mr.leak_free_links(tm, em, dev, fin)
    assert links["train_manifest_binds_output_checkpoint"] is False


# ---------- pre/post verification must MATCH the export, not merely be ok ----------

def test_pre_verify_wrong_games_fails():
    tm, em, dev, fin = _artifacts()
    tm["dataset_verified_pre"]["games"] = ["4", "2"]     # ok:true, but the wrong games
    links, _ = mr.leak_free_links(tm, em, dev, fin)
    assert links["pre_verify_matches_export"] is False


def test_pre_verify_wrong_hash_fails():
    tm, em, dev, fin = _artifacts()
    tm["dataset_verified_pre"]["export_sha256"] = SHA_C  # ok:true, but an unrelated hash
    links, _ = mr.leak_free_links(tm, em, dev, fin)
    assert links["pre_verify_matches_export"] is False


def test_unverified_dataset_fails():
    tm, em, dev, fin = _artifacts()
    tm["dataset_verified_pre"]["ok"] = False
    links, _ = mr.leak_free_links(tm, em, dev, fin)
    assert links["pre_verify_matches_export"] is False


def test_inconsistent_verify_counts_fail():
    tm, em, dev, fin = _artifacts()
    tm["dataset_verified_post"]["n_expected"] = 40000     # post disagrees with pre
    links, _ = mr.leak_free_links(tm, em, dev, fin)
    assert links["verify_counts_consistent"] is False


# ---------- loader guard must prove SPLIT + LABELS + count ----------

def test_loader_guard_without_split_flags_fails():
    """A bare {ok:true} guard (Codex fed exactly this) must not certify the split."""
    tm, em, dev, fin = _artifacts()
    tm["loader_guard"] = {"ok": True}
    links, _ = mr.leak_free_links(tm, em, dev, fin)
    assert links["loader_split_authenticated"] is False
    assert links["loader_labels_authenticated"] is False


def test_loader_guard_wrong_count_fails():
    tm, em, dev, fin = _artifacts()
    tm["loader_guard"]["n_files"] = 12345                 # ok + flags true, but count disagrees
    links, _ = mr.leak_free_links(tm, em, dev, fin)
    assert links["loader_labels_authenticated"] is False


def test_absent_loader_guard_fails():
    tm, em, dev, fin = _artifacts()
    tm["loader_guard"] = None
    links, _ = mr.leak_free_links(tm, em, dev, fin)
    assert links["loader_split_authenticated"] is False


# ---------- data.yaml must be clean AND its SHA recorded ----------

def test_data_yaml_problem_fails():
    tm, em, dev, fin = _artifacts()
    tm["data_yaml_problems"] = ["train != canonical images/train"]
    links, _ = mr.leak_free_links(tm, em, dev, fin)
    assert links["data_yaml_authenticated"] is False


def test_missing_data_yaml_sha_fails():
    """Codex supplied no data_yaml_sha256; the checkpoint is then not bound to a yaml."""
    tm, em, dev, fin = _artifacts()
    tm["data_yaml_sha256"] = None
    links, _ = mr.leak_free_links(tm, em, dev, fin)
    assert links["data_yaml_authenticated"] is False


# ---------- game policy ----------

def test_stale_game_set_fails():
    tm, em, dev, fin = _artifacts()
    em["observed_games"] = em["requested_games"] = ["6", "9"]
    links, _ = mr.leak_free_links(tm, em, dev, fin)
    assert links["export_games_are_exactly_train_games"] is False


def test_contaminated_game_set_fails():
    tm, em, dev, fin = _artifacts()
    em["observed_games"] = em["requested_games"] = ["3", "4", "5", "6", "9"]
    links, _ = mr.leak_free_links(tm, em, dev, fin)
    assert links["eval_games_excluded_from_training"] is False
    assert not all(links.values())


# ---------- eval protocol: derived from each manifest, not hardcoded ----------

def test_final_manifest_wrong_game_fails():
    """Codex changed the final manifest to represent a different game; build() hardcoded '2' and
    passed. Now the final game is DERIVED and checked."""
    tm, em, dev, fin = _artifacts()
    fin["provenance"]["split"]["meta"]["game_assignment"] = {"9": "test", "3": "val", "5": "train"}
    links, _ = mr.leak_free_links(tm, em, dev, fin)
    assert links["final_scored_final_game"] is False


def test_final_manifest_swapped_tracker_fails():
    tm, em, dev, fin = _artifacts()
    fin["provenance"]["tracker"]["sha256"] = SHA_C
    links, _ = mr.leak_free_links(tm, em, dev, fin)
    assert links["dev_final_same_eval_protocol"] is False


def test_final_manifest_swapped_imgsz_fails():
    tm, em, dev, fin = _artifacts()
    fin["provenance"]["detector"]["imgsz"] = 640
    links, _ = mr.leak_free_links(tm, em, dev, fin)
    assert links["dev_final_same_eval_protocol"] is False


def test_dev_and_final_different_checkpoints_fail():
    tm, em, dev, fin = _artifacts()
    fin["provenance"]["detector"]["sha256"] = SHA_C
    links, _ = mr.leak_free_links(tm, em, dev, fin)
    assert links["dev_and_final_same_checkpoint"] is False


# ---------- end-to-end: a swapped final manifest flips the committed verdict ----------

def test_build_final_swap_flips_verdict():
    """The full generator (not just the links) must refuse to certify a swapped final manifest."""
    tm, em, dev, fin = _artifacts()
    fin["provenance"]["split"]["meta"]["game_assignment"] = {"9": "test", "3": "val", "5": "train"}
    out = mr.build(dev, fin, None, None, em, tm, 0.48, 0.49)
    assert out["leak_free"]["provably_leave_one_game_out"] is False


def test_build_clean_evidence_certifies():
    tm, em, dev, fin = _artifacts()
    out = mr.build(dev, fin, None, None, em, tm, 0.48, 0.49)
    assert out["leak_free"]["provably_leave_one_game_out"] is True
    assert out["dev"]["eval_game"] == "4" and out["final"]["eval_game"] == "2"


def test_gates_are_read_not_hardcoded(tmp_path):
    """Gate thresholds must come from results/rung1_baseline.json."""
    p = tmp_path / "rung1.json"
    p.write_text('{"dev":{"metrics":{"HOTA":0.111}},"final":{"metrics":{"HOTA":0.222}}}')
    assert mr.load_gates(p) == (0.111, 0.222)
