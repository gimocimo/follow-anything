"""Mutation tests for the results generator — it must FAIL CLOSED.

Two adversarial rounds got all-green out of bad evidence. Round 4: contradictory manifests (the
links tested shallow `.ok` booleans, and `build()` hardcoded the final game). Round 5: evidence that
was mutually *consistent* but wrong — counts of 1 against a 45,075-file export, `ok:true` beside
`modified:['x']`, a non-empty loader `problems` list, a 1-sequence final evaluation, `conf` 0.99,
and both evaluations moved to imgsz 640 (an equality check between dev and final cannot notice).

The fix, exercised below: anchor on GROUND TRUTH — the committed per-file inventory, whose
fingerprint must reproduce `export_sha256` — and pin the canonical evaluation recipe rather than
comparing the two evaluations to each other.
"""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("make_results", ROOT / "scripts" / "13_make_results.py")
mr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mr)

SHA_B, SHA_C, SHA_D = "b" * 64, "c" * 64, "d" * 64

# Ground truth: games {5,6,9} in train, game 3 held out as cross-match val.
INV = ([{"subset": "train", "stem": f"t{i}", "game_id": g, "img_sha256": "x", "label_sha256": "y"}
        for i, g in enumerate(("5", "6", "9"))]
       + [{"subset": "val", "stem": "v0", "game_id": "3", "img_sha256": "x", "label_sha256": "y"}])


def _prov(assignment, source_split, role):
    return {"detector": {"sha256": SHA_B, "imgsz": 1536, "classes": "0,1,2,3", "conf": 0.25},
            "tracker": {"sha256": mr.TRACKER_SHA, "resolved": "configs/trackers/botsort_newtrk040.yaml"},
            "split": {"meta": {"game_assignment": assignment, "source_split": source_split,
                               "role": role, "policy": "leave-one-game-out"}},
            "packages": {}, "device": "cuda", "python": "3.12.3", "git_sha": "abc"}


def _artifacts():
    """A fully consistent evidence set anchored on INV: every leak-free link holds."""
    esha = mr.inventory_fingerprint(INV)
    em = {"observed_games": ["3", "5", "6", "9"], "requested_games": ["3", "5", "6", "9"],
          "export_sha256": esha}
    verify = {"ok": True, "fingerprint_match": True, "missing": [], "modified": [], "extra": [],
              "export_sha256": esha, "games": ["3", "5", "6", "9"], "n_expected": len(INV)}
    tm = {"dataset_export": {"export_sha256": esha, "observed_games": ["3", "5", "6", "9"],
                             "requested_games": ["3", "5", "6", "9"]},
          "dataset_verified_pre": dict(verify), "dataset_verified_post": dict(verify),
          "data_yaml_problems": [], "data_yaml_sha256": SHA_D,
          "loader_guard": {"ok": True, "n_files": 4, "n_train": 3, "n_val": 1, "n_labels_checked": 4,
                           "train_matches_manifest": True, "val_matches_manifest": True,
                           "labels_authenticated": True, "in_memory_labels_authenticated": True,
                           "problems": []},
          "output_best_sha256": SHA_B}
    dev = {"n_seqs": 18, "subset": False, "metrics": {"HOTA": 0.63},
           "provenance": _prov({"4": "test", "6": "val", "9": "train"}, "train", "development")}
    fin = {"n_seqs": 18, "subset": False, "metrics": {"HOTA": 0.61},
           "provenance": _prov({"2": "test", "3": "val", "5": "train"}, "valid", "final")}
    return tm, em, dev, fin, [dict(e) for e in INV]


def _links(tm, em, dev, fin, inv):
    return mr.leak_free_links(tm, em, dev, fin, inv)[0]


def test_clean_evidence_passes():
    links = _links(*_artifacts())
    assert all(links.values()), links


# ---------- ground truth: the inventory itself ----------

def test_missing_inventory_fails_closed():
    tm, em, dev, fin, _ = _artifacts()
    links = _links(tm, em, dev, fin, None)
    assert links["inventory_backs_export_sha256"] is False
    assert links["verify_counts_match_inventory"] is False


def test_tampered_inventory_breaks_the_fingerprint():
    tm, em, dev, fin, inv = _artifacts()
    inv[0]["label_sha256"] = "tampered"
    links = _links(tm, em, dev, fin, inv)
    assert links["inventory_backs_export_sha256"] is False


def test_counts_consistent_but_wrong_fail():
    """Round 5: pre/post/guard all said 1 against a 45,075-frame export and every link passed."""
    tm, em, dev, fin, inv = _artifacts()
    tm["dataset_verified_pre"]["n_expected"] = 1
    tm["dataset_verified_post"]["n_expected"] = 1
    tm["loader_guard"].update({"n_files": 1, "n_train": 1, "n_val": 0, "n_labels_checked": 1})
    links = _links(tm, em, dev, fin, inv)
    assert links["verify_counts_match_inventory"] is False
    assert links["loader_counts_match_inventory"] is False


# ---------- verification records must be internally clean ----------

def test_ok_true_beside_dirty_fields_fails():
    """Round 5: `fingerprint_match:false` + `modified:['x']` alongside `ok:true` still passed."""
    tm, em, dev, fin, inv = _artifacts()
    tm["dataset_verified_pre"].update({"fingerprint_match": False, "modified": ["x"]})
    links = _links(tm, em, dev, fin, inv)
    assert links["pre_verify_clean_and_matches_export"] is False


def test_pre_verify_wrong_games_fails():
    tm, em, dev, fin, inv = _artifacts()
    tm["dataset_verified_pre"]["games"] = ["4", "2"]
    assert _links(tm, em, dev, fin, inv)["pre_verify_clean_and_matches_export"] is False


def test_unverified_dataset_fails():
    tm, em, dev, fin, inv = _artifacts()
    tm["dataset_verified_pre"]["ok"] = False
    assert _links(tm, em, dev, fin, inv)["pre_verify_clean_and_matches_export"] is False


def test_embedded_export_games_must_match_standalone():
    """Round 5: `train_manifest.dataset_export.observed_games = ['4']` passed unnoticed."""
    tm, em, dev, fin, inv = _artifacts()
    tm["dataset_export"]["observed_games"] = ["4"]
    assert _links(tm, em, dev, fin, inv)["embedded_export_matches_standalone"] is False


# ---------- loader guard ----------

def test_loader_problems_non_empty_fails():
    """Round 5: `loader_guard.problems=['train contamination']` and every link stayed green."""
    tm, em, dev, fin, inv = _artifacts()
    tm["loader_guard"]["problems"] = ["train contamination"]
    assert _links(tm, em, dev, fin, inv)["loader_split_authenticated"] is False


def test_in_memory_labels_unauthenticated_fails():
    """The poisoned-cache defence must be reflected in the certified result."""
    tm, em, dev, fin, inv = _artifacts()
    tm["loader_guard"]["in_memory_labels_authenticated"] = False
    assert _links(tm, em, dev, fin, inv)["loader_labels_authenticated"] is False


def test_loader_guard_without_split_flags_fails():
    tm, em, dev, fin, inv = _artifacts()
    tm["loader_guard"] = {"ok": True}
    links = _links(tm, em, dev, fin, inv)
    assert links["loader_split_authenticated"] is False
    assert links["loader_labels_authenticated"] is False


def test_absent_loader_guard_fails():
    tm, em, dev, fin, inv = _artifacts()
    tm["loader_guard"] = None
    assert _links(tm, em, dev, fin, inv)["loader_split_authenticated"] is False


# ---------- hashes fail closed ----------

def test_missing_checkpoint_hashes_fail_closed():
    tm, em, dev, fin, inv = _artifacts()
    tm["output_best_sha256"] = None
    dev["provenance"]["detector"]["sha256"] = None
    fin["provenance"]["detector"]["sha256"] = None
    links = _links(tm, em, dev, fin, inv)
    assert links["train_manifest_binds_output_checkpoint"] is False
    assert links["dev_and_final_same_checkpoint"] is False


def test_malformed_hash_both_sides_equal_but_invalid_fails_closed():
    """The old test mutated only ONE side, so weakening `_same_hash` to 'both strings and equal'
    kept the suite green. Both sides equal but not a hash must FAIL."""
    tm, em, dev, fin, inv = _artifacts()
    tm["output_best_sha256"] = "not-a-hash"
    dev["provenance"]["detector"]["sha256"] = "not-a-hash"
    assert _links(tm, em, dev, fin, inv)["train_manifest_binds_output_checkpoint"] is False


def test_dev_and_final_different_checkpoints_fail():
    tm, em, dev, fin, inv = _artifacts()
    fin["provenance"]["detector"]["sha256"] = SHA_C
    assert _links(tm, em, dev, fin, inv)["dev_and_final_same_checkpoint"] is False


# ---------- data.yaml ----------

def test_data_yaml_problem_fails():
    tm, em, dev, fin, inv = _artifacts()
    tm["data_yaml_problems"] = ["train != canonical images/train"]
    assert _links(tm, em, dev, fin, inv)["data_yaml_authenticated"] is False


def test_missing_data_yaml_sha_fails():
    tm, em, dev, fin, inv = _artifacts()
    tm["data_yaml_sha256"] = None
    assert _links(tm, em, dev, fin, inv)["data_yaml_authenticated"] is False


# ---------- game policy ----------

def test_stale_game_set_fails():
    tm, em, dev, fin, inv = _artifacts()
    em["observed_games"] = em["requested_games"] = ["6", "9"]
    assert _links(tm, em, dev, fin, inv)["export_games_are_exactly_train_games"] is False


def test_contaminated_game_set_fails():
    tm, em, dev, fin, inv = _artifacts()
    em["observed_games"] = em["requested_games"] = ["3", "4", "5", "6", "9"]
    links = _links(tm, em, dev, fin, inv)
    assert links["eval_games_excluded_from_training"] is False
    assert not all(links.values())


# ---------- eval protocol pinned to the canonical recipe ----------

def test_final_manifest_wrong_game_fails():
    tm, em, dev, fin, inv = _artifacts()
    fin["provenance"]["split"]["meta"]["game_assignment"] = {"9": "test", "3": "val", "5": "train"}
    assert _links(tm, em, dev, fin, inv)["final_matches_canonical_protocol"] is False


def test_partial_final_evaluation_fails():
    """Round 5: scoring 1 of 18 sequences still certified."""
    tm, em, dev, fin, inv = _artifacts()
    fin["n_seqs"] = 1
    assert _links(tm, em, dev, fin, inv)["final_matches_canonical_protocol"] is False


def test_subset_evaluation_fails():
    tm, em, dev, fin, inv = _artifacts()
    fin["subset"] = True
    assert _links(tm, em, dev, fin, inv)["final_matches_canonical_protocol"] is False


def test_changed_confidence_fails():
    """Round 5: final conf 0.25 -> 0.99 was invisible."""
    tm, em, dev, fin, inv = _artifacts()
    fin["provenance"]["detector"]["conf"] = 0.99
    assert _links(tm, em, dev, fin, inv)["final_matches_canonical_protocol"] is False


def test_both_evaluations_moved_off_recipe_fail():
    """Round 5's sharpest one: an equality check between dev and final cannot notice that BOTH
    were moved to an arbitrary tracker, imgsz 640 and classes '0'. The recipe is pinned."""
    tm, em, dev, fin, inv = _artifacts()
    for m in (dev, fin):
        m["provenance"]["tracker"]["sha256"] = SHA_C
        m["provenance"]["detector"]["imgsz"] = 640
        m["provenance"]["detector"]["classes"] = "0"
    links = _links(tm, em, dev, fin, inv)
    assert links["dev_matches_canonical_protocol"] is False
    assert links["final_matches_canonical_protocol"] is False


def test_wrong_source_split_or_role_fails():
    tm, em, dev, fin, inv = _artifacts()
    fin["provenance"]["split"]["meta"]["source_split"] = "train"
    assert _links(tm, em, dev, fin, inv)["final_matches_canonical_protocol"] is False


# ---------- end to end ----------

def test_build_final_swap_flips_verdict():
    tm, em, dev, fin, inv = _artifacts()
    fin["provenance"]["split"]["meta"]["game_assignment"] = {"9": "test", "3": "val", "5": "train"}
    out = mr.build(dev, fin, None, None, em, tm, 0.48, 0.49, inv)
    assert out["leak_free"]["provably_leave_one_game_out"] is False


def test_build_clean_evidence_certifies():
    tm, em, dev, fin, inv = _artifacts()
    out = mr.build(dev, fin, None, None, em, tm, 0.48, 0.49, inv)
    assert out["leak_free"]["provably_leave_one_game_out"] is True
    assert out["dev"]["eval_game"] == "4" and out["final"]["eval_game"] == "2"


def test_gates_are_read_not_hardcoded(tmp_path):
    p = tmp_path / "rung1.json"
    p.write_text('{"dev":{"metrics":{"HOTA":0.111}},"final":{"metrics":{"HOTA":0.222}}}')
    assert mr.load_gates(p) == (0.111, 0.222)
