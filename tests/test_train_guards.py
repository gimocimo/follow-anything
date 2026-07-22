"""Tests for the trainer's dataset guards — the checks that bind a checkpoint to verified bytes.

Adversarial review bypassed directory-only verification twice: by **redirecting `data.yaml`**
(the loader reads the YAML, not the directory) and by hiding files where the scan didn't look.
These tests lock both paths, plus the authoritative guard that compares ultralytics' own
resolved file list against the manifest.
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
_spec = importlib.util.spec_from_file_location("train_detector", ROOT / "scripts" / "10_train_detector.py")
td = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(td)
from pitchvision.data.export_verify import sha256_file


def _mk_dataset(root, stems=(("train", "A_000001", "6"), ("val", "B_000002", "9"))):
    inv = []
    for sub, stem, gid in stems:
        (root / "images" / sub).mkdir(parents=True, exist_ok=True)
        (root / "labels" / sub).mkdir(parents=True, exist_ok=True)
        img = root / "images" / sub / f"{stem}.jpg"
        lbl = root / "labels" / sub / f"{stem}.txt"
        img.write_bytes(f"img-{stem}".encode())
        lbl.write_text("0 0.5 0.5 0.1 0.1\n")
        inv.append({"subset": sub, "stem": stem, "game_id": gid,
                    "img_sha256": sha256_file(img), "label_sha256": sha256_file(lbl)})
    (root / "export_inventory.json").write_text(json.dumps(inv))
    (root / "data.yaml").write_text(yaml.safe_dump(
        {"path": str(root), "train": "images/train", "val": "images/val", "names": {0: "person", 1: "ball"}}))
    return inv


# ---------- data.yaml authentication ----------

def test_data_yaml_clean_passes(tmp_path):
    _mk_dataset(tmp_path)
    assert td.check_data_yaml(tmp_path / "data.yaml", tmp_path) == []


def test_data_yaml_train_redirect_is_caught(tmp_path):
    """THE attack: aim `train:` at another folder while the directory still verifies clean."""
    _mk_dataset(tmp_path)
    outside = tmp_path.parent / "elsewhere_game4"
    outside.mkdir(exist_ok=True)
    p = tmp_path / "data.yaml"
    c = yaml.safe_load(p.read_text())
    c["train"] = str(outside)
    p.write_text(yaml.safe_dump(c))
    probs = td.check_data_yaml(p, tmp_path)
    assert probs and any("train" in s and "canonical" in s for s in probs), probs


def test_data_yaml_root_redirect_is_caught(tmp_path):
    _mk_dataset(tmp_path)
    p = tmp_path / "data.yaml"
    c = yaml.safe_load(p.read_text())
    c["path"] = str(tmp_path.parent)          # move the root out from under the manifest
    p.write_text(yaml.safe_dump(c))
    assert td.check_data_yaml(p, tmp_path)


def test_data_yaml_missing_split_is_caught(tmp_path):
    _mk_dataset(tmp_path)
    p = tmp_path / "data.yaml"
    c = yaml.safe_load(p.read_text())
    del c["val"]
    p.write_text(yaml.safe_dump(c))
    assert any("val" in s for s in td.check_data_yaml(p, tmp_path))


# ---------- loader guard (authoritative: what ultralytics actually resolved) ----------

class _FakeDS:
    def __init__(self, files, labels): self.im_files, self.labels = files, labels


class _FakeLoader:
    def __init__(self, files, labels): self.dataset = _FakeDS(files, labels)


class _FakeTrainer:
    def __init__(self, train_files, val_files, train_labels=None, val_labels=None):
        self.train_loader = _FakeLoader(train_files, train_labels)
        self.test_loader = _FakeLoader(val_files, val_labels)


def _expected_files(root, inv):
    return [str((root / "images" / e["subset"] / f"{e['stem']}.jpg").resolve()) for e in inv]


def _mem_labels(root, inv, subset):
    """In-memory annotations as ultralytics would parse them from the authenticated .txt
    (`_mk_dataset` writes `0 0.5 0.5 0.1 0.1`)."""
    return [{"im_file": str((root / "images" / subset / f"{e['stem']}.jpg").resolve()),
             "cls": [[0]], "bboxes": [[0.5, 0.5, 0.1, 0.1]]}
            for e in inv if e["subset"] == subset]


def test_loader_guard_passes_on_exact_match(tmp_path):
    inv = _mk_dataset(tmp_path)
    files = _expected_files(tmp_path, inv)
    state = {}
    td.make_loader_guard(tmp_path, state)(_FakeTrainer(
        files[:1], files[1:], _mem_labels(tmp_path, inv, "train"), _mem_labels(tmp_path, inv, "val")))
    assert state["loader_guard"]["ok"] is True
    assert state["loader_guard"]["in_memory_labels_authenticated"] is True


def test_loader_guard_rejects_extra_resolved_file(tmp_path):
    """Even with a clean directory, a file the LOADER resolved but the manifest never
    authorised (e.g. via a redirected data.yaml) must abort training."""
    inv = _mk_dataset(tmp_path)
    files = _expected_files(tmp_path, inv)
    state = {}
    guard = td.make_loader_guard(tmp_path, state)
    with pytest.raises(RuntimeError, match="LOADER MISMATCH"):
        guard(_FakeTrainer(files + ["/elsewhere/game4_000100.jpg"], []))
    assert state["loader_guard"]["ok"] is False


def test_loader_guard_rejects_missing_resolved_file(tmp_path):
    _mk_dataset(tmp_path)
    state = {}
    guard = td.make_loader_guard(tmp_path, state)
    with pytest.raises(RuntimeError, match="LOADER MISMATCH"):
        guard(_FakeTrainer([], []))


# ---------- hardened: split + label authentication (the two 2026-07 loader bypasses) ----------

def test_data_yaml_train_list_pulls_val_into_train_is_caught(tmp_path):
    """Attack 1 (yaml level): `train: [images/train, images/val]` trains on the held-out val
    while a union-of-paths check sees no change."""
    _mk_dataset(tmp_path)
    p = tmp_path / "data.yaml"
    c = yaml.safe_load(p.read_text())
    c["train"] = ["images/train", "images/val"]
    p.write_text(yaml.safe_dump(c))
    probs = td.check_data_yaml(p, tmp_path)
    assert probs and any("train" in s for s in probs), probs


def test_loader_guard_rejects_val_image_in_train(tmp_path):
    """Attack 1 (loader level): the val image is enumerated by the TRAIN loader. The union is
    unchanged, but split-aware auth must reject the leak."""
    inv = _mk_dataset(tmp_path)
    files = _expected_files(tmp_path, inv)            # [train_img, val_img]
    state = {}
    guard = td.make_loader_guard(tmp_path, state)
    with pytest.raises(RuntimeError, match="LOADER MISMATCH"):
        guard(_FakeTrainer(files, []))                # BOTH images in train, val empty
    assert state["loader_guard"]["ok"] is False
    assert state["loader_guard"]["train_matches_manifest"] is False


def test_loader_guard_rejects_aliased_images(tmp_path):
    """Attack 2 (image path): a REAL in-root alias dir of symlinks to the manifested images, with
    attacker labels beside them. `.resolve()` would map the symlinks back onto authorised paths;
    the lexical relpath sees the alias name, so it cannot pass as the manifested split."""
    inv = _mk_dataset(tmp_path)
    (tmp_path / "images" / "alias").mkdir(parents=True, exist_ok=True)
    (tmp_path / "labels" / "alias").mkdir(parents=True, exist_ok=True)
    alias_img = tmp_path / "images" / "alias" / "A_000001.jpg"
    alias_img.symlink_to(tmp_path / "images" / "train" / "A_000001.jpg")   # real symlink
    (tmp_path / "labels" / "alias" / "A_000001.txt").write_text("3 0.9 0.9 0.4 0.4\n")  # attacker label
    assert alias_img.resolve() == (tmp_path / "images" / "train" / "A_000001.jpg").resolve()
    state = {}
    guard = td.make_loader_guard(tmp_path, state)
    with pytest.raises(RuntimeError, match="LOADER MISMATCH"):
        guard(_FakeTrainer([str(alias_img)], _expected_files(tmp_path, inv)[1:]))
    assert state["loader_guard"]["ok"] is False
    assert state["loader_guard"]["train_matches_manifest"] is False


def test_loader_guard_rejects_tampered_consumed_label(tmp_path):
    """Attack 2 (label bytes): image paths are canonical, but the label the loader will consume
    was changed after export. Hashing consumed labels must catch it (closes the pre->train TOCTOU)."""
    inv = _mk_dataset(tmp_path)
    files = _expected_files(tmp_path, inv)
    (tmp_path / "labels" / "train" / "A_000001.txt").write_text("1 0.1 0.1 0.2 0.2\n")
    state = {}
    guard = td.make_loader_guard(tmp_path, state)
    with pytest.raises(RuntimeError, match="LOADER MISMATCH"):
        guard(_FakeTrainer(files[:1], files[1:]))
    assert state["loader_guard"]["labels_authenticated"] is False


# ---------- round-5 bypass: ultralytics' label CACHE, read before on_train_start ----------

def test_loader_guard_rejects_poisoned_cache_labels(tmp_path):
    """THE round-5 attack: images and .txt labels all hash clean, but the annotations ultralytics
    actually parsed (from a poisoned labels/*.cache) say class 3 instead of class 0. Hashing the
    .txt proves nothing here — only authenticating the in-memory labels catches it."""
    inv = _mk_dataset(tmp_path)
    files = _expected_files(tmp_path, inv)
    poisoned = _mem_labels(tmp_path, inv, "train")
    poisoned[0]["cls"] = [[3]]                       # cache says "ball"; the .txt says class 0
    state = {}
    guard = td.make_loader_guard(tmp_path, state)
    with pytest.raises(RuntimeError, match="LOADER MISMATCH"):
        guard(_FakeTrainer(files[:1], files[1:], poisoned, _mem_labels(tmp_path, inv, "val")))
    assert state["loader_guard"]["labels_authenticated"] is True        # disk hashes still clean
    assert state["loader_guard"]["in_memory_labels_authenticated"] is False


def test_loader_guard_fails_closed_without_in_memory_labels(tmp_path):
    """If the parsed annotations aren't available we cannot prove what was consumed — refuse."""
    inv = _mk_dataset(tmp_path)
    files = _expected_files(tmp_path, inv)
    state = {}
    guard = td.make_loader_guard(tmp_path, state)
    with pytest.raises(RuntimeError, match="LOADER MISMATCH"):
        guard(_FakeTrainer(files[:1], files[1:]))     # labels default to None
    assert state["loader_guard"]["in_memory_labels_authenticated"] is False


def test_verify_export_flags_label_cache_as_extra(tmp_path):
    """The extension-filtered scan ignored `labels/*.cache` — the very file the loader reads."""
    from pitchvision.data.export_verify import inventory_fingerprint, verify_export
    inv = _mk_dataset(tmp_path)
    (tmp_path / "export_manifest.json").write_text(json.dumps(
        {"export_sha256": inventory_fingerprint(inv), "observed_games": ["6", "9"]}))
    assert verify_export(tmp_path)["ok"] is True
    (tmp_path / "labels" / "train.cache").write_bytes(b"poisoned")
    rep = verify_export(tmp_path)
    assert rep["ok"] is False
    assert any(x.endswith("train.cache") for x in rep["extra"]), rep["extra"]


def test_purge_label_caches_removes_them(tmp_path):
    _mk_dataset(tmp_path)
    (tmp_path / "labels" / "train.cache").write_bytes(b"x")
    (tmp_path / "labels" / "val.cache").write_bytes(b"y")
    removed = td.purge_label_caches(tmp_path)
    assert sorted(removed) == ["labels/train.cache", "labels/val.cache"]
    assert not list(tmp_path.rglob("*.cache"))
