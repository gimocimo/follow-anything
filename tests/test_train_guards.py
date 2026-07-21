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
    def __init__(self, files): self.im_files = files


class _FakeLoader:
    def __init__(self, files): self.dataset = _FakeDS(files)


class _FakeTrainer:
    def __init__(self, train_files, val_files):
        self.train_loader = _FakeLoader(train_files)
        self.test_loader = _FakeLoader(val_files)


def _expected_files(root, inv):
    return [str((root / "images" / e["subset"] / f"{e['stem']}.jpg").resolve()) for e in inv]


def test_loader_guard_passes_on_exact_match(tmp_path):
    inv = _mk_dataset(tmp_path)
    files = _expected_files(tmp_path, inv)
    state = {}
    td.make_loader_guard(tmp_path, state)(_FakeTrainer(files[:1], files[1:]))
    assert state["loader_guard"]["ok"] is True


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
    """Attack 2 (image path): images enumerated under an in-root alias dir (symlinks to the real
    images). Lexical relpaths see the alias name, so it cannot pass as the manifested split."""
    _mk_dataset(tmp_path)
    alias_train = str(tmp_path / "images" / "alias" / "A_000001.jpg")
    alias_val = str(tmp_path / "images" / "val" / "B_000002.jpg")
    state = {}
    guard = td.make_loader_guard(tmp_path, state)
    with pytest.raises(RuntimeError, match="LOADER MISMATCH"):
        guard(_FakeTrainer([alias_train], [alias_val]))
    assert state["loader_guard"]["ok"] is False


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
