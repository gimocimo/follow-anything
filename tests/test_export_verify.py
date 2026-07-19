"""Adversarial tests for the export verifier — the guard that makes a fine-tuned checkpoint
*provably* leave-one-game-out.

Each test is a concrete attack that must be REJECTED. The headline one is
`test_rejects_injected_frame`: an export-time fingerprint alone cannot catch a file added
afterwards, because `data.yaml` makes ultralytics glob the whole tree — so the trainer would
happily learn from an injected eval-game frame while the manifest still looked clean.
"""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pitchvision.data.export_verify import inventory_fingerprint, verify_export

STEMS = (("train", "SNGS-100_000005", "6"), ("val", "SNGS-101_000010", "9"))


def _make_export(root, stems=STEMS):
    """Build a minimal, self-consistent scripts/09-style export."""
    inv = []
    for subset, stem, gid in stems:
        (root / "images" / subset).mkdir(parents=True, exist_ok=True)
        (root / "labels" / subset).mkdir(parents=True, exist_ok=True)
        img = root / "images" / subset / f"{stem}.jpg"
        img.write_bytes(f"IMG:{stem}".encode())
        label_txt = "0 0.5 0.5 0.1 0.2\n"
        (root / "labels" / subset / f"{stem}.txt").write_text(label_txt)
        inv.append({"subset": subset, "stem": stem, "game_id": gid,
                    "img_sha256": hashlib.sha256(img.read_bytes()).hexdigest(),
                    "label_sha256": hashlib.sha256(label_txt.encode()).hexdigest()})
    (root / "export_inventory.json").write_text(json.dumps(inv))
    (root / "export_manifest.json").write_text(json.dumps(
        {"observed_games": ["6", "9"], "requested_games": ["6", "9"],
         "export_sha256": inventory_fingerprint(inv)}))
    return inv


def test_clean_export_verifies(tmp_path):
    _make_export(tmp_path)
    rep = verify_export(tmp_path)
    assert rep["ok"], rep
    assert rep["n_expected"] == 2 and rep["fingerprint_match"]


def test_rejects_injected_frame(tmp_path):
    """THE attack: a frame added AFTER export (e.g. from an eval game). Must be flagged EXTRA."""
    _make_export(tmp_path)
    (tmp_path / "images" / "train" / "SNGS-999_000001.jpg").write_bytes(b"EVAL-GAME-FRAME")
    rep = verify_export(tmp_path)
    assert not rep["ok"], "an injected training frame must fail verification"
    assert any("SNGS-999" in e for e in rep["extra"]), rep


def test_rejects_modified_label(tmp_path):
    _make_export(tmp_path)
    p = next((tmp_path / "labels" / "train").glob("*.txt"))
    p.write_text("1 0.1 0.1 0.9 0.9\n")
    rep = verify_export(tmp_path)
    assert not rep["ok"] and rep["modified"], rep


def test_rejects_modified_image(tmp_path):
    _make_export(tmp_path)
    p = next((tmp_path / "images" / "train").glob("*.jpg"))
    p.write_bytes(b"DIFFERENT-PIXELS")
    rep = verify_export(tmp_path)
    assert not rep["ok"] and rep["modified"], rep


def test_rejects_missing_file(tmp_path):
    _make_export(tmp_path)
    next((tmp_path / "images" / "train").glob("*.jpg")).unlink()
    rep = verify_export(tmp_path)
    assert not rep["ok"] and rep["missing"], rep


def test_rejects_tampered_inventory(tmp_path):
    """Rewriting the inventory to 'authorise' an injected file must break the manifest fingerprint
    (the manifest's export_sha256 is the committed anchor)."""
    inv = _make_export(tmp_path)
    inv.append({"subset": "train", "stem": "SNGS-999_000001", "game_id": "4",
                "img_sha256": "x", "label_sha256": "y"})
    (tmp_path / "export_inventory.json").write_text(json.dumps(inv))
    rep = verify_export(tmp_path)
    assert not rep["ok"] and not rep["fingerprint_match"], rep


def test_rejects_nested_injected_frame(tmp_path):
    """A nested file bypasses a non-recursive scan — and one whose STEM collides with a
    manifested file bypasses stem-based keying. The loader reads it regardless."""
    _make_export(tmp_path)
    nested = tmp_path / "images" / "train" / "sub"
    nested.mkdir(parents=True)
    (nested / "SNGS-100_000005.jpg").write_bytes(b"EVAL-GAME-FRAME")  # colliding stem, different path
    rep = verify_export(tmp_path)
    assert not rep["ok"] and any("sub/" in e for e in rep["extra"]), rep


def test_rejects_exotic_image_format(tmp_path):
    """ultralytics reads .bmp/.webp/.tif too — a narrower extension list here is a bypass."""
    _make_export(tmp_path)
    (tmp_path / "images" / "train" / "sneaky.bmp").write_bytes(b"EVAL-GAME-FRAME")
    rep = verify_export(tmp_path)
    assert not rep["ok"] and any("sneaky.bmp" in e for e in rep["extra"]), rep


def test_unmanifested_dataset_is_not_ok(tmp_path):
    """No manifest at all => cannot be called provably leak-free."""
    (tmp_path / "images" / "train").mkdir(parents=True)
    rep = verify_export(tmp_path)
    assert not rep["ok"]
