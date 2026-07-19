"""Verify a `scripts/09` YOLO export against its own manifest — the guard that binds a
trained checkpoint to the exact bytes it consumed.

The export manifest fingerprints the dataset at **export** time. This module re-checks it at
**train** time. Without that second check there is a real hole: `data.yaml` makes ultralytics
glob the whole tree, so a file added or modified *after* export (e.g. an eval-game frame) is
silently trained on while the manifest still describes the clean export — the training set is
then unprovable, which is exactly what the leave-one-game-out invariant (PROJECT_PLAN §3) forbids.

`verify_export` detects all three drift modes: **missing**, **modified**, and — the dangerous
one — **extra** files the manifest never authorised.

Two subtleties learned from adversarial review, both of which previously allowed a bypass:
  * the scan is **recursive** and keyed on **relative paths**, not filename stems. A nested
    `images/train/sub/x.jpg` (or one whose stem collides with a manifested file) is caught.
  * the image-extension set is taken from **ultralytics itself**, so a `.bmp`/`.webp`/`.tif`
    the loader would happily read cannot slip past a narrower list here.

NOTE: passing this check is necessary but not sufficient — it verifies a *directory*. The
loader must additionally be pinned to that directory (see `scripts/10`'s data.yaml check and
its `on_train_start` guard, which compares ultralytics' own resolved file list).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

try:  # match the loader's own notion of "an image" rather than guessing
    from ultralytics.data.utils import IMG_FORMATS as _FMTS
    IMG_EXTS = tuple(f".{e.lower().lstrip('.')}" for e in _FMTS)
except Exception:  # ultralytics not importable (e.g. metric-only CI)
    IMG_EXTS = (".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif",
                ".tiff", ".webp", ".pfm", ".heic")


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def inventory_fingerprint(entries) -> str:
    """Aggregate fingerprint over the exact (subset, stem, game, image, label) set."""
    fp = hashlib.sha256()
    for e in entries:
        fp.update(f"{e['subset']}/{e['stem']}\t{e['game_id']}\t"
                  f"{e['img_sha256']}\t{e['label_sha256']}".encode())
    return fp.hexdigest()


def expected_relpaths(inv):
    """The exact relative paths this export authorises."""
    imgs = {f"images/{e['subset']}/{e['stem']}.jpg" for e in inv}
    lbls = {f"labels/{e['subset']}/{e['stem']}.txt" for e in inv}
    return imgs, lbls


def verify_export(dataset_dir) -> dict:
    """Re-hash the on-disk dataset and compare it against export_inventory/export_manifest.

    Returns {ok, missing, modified, extra, fingerprint_match, n_expected, export_sha256, games}.
    `ok` is True only when the dataset on disk is byte-identical to the manifested export —
    nothing missing, nothing altered, nothing extra anywhere in the tree.
    """
    d = Path(dataset_dir)
    man_p, inv_p = d / "export_manifest.json", d / "export_inventory.json"
    if not man_p.exists() or not inv_p.exists():
        return {"ok": False, "reason": "no export_manifest.json / export_inventory.json beside data.yaml",
                "missing": [], "modified": [], "extra": [], "fingerprint_match": False,
                "n_expected": 0, "export_sha256": None, "games": None}

    man = json.loads(man_p.read_text())
    inv = json.loads(inv_p.read_text())
    exp_imgs, exp_lbls = expected_relpaths(inv)

    missing, modified = [], []
    for e in inv:
        key = f"{e['subset']}/{e['stem']}"
        img = d / "images" / e["subset"] / f"{e['stem']}.jpg"
        lbl = d / "labels" / e["subset"] / f"{e['stem']}.txt"
        if not img.exists() or not lbl.exists():
            missing.append(key)
            continue
        if sha256_file(img) != e.get("img_sha256"):
            modified.append(f"{key} (image)")
        if hashlib.sha256(lbl.read_bytes()).hexdigest() != e.get("label_sha256"):
            modified.append(f"{key} (label)")

    # EXTRA: walk the whole tree recursively; anything the loader could read that the
    # manifest does not authorise (by exact relative path) is drift.
    extra = []
    for p in sorted((d / "images").rglob("*")):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            rel = p.relative_to(d).as_posix()
            if rel not in exp_imgs:
                extra.append(rel)
    for p in sorted((d / "labels").rglob("*")):
        if p.is_file() and p.suffix.lower() == ".txt":
            rel = p.relative_to(d).as_posix()
            if rel not in exp_lbls:
                extra.append(rel)

    fp_match = inventory_fingerprint(inv) == man.get("export_sha256")
    return {"ok": not missing and not modified and not extra and fp_match,
            "missing": missing, "modified": modified, "extra": extra,
            "fingerprint_match": fp_match, "n_expected": len(inv),
            "export_sha256": man.get("export_sha256"), "games": man.get("observed_games")}


def format_report(rep: dict, limit: int = 5) -> str:
    if rep.get("ok"):
        return (f"dataset verified: {rep['n_expected']} files, games {rep.get('games')}, "
                f"export_sha256 {str(rep.get('export_sha256'))[:16]}…")
    bits = [f"DATASET VERIFICATION FAILED ({rep.get('reason', 'drift detected')})"]
    for k in ("missing", "modified", "extra"):
        v = rep.get(k) or []
        if v:
            bits.append(f"  {k} ({len(v)}): " + ", ".join(map(str, v[:limit])) + (" …" if len(v) > limit else ""))
    if not rep.get("fingerprint_match"):
        bits.append("  inventory fingerprint does not match export_manifest.export_sha256")
    return "\n".join(bits)
