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
import os
from pathlib import Path

import numpy as np

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


def _relpaths(root_str, files):
    """Lexical (symlink-agnostic) posix relpaths of `files` under `root_str`. Lexical is the
    whole point: an in-root alias directory resolves to ITSELF here, not to its symlink target,
    so aliasing manifested images under a different directory name stays visible."""
    return {Path(os.path.relpath(str(f), root_str)).as_posix() for f in files}


def parse_label_file(path):
    """(cls, (cx,cy,w,h)) rows, exactly as ultralytics parses a YOLO label file."""
    rows = []
    txt = Path(path).read_text().strip()
    if not txt:
        return rows
    for ln in txt.splitlines():
        p = ln.split()
        if len(p) >= 5:
            rows.append((int(float(p[0])), tuple(float(x) for x in p[1:5])))
    return rows


def labels_in_memory_mismatched(dataset_dir, label_entries, tol=1e-4):
    """Compare ultralytics' PARSED in-memory annotations against the authenticated .txt on disk.

    Hashing the .txt files is NOT sufficient. Ultralytics writes `labels/{train,val}.cache` and
    loads it *before* `on_train_start`; while a valid cache exists the .txt files are never re-read,
    so a poisoned cache trains on annotations the .txt files do not contain — the manifest would
    still say every label hashed clean (adversarial review, round 5). This authenticates what the
    model will actually consume: `dataset.labels`.
    """
    root = str(Path(dataset_dir).resolve())
    mismatched = []
    for lab in (label_entries or []):
        im = (lab or {}).get("im_file")
        if im is None:
            mismatched.append("<label entry without im_file>")
            continue
        rel = Path(os.path.relpath(str(im), root)).as_posix()
        lbl = Path(dataset_dir) / (rel.replace("images", "labels", 1).rsplit(".", 1)[0] + ".txt")
        if not lbl.exists():
            mismatched.append(rel)
            continue
        want = parse_label_file(lbl)
        try:
            cls = [int(float(c)) for c in np.asarray(lab["cls"]).reshape(-1)]
            box = [tuple(float(v) for v in row) for row in np.asarray(lab["bboxes"]).reshape(-1, 4)]
        except Exception:
            mismatched.append(rel)
            continue
        got = list(zip(cls, box))
        if len(got) != len(want) or not all(
                gc == wc and all(abs(g - w) <= tol for g, w in zip(gb, wb))
                for (gc, gb), (wc, wb) in zip(got, want)):
            mismatched.append(rel)
    return mismatched


def authenticate_loader(dataset_dir, inv, train_files, val_files,
                        train_labels=None, val_labels=None) -> dict:
    """Authoritative loader check — SPLIT-AWARE and LABEL-AWARE.

    Adversarial review (2026-07) broke the weaker guards: verifying the *directory* or the
    *union* of resolved image paths cannot stop a data.yaml that (a) pulls the held-out VAL
    images into the TRAIN loader (union unchanged), or (b) aliases manifested images under a new
    directory carrying attacker-controlled LABELS (resolved image paths still match). This
    verifies each loader independently against its inventory subset AND authenticates the labels
    each loader will actually consume:

      * train images == manifest train subset EXACTLY (no val leakage), val == val subset, no overlap;
      * paths compared LEXICALLY (no symlink resolution) so an alias directory is caught by name;
      * for every consumed image the label ultralytics derives (images/->labels/, *.*->*.txt) is
        hashed on disk and must match the inventory, so swapped/aliased/tampered labels cannot pass.

    Returns a report dict; `ok` only when every split and every consumed label matches.
    """
    root = str(Path(dataset_dir).resolve())
    exp = {"train": set(), "val": set()}
    lbl_sha = {}
    for e in inv:
        sub = e["subset"]
        exp[sub].add(f"images/{sub}/{e['stem']}.jpg")
        lbl_sha[f"labels/{sub}/{e['stem']}.txt"] = e.get("label_sha256")

    tr, va = _relpaths(root, train_files), _relpaths(root, val_files)
    problems = []
    if tr != exp["train"]:
        problems.append(f"train loader != manifest train set "
                        f"(+{sorted(tr - exp['train'])[:3]} -{sorted(exp['train'] - tr)[:3]})")
    if va != exp["val"]:
        problems.append(f"val loader != manifest val set "
                        f"(+{sorted(va - exp['val'])[:3]} -{sorted(exp['val'] - va)[:3]})")
    if tr & va:
        problems.append(f"train/val overlap: {sorted(tr & va)[:3]}")

    bad = []
    for rel in sorted(tr | va):
        lbl_rel = rel.replace("images", "labels", 1).rsplit(".", 1)[0] + ".txt"
        p = Path(dataset_dir) / lbl_rel
        want = lbl_sha.get(lbl_rel)
        if want is None or not p.exists() or sha256_file(p) != want:
            bad.append(lbl_rel)
    if bad:
        problems.append(f"consumed labels not authenticated on disk ({len(bad)}): {bad[:3]}")

    # ...and what ultralytics actually PARSED. A poisoned labels/*.cache is read before
    # on_train_start and makes the on-disk .txt irrelevant, so the disk hash alone proves nothing.
    # FAILS CLOSED when the in-memory annotations are unavailable.
    if train_labels is None or val_labels is None:
        mem_bad = ["<in-memory labels unavailable — cannot authenticate what the loader parsed>"]
        n_checked = 0
    else:
        mem_bad = (labels_in_memory_mismatched(dataset_dir, train_labels)
                   + labels_in_memory_mismatched(dataset_dir, val_labels))
        n_checked = len(train_labels) + len(val_labels)
        if n_checked != len(tr) + len(va):
            problems.append(f"in-memory label count {n_checked} != loader image count {len(tr) + len(va)}")
    if mem_bad:
        problems.append(f"in-memory labels differ from the authenticated .txt "
                        f"({len(mem_bad)}): {mem_bad[:3]}")

    return {"ok": not problems, "n_train": len(tr), "n_val": len(va), "n_files": len(tr | va),
            "train_matches_manifest": tr == exp["train"], "val_matches_manifest": va == exp["val"],
            "labels_authenticated": not bad,
            "in_memory_labels_authenticated": not mem_bad, "n_labels_checked": n_checked,
            "problems": problems}


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

    # EXTRA: walk the whole tree recursively; ANY file the manifest does not authorise (by exact
    # relative path) is drift — no extension filter. The previous scan only considered image
    # extensions and `.txt`, which silently ignored ultralytics' `labels/*.cache` — the file the
    # loader actually reads instead of the .txt labels (adversarial review, round 5).
    extra = []
    for sub, allowed in (("images", exp_imgs), ("labels", exp_lbls)):
        for p in sorted((d / sub).rglob("*")):
            if p.is_file():
                rel = p.relative_to(d).as_posix()
                if rel not in allowed:
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
