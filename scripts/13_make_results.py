#!/usr/bin/env python3
"""Generate `results/rung3_baseline.json` FROM the runtime manifests (Rung 3).

The committed result must be *derived*, not hand-curated: an auditor should re-run this
against the same artifacts and get the same file.

**Fails closed.** Every leak-free link is computed from evidence, and missing evidence is a
FAILURE, never a pass. Two bugs adversarial review found here, both now locked by tests:
absent hashes compared `None == None` and reported the link satisfied; and the final verdict
trusted a flag stored *in the manifest being audited* rather than deriving it. The gate
thresholds are read from `results/rung1_baseline.json`, not hardcoded.

    python scripts/13_make_results.py
"""
import argparse
import json
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
_HEX = set("0123456789abcdef")


def _valid_sha(s) -> bool:
    return isinstance(s, str) and len(s) == 64 and set(s.lower()) <= _HEX


def _same_hash(a, b) -> bool:
    """Hash equality that fails closed: missing or malformed evidence is never a match."""
    return _valid_sha(a) and _valid_sha(b) and a.lower() == b.lower()


# The canonical Rung-3 game policy. NON-OVERRIDABLE for the committed result: every link below
# checks the evidence against THESE constants, so a stray CLI override can't rubber-stamp the
# wrong games (adversarial review, 2026-07).
TRAIN_GAMES = ("3", "5", "6", "9")
DEV_GAME = "4"
FINAL_GAME = "2"


def _games(x):
    return {str(g) for g in (x or [])}


def _det(m):
    return ((m.get("provenance") or {}).get("detector") or {})


def _trk(m):
    return ((m.get("provenance") or {}).get("tracker") or {})


def _eval_game(m):
    """The game a metrics manifest actually scored = the single 'test' game in its split meta.
    Derived from evidence, not trusted from a hardcoded constant in the generator."""
    ga = (((m.get("provenance") or {}).get("split") or {}).get("meta") or {}).get("game_assignment") or {}
    tests = [str(g) for g, role in ga.items() if role == "test"]
    return tests[0] if len(tests) == 1 else None


def _load(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def load_gates(rung1_path):
    """Read the full-precision gates from Rung 1 rather than hardcoding them."""
    d = _load(rung1_path)
    if not d:
        raise SystemExit(f"cannot read gates: {rung1_path} missing")
    return d["dev"]["metrics"]["HOTA"], d["final"]["metrics"]["HOTA"]


def leak_free_links(tm, em, dev, fin):
    """Derive every link from evidence, CROSS-BINDING identities so contradictory records cannot
    pass. Adversarial review (2026-07) fed this contradictory pre/post records, a count-less loader
    guard, a missing data.yaml SHA, and a final manifest describing a different game/tracker/imgsz —
    and still got all-green. Each of those is now a bound, failing check. All must hold."""
    de = tm.get("dataset_export") or {}
    pre = tm.get("dataset_verified_pre") or {}
    post = tm.get("dataset_verified_post") or {}
    guard = tm.get("loader_guard") or {}
    esha = em.get("export_sha256")
    bestsha = tm.get("output_best_sha256")
    wsha, fsha = _det(dev).get("sha256"), _det(fin).get("sha256")
    observed, requested = _games(em.get("observed_games")), _games(em.get("requested_games"))
    n_exp = pre.get("n_expected")
    links = {
        # dataset identity: export manifest, train manifest's embedded export, and BOTH pre/post
        # verifications must all cite the same export_sha256 over the training game set.
        "train_manifest_links_export": _same_hash(de.get("export_sha256"), esha),
        "pre_verify_matches_export": (pre.get("ok") is True and _same_hash(pre.get("export_sha256"), esha)
                                      and _games(pre.get("games")) == set(TRAIN_GAMES)),
        "post_verify_matches_export": (post.get("ok") is True and _same_hash(post.get("export_sha256"), esha)
                                       and _games(post.get("games")) == set(TRAIN_GAMES)),
        "verify_counts_consistent": (isinstance(n_exp, int) and n_exp > 0 and post.get("n_expected") == n_exp),
        # loader: the hardened guard confirms the SPLIT and the LABELS, over the same file count.
        "loader_split_authenticated": (guard.get("ok") is True and guard.get("train_matches_manifest") is True
                                       and guard.get("val_matches_manifest") is True),
        "loader_labels_authenticated": (guard.get("labels_authenticated") is True and guard.get("n_files") == n_exp),
        # data.yaml authenticated AND its SHA recorded, so the checkpoint is bound to THIS yaml.
        "data_yaml_authenticated": (tm.get("data_yaml_problems") == [] and _valid_sha(tm.get("data_yaml_sha256"))),
        # checkpoint: train-manifest output == the weights BOTH evals actually loaded.
        "train_manifest_binds_output_checkpoint": _same_hash(bestsha, wsha),
        "dev_and_final_same_checkpoint": _same_hash(wsha, fsha),
        # games: export is exactly the training set; neither eval game is present.
        "export_games_are_exactly_train_games": observed == requested == set(TRAIN_GAMES),
        "eval_games_excluded_from_training": bool(observed) and {DEV_GAME, FINAL_GAME}.isdisjoint(observed),
        # eval protocol: dev scored game 4, final scored game 2, both via the SAME tracker/imgsz/classes
        # (derived from each manifest, not hardcoded — Codex swapped these freely in the final).
        "dev_scored_dev_game": _eval_game(dev) == DEV_GAME,
        "final_scored_final_game": _eval_game(fin) == FINAL_GAME,
        "dev_final_same_eval_protocol": (_same_hash(_trk(dev).get("sha256"), _trk(fin).get("sha256"))
                                         and _det(dev).get("imgsz") == _det(fin).get("imgsz")
                                         and str(_det(dev).get("classes")) == str(_det(fin).get("classes"))),
    }
    return links, wsha


def build(dev, fin, pc, oc, em, tm, dev_gate, fin_gate):
    links, wsha = leak_free_links(tm, em, dev, fin)
    return {
        "note": "Rung-3 result: yolo11l @ 1536 (4-class: player/goalkeeper/referee/ball) fine-tuned on "
                "every frame of GSR games {5,6,9}, with game 3 held out entirely as cross-match validation "
                "(stride 10); tracked with tuned BoT-SORT and scored with the SAME eval path as Rung 1. "
                "GENERATED by scripts/13_make_results.py from the runtime manifests in "
                "results/rung3_provenance/; every leak_free field is DERIVED from evidence and fails closed. "
                "dev = train game 4. final = valid game 2 — a held-out, game-disjoint benchmark REUSED across "
                "model iterations, NOT a pristine one-shot (disclosed).",
        "leak_free": {
            "detector_training": "yolo11l @ 1536 (4-class) fine-tuned on every frame of GSR games {5,6,9} "
                                 "(game 3 = cross-match val, stride 10); leave-one-game-out — disjoint from "
                                 "both eval games (4 dev, 2 final)",
            "policy": {"train_games": list(TRAIN_GAMES), "dev_game": DEV_GAME, "final_game": FINAL_GAME},
            "export_games": sorted(_games(em.get("observed_games"))),
            "export_sha256": em.get("export_sha256"),
            "weights_sha256": wsha,
            **links,
            "provably_leave_one_game_out": all(links.values()),
        },
        "tracker": {"config": _trk(dev).get("resolved"), "sha256": _trk(dev).get("sha256")},
        "dev": {"eval_game": _eval_game(dev), "n_seqs": dev.get("n_seqs"), "metrics": dev["metrics"],
                "imgsz": _det(dev).get("imgsz"), "classes": _det(dev).get("classes"),
                "baseline_HOTA": dev_gate, "delta_HOTA": dev["metrics"]["HOTA"] - dev_gate,
                "git_sha": dev["provenance"].get("git_sha")},
        "final": {"eval_game": _eval_game(fin), "n_seqs": fin.get("n_seqs"), "metrics": fin["metrics"],
                  "baseline_HOTA": fin_gate, "delta_HOTA": fin["metrics"]["HOTA"] - fin_gate,
                  "gate": f"beat {fin_gate} -> " + ("MET" if fin["metrics"]["HOTA"] > fin_gate else "NOT MET"),
                  "tracker_sha256": _trk(fin).get("sha256"),
                  "git_sha": fin["provenance"].get("git_sha")},
        "per_class_dev": (pc or {}).get("per_class"),
        "occlusion_dev": oc,
        "packages": dev["provenance"].get("packages"),
        "device": dev["provenance"].get("device"),
        "python": dev["provenance"].get("python"),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dev-dir", default="outputs/gsr_ft_dev")
    ap.add_argument("--final-dir", default="outputs/gsr_ft_final")
    ap.add_argument("--prov-dir", default="results/rung3_provenance")
    ap.add_argument("--rung1", default="results/rung1_baseline.json")
    ap.add_argument("--out", default="results/rung3_baseline.json")
    args = ap.parse_args()

    dev = _load(Path(args.dev_dir) / "baseline_metrics.json")
    fin = _load(Path(args.final_dir) / "baseline_metrics.json")
    pc = _load(Path(args.dev_dir) / "perclass_metrics.json")
    oc = _load(Path(args.dev_dir) / "occlusion_metrics.json")
    em = _load(Path(args.prov_dir) / "export_manifest.json")
    tm = _load(Path(args.prov_dir) / "train_manifest.json")
    missing = [n for n, v in (("dev metrics", dev), ("final metrics", fin),
                              ("export_manifest", em), ("train_manifest", tm)) if v is None]
    if missing:
        raise SystemExit(f"missing required inputs: {missing}")

    dev_gate, fin_gate = load_gates(args.rung1)
    out = build(dev, fin, pc, oc, em, tm, dev_gate, fin_gate)
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    lf = out["leak_free"]
    print(f"wrote {args.out}")
    print(f"  dev   HOTA {out['dev']['metrics']['HOTA']:.4f}  ({out['dev']['delta_HOTA']:+.4f})")
    print(f"  final HOTA {out['final']['metrics']['HOTA']:.4f}  ({out['final']['delta_HOTA']:+.4f})  -> {out['final']['gate']}")
    print("  leak-free links:")
    for k, v in lf.items():
        if isinstance(v, bool):
            print(f"    {'PASS' if v else 'FAIL'}  {k}")
    print(f"  => provably_leave_one_game_out: {lf['provably_leave_one_game_out']}")


if __name__ == "__main__":
    main()
