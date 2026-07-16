#!/usr/bin/env python3
"""Rung 3 experiment harness: sweep detector/tracker configs on the DEV set and rank
by HOTA, reusing the EXACT sealed eval path (scripts/06_gsr_baseline_eval.py) so every
number is directly comparable to the committed baseline (dev 0.481 / final 0.492).

All tuning happens on DEV (train game 4). The sealed final set (valid game 2) is NEVER
touched here — the winning config is confirmed once, separately, at the end of the rung.

    python scripts/08_rung3_sweep.py --set A                 # association sweep (yolo11n)
    python scripts/08_rung3_sweep.py --set B                 # detector sweep
    python scripts/08_rung3_sweep.py --set A --max-seqs 4    # quick smoke

Each config's metrics + provenance land in outputs/rung3_sweep/<set>/<name>/, and a
ranked summary is written to outputs/rung3_sweep/<set>/summary.json.
"""
import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

import yaml

PROJ = Path(__file__).resolve().parents[1]
BASE_TRACKER = PROJ / "configs" / "trackers" / "botsort.yaml"
BASELINE_HOTA = {"test": 0.4814830134807477}  # dev game-4 baseline (results/rung1_baseline.json)

# Each config: name + detector (weights/imgsz/conf) + tracker (either an explicit yaml
# under configs/trackers, or `overrides` applied to the vendored botsort.yaml).
EXPERIMENTS = {
    # ---- Set A: association levers on the frozen yolo11n detector (targets AssA / IDSW) ----
    "A": [
        {"name": "baseline",         "tracker": "botsort.yaml"},
        {"name": "buf60",            "overrides": {"track_buffer": 60}},
        {"name": "buf90",            "overrides": {"track_buffer": 90}},
        {"name": "newtrk040",        "overrides": {"new_track_thresh": 0.40}},
        {"name": "buf60_newtrk040",  "overrides": {"track_buffer": 60, "new_track_thresh": 0.40}},
        {"name": "reid",             "tracker": "botsort_reid.yaml"},
        {"name": "reid_appear050",   "overrides": {"with_reid": True, "model": "auto", "appearance_thresh": 0.50}},
        {"name": "reid_buf60",       "overrides": {"with_reid": True, "model": "auto", "track_buffer": 60}},
    ],
    # ---- Set B: detector strength (targets DetA), on top of the Set-A winner
    #      (new_track_thresh=0.40, held constant so the ranking isolates the detector).
    #      n_1280 doubles as a cross-check: should reproduce the Set-A newtrk040 = 0.4982. ----
    "B": [
        {"name": "n_1280",  "weights": "yolo11n.pt", "imgsz": 1280, "overrides": {"new_track_thresh": 0.40}},
        {"name": "s_1280",  "weights": "yolo11s.pt", "imgsz": 1280, "overrides": {"new_track_thresh": 0.40}},
        {"name": "m_1280",  "weights": "yolo11m.pt", "imgsz": 1280, "overrides": {"new_track_thresh": 0.40}},
        {"name": "x_1280",  "weights": "yolo11x.pt", "imgsz": 1280, "overrides": {"new_track_thresh": 0.40}},
        {"name": "m_1536",  "weights": "yolo11m.pt", "imgsz": 1536, "overrides": {"new_track_thresh": 0.40}},
        {"name": "x_1536",  "weights": "yolo11x.pt", "imgsz": 1536, "overrides": {"new_track_thresh": 0.40}},
    ],
}


def resolve_tracker(cfg, dst_dir):
    """Return a tracker-yaml path for this config. Explicit `tracker` -> that vendored
    file; `overrides` -> a generated yaml (vendored botsort.yaml + overrides)."""
    if "tracker" in cfg:
        return str(PROJ / "configs" / "trackers" / cfg["tracker"])
    base = yaml.safe_load(BASE_TRACKER.read_text())
    base.update(cfg.get("overrides", {}))
    dst_dir.mkdir(parents=True, exist_ok=True)
    out = dst_dir / f"{cfg['name']}.yaml"
    out.write_text(yaml.safe_dump(base, sort_keys=False))
    return str(out)


def run_config(cfg, args, out_root):
    name = cfg["name"]
    out_dir = out_root / name
    metrics_json = out_dir / "baseline_metrics.json"
    tracker = resolve_tracker(cfg, out_root / "trackers")
    weights = cfg.get("weights", "yolo11n.pt")
    imgsz = cfg.get("imgsz", 1280)
    conf = cfg.get("conf", 0.25)

    if metrics_json.exists() and not args.force:
        c = json.loads(metrics_json.read_text())
        pa = c.get("provenance", {}).get("args", {})
        # Reuse ONLY if EVERY knob that changes the number matches: split + splits-file,
        # subset scope + exact #seqs, detector weights, imgsz, conf, resolved tracker path.
        # (A split+subset-only key silently reuses e.g. a 1-clip smoke for a 6-clip run.)
        same = (c.get("split") == args.split
                and str(pa.get("splits_file")) == str(args.splits_file)
                and c.get("subset") == bool(args.max_seqs)
                and (not args.max_seqs or c.get("n_seqs") == args.max_seqs)
                and str(pa.get("weights")) == str(weights)
                and int(pa.get("imgsz", -1)) == int(imgsz)
                and float(pa.get("conf", -1.0)) == float(conf)
                and str(pa.get("tracker")) == str(tracker))
        if same:
            print(f">>> [{name}] cached (exact match) — reuse", flush=True)
            return c

    cmd = [
        sys.executable, str(PROJ / "scripts" / "06_gsr_baseline_eval.py"),
        "--splits-file", args.splits_file, "--split", args.split,
        "--weights", str(weights), "--imgsz", str(imgsz), "--conf", str(conf),
        "--tracker", tracker, "--device", args.device, "--out-dir", str(out_dir),
    ]
    if args.max_seqs:
        cmd += ["--max-seqs", str(args.max_seqs)]
    print(f"\n>>> [{name}] {' '.join(cmd[3:])}", flush=True)
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"!!! [{name}] FAILED (exit {e.returncode}) — skipping this config", flush=True)
        return None
    return json.loads(metrics_json.read_text())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", dest="expset", default="A", choices=list(EXPERIMENTS))
    ap.add_argument("--splits-file", default="outputs/gsr_splits.json")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--device", default="mps")
    ap.add_argument("--max-seqs", type=int, default=None)
    ap.add_argument("--force", action="store_true", help="re-run configs even if cached")
    ap.add_argument("--only", default=None, help="comma-separated config names to run (subset of the set)")
    ap.add_argument("--out-root", default=None)
    args = ap.parse_args()

    out_root = Path(args.out_root) if args.out_root else PROJ / "outputs" / "rung3_sweep" / args.expset
    out_root.mkdir(parents=True, exist_ok=True)
    base = BASELINE_HOTA.get(args.split)

    configs = EXPERIMENTS[args.expset]
    if args.only:
        want = {n.strip() for n in args.only.split(",")}
        configs = [c for c in configs if c["name"] in want]

    rows = []
    for cfg in configs:
        res = run_config(cfg, args, out_root)
        if res is None:
            continue  # config failed (e.g. weight download) — already logged; keep going
        m = res["metrics"]
        rows.append({"name": cfg["name"], "HOTA": m["HOTA"], "DetA": m["DetA"],
                     "AssA": m["AssA"], "MOTA": m["MOTA"], "IDF1": m["IDF1"], "IDSW": m["IDSW"]})
    if not rows:
        print("no configs succeeded — check the log")
        return

    # Δ is measured against the sweep's OWN baseline row when present (identical
    # sequences — apples-to-apples), else the committed full-dev baseline.
    in_run_base = next((r["HOTA"] for r in rows if r["name"] == "baseline"), None)
    ref = in_run_base if in_run_base is not None else base
    ref_label = "in-run baseline" if in_run_base is not None else "committed dev baseline"

    rows.sort(key=lambda r: r["HOTA"], reverse=True)
    print(f"\n==== Rung-3 sweep '{args.expset}' on split={args.split}"
          f"{' (SUBSET)' if args.max_seqs else ''} — ranked by HOTA ====")
    print(f"{'config':20s} {'HOTA':>8} {'ΔHOTA':>8} {'DetA':>7} {'AssA':>7} {'MOTA':>7} {'IDF1':>7} {'IDSW':>6}")
    for r in rows:
        d = f"{r['HOTA']-ref:+.4f}" if ref else "   n/a"
        print(f"{r['name']:20s} {r['HOTA']:8.4f} {d:>8} {r['DetA']:7.4f} "
              f"{r['AssA']:7.4f} {r['MOTA']:7.4f} {r['IDF1']:7.4f} {r['IDSW']:6d}")
    print(f"\n(ΔHOTA vs {ref_label} = {ref:.4f}; ΔHOTA > 0 beats it on DEV — "
          f"final-set confirmation is separate & one-shot)")
    if in_run_base is not None and base:
        print(f" in-run baseline {in_run_base:.4f} vs committed {base:.4f} "
              f"(Δ {in_run_base-base:+.4f} — should be ~0 if the harness reproduces)")

    summary = out_root / "summary.json"
    summary.write_text(json.dumps({"set": args.expset, "split": args.split,
                                   "committed_baseline_HOTA": base, "in_run_baseline_HOTA": in_run_base,
                                   "ranked": rows}, indent=2))
    print(f"saved -> {summary}")


if __name__ == "__main__":
    main()
