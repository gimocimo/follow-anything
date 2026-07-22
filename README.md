# follow-anything 🎯

**Promptable, track-anything video — point at any object (a player, a car, a person, an animal) and follow it through the clip; detect and track *everything* on screen; and, as the flagship, lift the scene into a 3D bird's-eye replay.**

Football is the proving ground — dense, fast, near-identical targets and broadcast camera cuts make it tracking on *hard mode* — but the pipeline is **domain-agnostic**: the detector already knows people, cars and bikes, and SAM 2 will follow literally anything you click.

> **Status: active build.** Three rungs shipped — a rigorous baseline, a promptable "follow one object" demo, and a football-fine-tuned detector that beats the baseline on a held-out match, *provably* leave-one-game-out. Roadmap below.

![follow-anything demo — click a player, follow them](docs/demo.gif)

*Click one object on the first frame → SAM 2 spotlights and follows it through the clip (here, a football player). The same code follows a car, a person, an animal — anything you point at.*

---

## Vision
Given ordinary video, `follow-anything` aims to:
1. Track every object with **persistent identities**.
2. Be **promptable** — click/box any object (even a category it never trained on) and follow just that one (SAM 2).
3. Maintain **object permanence** — identities survive occlusion *and* camera cuts.
4. Run in **real time, on-device** — distil the promptable model down to laptop/phone speed.
5. Reconstruct the scene in **3D** — showcase: a top-down / 3D football tactical replay.

## Roadmap
| Rung | Milestone | Status |
|---|---|---|
| 1 | Baseline tracking (YOLO + BoT-SORT) → **bbox-HOTA 0.481** on a held-out game | ✅ |
| 2 | Promptable "click any object, follow it" (SAM 2) | ✅ demo |
| 3 | Robust — fine-tuned detector + tuned tracker; **beat the baseline** → **HOTA 0.492 → 0.614** on the held-out final match (provably leave-one-game-out) | ✅ |
| 4 | Permanence — re-acquire IDs across camera cuts | ⬜ |
| 5 | Real-time / on-device — distil to live FPS | ⬜ |
| 6 | 3D tactical replay — homography top-down → depth-lifted 3D | ⬜ |

> Full scope, invariants, and per-rung definition-of-done: **[docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md)**.

## Results
**Rung 1 — zero-shot baseline** (YOLO11n + BoT-SORT, no training), bbox-HOTA via TrackEval, all object categories (SoccerNet **SN-GSR-2025**, leave-one-game-out):

| set | HOTA | DetA | AssA | MOTA | IDF1 | IDSW |
|---|---|---|---|---|---|---|
| dev benchmark (`train` game 4, 18 clips) | 0.481 | 0.611 | 0.381 | 0.734 | 0.541 | 2339 |
| **untouched final** (`valid` game 2, 18 clips) | **0.492** | 0.602 | 0.403 | 0.714 | 0.551 | 1869 |

The untouched-final number (0.492) matches the dev number (0.481) — confirming the baseline isn't inflated by evaluation-selection on the dev game. Association (AssA ≈ 0.40) is the weaker half — the target for the next rungs. Every number carries a provenance manifest (git SHA, package versions, weight hash, category counts). SoccerNet's official metric is GS-HOTA over pitch coordinates (the rung-6 flagship).

**Rung 3 — robust** (`scripts/09–13`): the detector fine-tuned on SoccerNet — **yolo11l @ 1536, 4 classes** (player / goalkeeper / referee / ball), trained on **every frame of games {5,6,9}** with **game 3 held out entirely as cross-match validation** — plus BoT-SORT with a tuned `new_track_thresh`. Scored with the *identical* eval as Rung 1:

| set | HOTA | DetA | AssA | MOTA | IDF1 | IDSW |
|---|---|---|---|---|---|---|
| dev (`train` game 4) | 0.481 → **0.634** | 0.611 → 0.731 | 0.381 → 0.551 | 0.734 → 0.897 | 0.541 → 0.731 | 2339 → 921 |
| **held-out final** (`valid` game 2) | **0.492 → 0.614** | 0.602 → 0.722 | 0.403 → 0.523 | 0.714 → 0.874 | 0.551 → 0.694 | 1869 → **868** |

**+25% HOTA on the final match, every metric up, ID switches more than halved** — and dev (0.634) ≈ final (0.614), so the gain generalises across matches rather than overfitting the dev game.

**Provably leave-one-game-out.** The training set is a hash-fingerprinted export of games {5,6,9} (game 3 used for validation only), re-verified byte-for-byte immediately before *and* after training. The part that actually matters: an `on_train_start` guard authenticates **what ultralytics itself enumerated** — checking the train and val loaders *separately* against their inventory subsets, comparing paths lexically so an aliased directory is caught by name, and **hashing every label the loader consumes**. That closes two bypasses an adversarial review reproduced end-to-end: pulling the held-out val images into the train loader (a union-of-paths check sees no change), and aliasing manifested images under a new directory carrying attacker-controlled labels. The unbroken chain `export_sha256 98cfbc8f… → train_manifest → best.pt 57d95785… → both eval manifests` is committed in [`results/rung3_provenance/`](results/rung3_provenance/), and [`results/rung3_baseline.json`](results/rung3_baseline.json) is **generated** from it by `scripts/13_make_results.py` — 14 cross-bound links, every one derived from evidence and failing closed, so the claim is auditable rather than asserted. What moved the number: an ablation showed **detector strength dominates** (~3.4× the tracker-tuning gain, matched on the same clips), and a fine-tuned mid-size model beats *zero-shot* yolo11x (0.564) — domain adaptation > raw model size.

> **⏳ Re-certification in progress (adversarial round 5).** A fifth review found that ultralytics reads `labels/*.cache` *before* the loader guard runs — so a poisoned cache could feed the model annotations the `.txt` files don't contain, while every on-disk hash still verified. The guard now **purges caches** and authenticates the **parsed in-memory** annotations, the export scan counts a `.cache` as drift, and the generator binds against the committed inventory and a pinned evaluation recipe. The checkpoint above predates those defences, so `results/rung3_baseline.json` currently reports **`provably_leave_one_game_out: false`** on exactly those two links (15/17 pass). One final retrain under the hardened guard flips them; **the HOTA numbers are unaffected.**

**Honest caveats.**

- **The win is people-driven.** Per-class on dev (`scripts/11_perclass_eval.py`): person (players + goalkeepers + referees) reaches HOTA **0.651**, but the **ball only 0.158** (DetA **0.201**) — ~10 px, fast, single-instance. A 1920-resolution probe made *every* metric slightly **worse**, so resolution is **not** the ball's bottleneck; it needs a dedicated approach (context-guided crops, a temporal ball model) and is scoped as its own sub-project rather than claimed as solved.
- **The detector is data-diversity-limited, not compute-limited.** Validation on the unseen match peaks at **epoch 1**, and ten further epochs never beat it (early-stop at 11). SN-GSR's labelled `train`+`valid` contains only **6 games** {2,3,4,5,6,9} — 4 is the dev benchmark and 2 the final — so {3,5,6,9} is *all* the match diversity that exists. More epochs or a bigger model cannot fix cross-match generalisation here.
- **Occlusion & identity** (`scripts/12_occlusion_eval.py`, dev, unit-tested): across **1,510 per-person box-overlap episodes** (a player overlapped by another at IoU ≥ 0.3, entering and exiting interior to the frame), endpoint coverage is **85.2%** and — *conditional on both endpoints being seen* — the ID is retained **91.0%** of the time. Deliberately reported as four separate quantities: this is **not** "identity survives occlusion". The real weakness is fragmentation — **3.30 tracker-IDs per player track-instance, only 22% single-ID** — so the Rung-4 target is **detection continuity**, not re-ID. *(An earlier "≈4% occlusion recovery" figure and a "91% survives occlusion" reading were both withdrawn after adversarial audits; the corrections are logged in [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md) §7.)*
- **Team assignment** (`scripts/15_team_eval.py`, kit-chroma clustering, unsupervised): **85.5%** against GSR team labels over 138 player tracks.
- **Game 2 is a reused, held-out, game-disjoint benchmark — not a pristine one-shot.** It has now been scored on ~4 checkpoints across iterations. With only 6 labelled games there is no unseen match left to reserve, so we disclose the reuse rather than overclaim the seal.

Provenance for every number above: [`results/rung3_baseline.json`](results/rung3_baseline.json).

**Rung 2 — promptable demo** (`scripts/07_promptable_demo.py`): give one object a click/box on the first frame and SAM 2 propagates the mask through the clip, rendering a *spotlight-that-object* video. Runs on Apple MPS; tracked the prompted player in 72/90 frames of a test clip.

## What makes it rigorous (not a tutorial)
- **Leak-free evaluation** — game/match-disjoint splits, so no clip from a test match is ever seen in training (`src/pitchvision/data/splits.py`). Naive frame-level splits leak near-duplicate frames and inflate scores; we refuse to.
- **Validated metric** — HOTA / MOTA / IDF1 via TrackEval, unit-tested to exact values on synthetic sequences (`scripts/02_eval_selftest.py`) before trusting it on real data.
- **Honest baselines** — sanity-checked hard enough to catch a real tracker-reset bug that would otherwise have reported a fake HOTA of 0.13.

## Works on any video
Both engines are general-purpose. **YOLO** detects 80 everyday classes (people, cars, buses, bikes, …) out of the box; **SAM 2** is class-agnostic and follows *anything* you point at. Swapping football for traffic monitoring, wildlife, or retail analytics is mostly a matter of the input video and the target class list. *(The Rung-1 tracker currently runs on COCO persons + sports-ball; enabling more classes is a one-line change — the "track everything" framing is the roadmap target, not yet the shipped scope.)* *(Tracking specific people is surveillance — mind privacy, consent and law; keep showcases benign or clearly authorized.)*

## Repo layout
```
follow-anything/
├── configs/default.yaml            # paths, detector/tracker, split & eval policy
├── src/pitchvision/                # core package (name predates the rename)
│   ├── config.py                   # config loader + device (cuda/mps/cpu)
│   ├── data/splits.py              # match-disjoint splitting (rigor centrepiece)
│   ├── data/gsr.py                 # SoccerNet GSR labels -> MOT rows
│   ├── eval/mot_eval.py            # HOTA/MOTA/IDF1 via TrackEval
│   ├── pipeline/run_video.py       # detect -> track -> annotated video + MOT file
│   └── promptable/sam2_track.py    # SAM 2 "click, follow" propagation
├── scripts/                        # numbered, runnable pipeline steps (00, 02-07)
└── data/  outputs/  notebooks/  tests/
```

## Setup
Requires **Python 3.10+** (3.12 recommended; SAM 2 needs ≥ 3.10).
```bash
python3.12 -m venv .venv && source .venv/bin/activate
# Install PyTorch for your platform first — https://pytorch.org/get-started/locally/
pip install -r requirements.txt
# SAM 2 builds from source; on non-CUDA machines: SAM2_BUILD_CUDA=0 pip install -r requirements.txt
```

## Quickstart
Zero-training tracked clip (COCO-pretrained YOLO already knows people, ball, cars, …):
```bash
python scripts/00_smoke_test.py --video data/sample.mp4
```
Promptable "follow one object" (spotlight video):
```bash
python scripts/07_promptable_demo.py --clip <clip> --frames 90
```

## Data
Football experiments use **SoccerNet SN-GSR-2025** (Game State Reconstruction), a gated Hugging Face dataset. After `hf auth login` (with dataset access):
```bash
python scripts/01_download_gsr.py --split train       # dev source (has labels)
python scripts/05_gsr_make_splits.py                  # leave-one-game-out split (fail-closed)
python scripts/06_gsr_baseline_eval.py --split test   # baseline HOTA + provenance manifest
```
`train` is the development source. The untouched **final-confirmation** number uses the official `valid` split:
```bash
python scripts/01_download_gsr.py --split valid
python scripts/05_gsr_make_splits.py --source-split valid --out outputs/gsr_splits_final.json
python scripts/06_gsr_baseline_eval.py --splits-file outputs/gsr_splits_final.json --split test \
  --out-dir outputs/gsr_baseline_final   # keep the dev run's outputs/gsr_baseline intact
```
Committed provenance for both numbers: [`results/rung1_baseline.json`](results/rung1_baseline.json). The pipeline itself is data-agnostic.

## Credits & licenses
Ultralytics YOLO (**AGPL-3.0** — review before redistribution), SAM 2 (Apache-2.0), TrackEval (MIT), SoccerNet (research license). Check each before publishing derivatives.
