# pitch-vision ⚽️👁️

**Promptable multi-object tracking with object permanence for football — from a single broadcast feed to a 3D tactical replay.**

Click any player and follow them through occlusions and camera cuts; track the ball; and lift the whole play onto a top-down tactical map and into 3D — from ordinary broadcast video.

> **Status: Phase 0 — scaffolding.** Under active construction; see the roadmap.

---

## The vision
Given ordinary football broadcast video, `pitch-vision` aims to:
1. Track every player + the ball with persistent identities.
2. Be **promptable** — point at any object and follow just that one (SAM 2).
3. Maintain **object permanence** — identities survive occlusion *and* broadcast camera cuts.
4. Run in **near real time**.
5. Produce a **top-down tactical map** and a **3D reconstruction** of the play.

## Roadmap
| Rung | Milestone | Key deliverable | Status |
|---|---|---|---|
| 1 | Baseline tracking | YOLO11n + BoT-SORT → **bbox-HOTA 0.481** on a held-out game (SN-GSR-2025) | ✅ |
| 2 | Promptable | SAM 2 "click-a-player, follow them" | ⬜ |
| 3 | Robust | occlusion + ball + re-ID; beat baseline HOTA | ⬜ |
| 4 | Permanence | re-acquire IDs across camera cuts | ⬜ |
| 5 | Real-time | distil / export to live FPS | ⬜ |
| 6 | 3D replay | homography top-down → depth-lifted 3D | ⬜ |

## Results
**Rung 1 — zero-shot baseline** (YOLO11n + BoT-SORT, no training) on the held-out game (leave-one-game-out over SN-GSR-2025 `train`, 18 clips), scored with bbox-HOTA via TrackEval:

| HOTA | DetA | AssA | MOTA | IDF1 | IDSW |
|---|---|---|---|---|---|
| **0.481** | 0.611 | 0.381 | 0.734 | 0.541 | 2339 |

Association (AssA) is the weaker half — the target for the next rungs. SoccerNet's official metric is GS-HOTA over pitch coordinates (the rung-6 flagship).

## What makes it rigorous (not a tutorial)
- **Match-disjoint splits** — no frames from the same match ever appear in both train and test (`pitchvision.data.splits`). Naive frame-level splits leak near-duplicate frames and inflate metrics; we refuse to do that.
- **Modern tracking metrics** — HOTA (primary), plus MOTA / IDF1 / ID-switches via TrackEval. The eval harness *raises* rather than reporting fake numbers until it's wired up.
- **Generalization check** — evaluate on a held-out sport (SportsMOT) to prove it isn't overfit to football.

## Repo layout
```
pitch-vision/
├── configs/default.yaml          # paths, detector/tracker, split & eval policy
├── src/pitchvision/
│   ├── config.py                 # config loader + device (cuda/mps/cpu)
│   ├── data/splits.py            # match-disjoint splitting (rigor centrepiece)
│   ├── detect/  track/  promptable/
│   ├── eval/mot_eval.py          # HOTA/MOTA/IDF1 via TrackEval
│   └── pipeline/run_video.py     # video -> detect -> track -> annotated video + MOT file
├── scripts/00_smoke_test.py      # zero-training tracked football clip
├── data/  outputs/  notebooks/  tests/
```

## Setup
Requires **Python 3.10+** (system Python 3.9 is too old for SAM 2).
```bash
python3.11 -m venv .venv && source .venv/bin/activate
# Install PyTorch for your platform first — https://pytorch.org/get-started/locally/
#   macOS (Apple Silicon / MPS):  pip install torch torchvision
#   CUDA 12.x (Linux training box): pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

## Quickstart — smoke test (no training)
COCO-pretrained YOLO already detects `person` and `sports ball`, so you get a tracked football clip immediately:
```bash
python scripts/00_smoke_test.py --video data/sample.mp4
```
Writes an annotated video + MOTChallenge-format results to `outputs/`.

## Data
[SoccerNet](https://www.soccer-net.org/) — Tracking, Re-Identification, and Game-State-Reconstruction splits. Downloader lands in Phase 0 (`scripts/01_download_soccernet.py`); requires agreeing to the SoccerNet terms.

## Credits & licenses
Ultralytics (**AGPL-3.0** — review before any public release/hosting), SAM 2 (Apache-2.0), TrackEval (MIT), SoccerNet (research license). Check each before publishing.
