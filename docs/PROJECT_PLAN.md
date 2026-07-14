# follow-anything — Project Plan & Contract

> **Single source of truth** for what `follow-anything` is, what it will and won't do, and how "done" is defined per stage. It is **version-controlled on purpose**: any change to scope or plan must be an explicit edit to this file, committed with a clear message and logged in §7. That is the anti-drift mechanism — **if a change isn't reflected here (and in git history), it is not an agreed change.**

## 1. North-star goal
Build a **promptable, track-anything video system**: point at any object (a player, a car, a person, an animal — even a category never trained on) and follow it through a clip; detect and track *everything* on screen with persistent identities; and, as the flagship, lift the scene into a **3D bird's-eye reconstruction**. **Football is the proving ground / showcase** (dense, fast, near-identical targets, broadcast camera cuts = tracking on hard mode); the pipeline is domain-agnostic.

**Portfolio thesis:** demonstrate, end-to-end and *rigorously*, the ability to build a modern CV tracking system — the standout artifact being a **real-time, on-device promptable tracker** characterized on an honest accuracy-vs-latency Pareto (engineering + rigorous characterization, not new-SOTA research).

## 2. Scope
**In scope:** promptable single/multi-object tracking (SAM 2) + full-scene detect+track (YOLO + BoT-SORT); football showcase on SoccerNet SN-GSR-2025; the 6-rung ladder (§4); rigorous evaluation; efficiency/real-time/on-device work (distillation, quantization, export); a public repo + shareable demos.

**Out of scope (unless explicitly added here):** multi-camera stadium rigs; production/live-broadcast infrastructure; a commercial guaranteed-uptime system; non-video modalities; beating foundation-model SOTA / novel-architecture research; fully-automated 90-minute live production.

**Extending scope is allowed but must be intentional:** edit §2 + §4, commit, and log it in §7. Codex flags anything in the repo that isn't traceable to this section.

## 3. Invariants (non-negotiable — checked every review)
1. **Leak-free evaluation** — splits are game/match-disjoint; no clip from a test match appears in training. Frame-level random splits are forbidden.
2. **Metrics validated before use** — every metric is unit-tested to known values before being trusted on real data.
3. **Honest baselines** — no reported number without a sanity check; a too-good/too-bad result is investigated before it is published.
4. **Reproducibility** — every reported number reproduces from a committed script + config.
5. **No silent scope drift** — all work maps to a rung in §4; anything else is flagged.

## 4. The rungs (definition of done)
A rung is "done" only when its **gate** is met *and verified* (by us + a Codex adversarial pass).

| # | Rung | Gate (measurable definition of done) | Status |
|---|---|---|---|
| 1 | **Baseline + measurement foundation** | Leak-free game-disjoint split; HOTA/MOTA/IDF1 via TrackEval, unit-tested to *exact* values; a reproducible (provenance-stamped) zero-shot baseline confirmed on an untouched final set | ✅ **done** — dev 0.481 / **untouched-final 0.492** (§7) |
| 2 | **Promptable "click, follow"** | A working spotlight video from a single click/box prompt on real football (SAM 2 mask propagation) | ✅ done (demo) |
| 3 | **Robust** | HOTA **> 0.481** confirmed on the *untouched* final set (official `valid` split), tuned only on the dev benchmark (game 4); write up *what* moved it; occlusion/re-ID/ball measured (AssA ↑, IDSW ↓) | ⬜ |
| 4 | **Permanence** | A benchmark of ID consistency across shot cuts/viewpoints + a method that measurably improves it vs. Rung 3 | ⬜ |
| 5 | **Real-time / on-device** (novelty focus) | An **accuracy (HOTA/IoU) vs latency (fps) Pareto** on our data + target hardware, with an exported runtime (CoreML/ONNX/TensorRT) and ≥1 point improving on off-the-shelf efficient variants (EdgeSAM/EdgeTAM/EfficientTAM lineage) | ⬜ |
| 6 | **3D tactical replay** (flagship) | Pitch homography → top-down minimap → depth-lifted 3D; a demo from a *single* broadcast camera | ⬜ |

Detailed per-rung deliverables & rationale live in each `handoff/rung-N/handoff.md`.

## 5. Current status
- **Rungs 1–2 done** (Rung 1 hardened + confirmed on an untouched final set after an adversarial review, §7). Public repo: https://github.com/gimocimo/follow-anything
- **Baseline to beat: HOTA 0.481 (dev, game 4) / 0.492 (untouched-final, `valid` game 2)** — bbox-HOTA, all object categories; the two agree, so it isn't contamination-inflated. Official metric later: **GS-HOTA** (pitch coordinates).
- Caveats: internal package still named `pitchvision`; local dev on Apple **MPS** (~1 fps SAM 2, a Mac/hardware limit as much as a model one); disk tight; GSR-`train` has only 3 games → **leave-one-game-out** split.

## 6. Review protocol (Codex adversarial passes)
After each rung is completed:
1. Write `handoff/rung-N/handoff.md` (from `handoff/TEMPLATE-handoff.md`): what was built, results, decisions, known issues, how to reproduce.
2. Run Codex with `handoff/rung-N/codex-review-prompt.md`. Codex reads **this plan + the handoff + the repo**, then runs an **adversarial audit**: (a) verify the rung's gate is *genuinely* met; (b) try to break the claims/numbers; (c) check the §3 invariants; (d) check repo state vs this plan (drift); (e) prioritized fixes + enhancements.
3. Paste Codex's output into `handoff/rung-N/codex-report.md`; triage; fix; optionally re-review.
4. **Do not advance** to the next rung until the current gate is verified and blocking findings are resolved.

`handoff/` is **gitignored** (local working scaffolding, not the public deliverable). This plan is **tracked** (the contract).

## 7. Decision log (append-only — every intentional scope/plan change)
- **2026-07-13** — Plan laid down. Data source pivoted SoccerNet-Tracking → **SN-GSR-2025** (KAUST share returned 401/dead). Repo renamed `pitch-vision` → **follow-anything** (general scope; football = showcase). Rungs 1–2 complete. Codex adversarial-review process adopted.
- **2026-07-13 (later)** — **Rung-1 Codex review returned NO.** Adopted a hardening pass: committed a fail-closed **leave-one-game-out** split (fixes the empty-test-set default); strengthened the metric self-test to **exact** values (+ IoU-sensitivity case) under pytest/CI; added a **provenance manifest** to the baseline eval; require `info.game_id` for gate splits; **category policy = include category-7 "other"** (baseline stays 0.481, per-class diagnostics added); relabeled **game 4 as a dev benchmark** and reserved the official **`valid`** split as the untouched final-confirmation set (downloading); removed dead SoccerNet-Tracking code (`scripts/01`(old)/`03`/`04`, `data/soccernet.py`) and fixed the public data path.
- **2026-07-13 (later still)** — **Rung-1 hardening complete + confirmed.** P1s closed (commit 26d3f24): vendored `botsort.yaml`, strict fail-closed conversion + integrality guards, tracker-persistence regression test in CI, tabulate. Downloaded `valid` (11 GB, via stall-detecting curl after the Python downloaders kept stalling) and ran the **untouched final baseline**: **HOTA 0.492** (`valid` game 2, LOO) — matches the dev 0.481, so the baseline is *not* evaluation-selection-inflated. **Rung 1 → done**; ready for Codex re-review, then Rung 3.
