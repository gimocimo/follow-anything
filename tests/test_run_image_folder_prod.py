"""Guards the PRODUCTION tracking path: ``run_image_folder`` must make exactly ONE
streaming ``model.track`` call over the WHOLE folder (``source`` == the directory),
with ``persist=True``, ``stream=True``, and the resolved (vendored) tracker config.

The per-frame-reset bug — calling ``track`` once per frame, which recycles track IDs
and collapses AssA (a fake HOTA of ~0.13) — is the specific regression this must
catch. An earlier version of this test created only ONE frame, which made "exactly
one call" vacuously true: a per-frame implementation over a single frame also makes
one call, so the guard caught nothing. This version uses >=2 frames AND asserts
``source`` is the folder, then proves — via two negative controls — that the same
contract check actually REJECTS a per-frame implementation. Without the negative
controls, a passing assertion doesn't prove the assertion can fail.

Uses a fake ``ultralytics`` module so it runs without torch/ultralytics.
"""
import sys
import types
from pathlib import Path

import pytest

VENDORED = "configs/trackers/botsort.yaml"
# The exact vendored tracker the production path must resolve to (repo-root / VENDORED).
_VENDORED_TRACKER = Path(__file__).resolve().parents[1] / VENDORED


def _assert_single_streaming_call(calls, img_dir):
    """The production tracking contract. Raises ``AssertionError`` if violated.

    A per-frame-reset implementation breaks this in two independent ways: it makes
    one call PER frame (so ``len(calls) != 1`` for a multi-frame folder), and each
    call's ``source`` is a single frame path, not the folder itself.
    """
    assert len(calls) == 1, f"expected exactly ONE streaming track call, got {len(calls)}"
    kw = calls[0]
    assert Path(str(kw.get("source"))).resolve() == Path(img_dir).resolve(), \
        f"must stream the whole folder; source was {kw.get('source')!r}, not {img_dir}"
    assert kw.get("stream") is True, "must stream the whole folder (not per-frame)"
    assert kw.get("persist") is True, "must persist tracker state"
    # Exact resolved-path equality (not a suffix match): the production path must
    # resolve to *this* repo's vendored, version-pinned tracker config.
    assert Path(str(kw.get("tracker", ""))).resolve() == _VENDORED_TRACKER.resolve(), \
        f"production must use the vendored tracker {_VENDORED_TRACKER}, got {kw.get('tracker')!r}"


def _make_frames(img_dir, n):
    """Create ``n`` empty frame files (existence only; the fake YOLO never reads pixels)."""
    img_dir.mkdir()
    for i in range(1, n + 1):
        (img_dir / f"{i:06d}.jpg").write_bytes(b"")
    return img_dir


def _install_fake_yolo(monkeypatch, calls):
    """Install a fake ``ultralytics`` whose ``YOLO.track`` records its kwargs and,
    when streaming a folder, yields one result per frame file (realistic streaming)."""

    class _Boxes:
        id = None  # keeps this test on the call-contract, not tensor parsing

    class _Result:
        def __init__(self, p):
            self.path = p
            self.boxes = _Boxes()

    class _FakeYOLO:
        def __init__(self, weights):
            pass

        def track(self, **kw):
            calls.append(kw)
            src = Path(str(kw["source"]))
            frames = sorted(src.glob("*.jpg")) if src.is_dir() else [src]
            return iter([_Result(str(f)) for f in frames])

    fake = types.ModuleType("ultralytics")
    fake.YOLO = _FakeYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", fake)


def _load_run_image_folder():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from pitchvision.pipeline.run_video import run_image_folder
    return run_image_folder


def test_run_image_folder_streaming_persist_vendored(tmp_path, monkeypatch):
    """Production path: exactly one streaming call over the folder, vendored tracker.

    >=2 frames on disk, so "exactly one call" is a real constraint (a per-frame impl
    would make two calls here).
    """
    calls = []
    _install_fake_yolo(monkeypatch, calls)
    run_image_folder = _load_run_image_folder()

    img = _make_frames(tmp_path / "img1", n=2)
    rows, n_frames = run_image_folder(img, device="cpu")

    _assert_single_streaming_call(calls, img)   # the contract
    assert n_frames == 2, "must see every frame in the folder"


def test_contract_rejects_per_frame_reset(tmp_path, monkeypatch):
    """Negative control (call count): a real per-frame driver is REJECTED.

    Runs the fake YOLO per-frame (the bug) and proves the SAME contract check the
    production test passes will reject it — so the guard genuinely discriminates.
    """
    calls = []
    _install_fake_yolo(monkeypatch, calls)
    from ultralytics import YOLO  # the fake just installed

    img = _make_frames(tmp_path / "img2", n=2)
    model = YOLO("yolo11n.pt")
    for f in sorted(img.glob("*.jpg")):          # the per-frame-reset bug
        list(model.track(source=str(f), stream=True, persist=True,
                         tracker=f"/repo/{VENDORED}"))

    assert len(calls) == 2, "sanity: the per-frame bug makes one call per frame"
    with pytest.raises(AssertionError):
        _assert_single_streaming_call(calls, img)


def test_contract_rejects_single_frame_source(tmp_path):
    """Negative control (source): even a SINGLE call is rejected when its source is
    one frame rather than the folder — the exact hole a 1-frame fixture left open."""
    img = _make_frames(tmp_path / "img3", n=2)
    frame0 = sorted(img.glob("*.jpg"))[0]
    bad = [{"source": str(frame0), "stream": True, "persist": True,
            "tracker": f"/repo/{VENDORED}"}]
    with pytest.raises(AssertionError):
        _assert_single_streaming_call(bad, img)
