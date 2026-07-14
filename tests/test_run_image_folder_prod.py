"""Guards the PRODUCTION tracking path: `run_image_folder` must make exactly ONE
streaming `model.track` call, with `persist=True`, `stream=True`, and the resolved
(vendored) tracker config. A revert to the per-frame-reset bug, or dropping the
vendored-tracker resolution, fails CI here — not just the static golden fixture.

Uses a fake `ultralytics` module so it runs without torch/ultralytics.
"""
import sys
import types
from pathlib import Path


def test_run_image_folder_streaming_persist_vendored(tmp_path, monkeypatch):
    calls = []

    class _Boxes:
        id = None

    class _Result:
        def __init__(self, p):
            self.path = p
            self.boxes = _Boxes()

    class _FakeYOLO:
        def __init__(self, weights):
            pass

        def track(self, **kw):
            calls.append(kw)
            return iter([_Result(str(Path(kw["source"]) / "000001.jpg"))])

    fake = types.ModuleType("ultralytics")
    fake.YOLO = _FakeYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", fake)

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from pitchvision.pipeline.run_video import run_image_folder

    img = tmp_path / "img1"
    img.mkdir()
    (img / "000001.jpg").write_bytes(b"")  # existence only; fake YOLO never reads it

    run_image_folder(img, device="cpu")

    assert len(calls) == 1, f"expected exactly ONE streaming track call, got {len(calls)}"
    kw = calls[0]
    assert kw.get("stream") is True, "must stream the whole folder (not per-frame)"
    assert kw.get("persist") is True, "must persist tracker state"
    assert str(kw.get("tracker", "")).endswith("configs/trackers/botsort.yaml"), \
        f"vendored tracker not used by production path: {kw.get('tracker')!r}"
