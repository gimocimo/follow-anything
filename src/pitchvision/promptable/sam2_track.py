"""
Promptable single-object tracking with SAM 2 ("click a player, follow them").

SAM 2's video predictor takes a folder of frames, a prompt (point or box) placed
on one frame for an object id, and propagates a mask through the whole clip.

NB: field/method names below follow the facebookresearch/sam2 API; they're
verified against the installed package in scripts/07 before the first real run.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import get_device


def follow_object(
    frames_dir,
    box=None,            # (x1, y1, x2, y2) pixels on the prompt frame
    point=None,          # (x, y) positive click on the prompt frame
    obj_id: int = 1,
    prompt_frame: int = 0,
    model_id: str = "facebook/sam2.1-hiera-small",
    device: str = "auto",
):
    """Prompt one object and propagate its mask across the clip.

    Returns ``(frame_files, masks)`` where ``frame_files`` are the frames SAM 2
    used (sorted) and ``masks[frame_pos]`` is a bool HxW array for the object.
    """
    import torch
    from sam2.sam2_video_predictor import SAM2VideoPredictor

    frames_dir = str(frames_dir)
    dev = get_device(device)
    predictor = SAM2VideoPredictor.from_pretrained(model_id, device=dev)

    frame_files = sorted(
        Path(frames_dir).glob("*.jpg"),
        key=lambda p: int("".join(c for c in p.stem if c.isdigit()) or 0),
    )

    masks = {}
    with torch.inference_mode():
        state = predictor.init_state(video_path=frames_dir, offload_video_to_cpu=True)
        if box is not None:
            predictor.add_new_points_or_box(
                inference_state=state, frame_idx=prompt_frame, obj_id=obj_id,
                box=np.asarray(box, dtype=np.float32),
            )
        elif point is not None:
            predictor.add_new_points_or_box(
                inference_state=state, frame_idx=prompt_frame, obj_id=obj_id,
                points=np.asarray([point], dtype=np.float32),
                labels=np.asarray([1], dtype=np.int32),
            )
        else:
            raise ValueError("provide box=(x1,y1,x2,y2) or point=(x,y)")

        for frame_pos, _obj_ids, mask_logits in predictor.propagate_in_video(state):
            m = (mask_logits[0] > 0.0).cpu().numpy()
            if m.ndim == 3:
                m = m[0]
            masks[frame_pos] = m
    return frame_files, masks
