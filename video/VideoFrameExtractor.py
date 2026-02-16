"""
video_frame_extractor.py

Extract frames from one or more videos and save them as images.

Supports
- Extract every N frames (with optional deterministic or random jitter offset)
- Or extract a fixed number of frames per video (uniformly across the video)
- Optional resizing (scale factor or fixed output size)
- Optional cropping (fixed crop or random crop)

IMPORTANT NOTE: clearing of the target directory where we store the frames is on by default. This is a destructive
action, proceed with caution

Author: Martijn Folmer
Date created: 2026-02-16
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import os

PathLike = Union[str, Path]
SizeHW = Tuple[int, int]  # (height, width)
CropBox = Tuple[int, int, int, int]  # (x, y, w, h)


@dataclass(frozen=True)
class CropConfig:
    """
    Crop configuration.

    Exactly one of 'fixed_box' or 'random_size' may be set.
    - fixed_box: crop a fixed rectangle (x, y, w, h)
    - random_size: crop a random rectangle of size (h, w) per frame
    """

    fixed_box: Optional[CropBox] = None
    random_size: Optional[SizeHW] = None


def _validate_crop_config(crop: Optional[CropConfig]) -> None:
    if crop is None:
        return

    if crop.fixed_box is not None and crop.random_size is not None:
        raise ValueError("CropConfig: only one of fixed_box or random_size may be set")

    if crop.fixed_box is not None:
        x, y, w, h = crop.fixed_box
        if any(v < 0 for v in (x, y, w, h)) or w == 0 or h == 0:
            raise ValueError("CropConfig.fixed_box must have x,y>=0 and w,h>0")

    if crop.random_size is not None:
        h, w = crop.random_size
        if h <= 0 or w <= 0:
            raise ValueError("CropConfig.random_size must be (h, w) with both > 0")


def _clear_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for item in path.iterdir():
        if item.is_file():
            item.unlink()


def _apply_crop(frame: np.ndarray, crop: Optional[CropConfig], rng: np.random.Generator) -> np.ndarray:
    if crop is None:
        return frame

    h_img, w_img = frame.shape[:2]

    if crop.fixed_box is not None:
        x, y, w, h = crop.fixed_box
        if x + w > w_img or y + h > h_img:
            raise ValueError(
                f"Fixed crop box {(x, y, w, h)} exceeds frame size {(w_img, h_img)} (w,h)."
            )
        return frame[y : y + h, x : x + w]

    if crop.random_size is not None:
        h_crop, w_crop = crop.random_size
        if w_crop > w_img or h_crop > h_img:
            raise ValueError(
                f"Random crop size {(h_crop, w_crop)} exceeds frame size {(h_img, w_img)} (h,w)."
            )
        x0 = int(rng.integers(0, w_img - w_crop + 1))
        y0 = int(rng.integers(0, h_img - h_crop + 1))
        return frame[y0 : y0 + h_crop, x0 : x0 + w_crop]

    return frame


def _apply_resize(
    frame: np.ndarray,
    resize_frac: float,
    resize_to: Optional[SizeHW],
) -> np.ndarray:
    if resize_to is not None and resize_frac != 1.0:
        raise ValueError("Use either resize_to or resize_frac, not both.")

    if resize_to is not None:
        out_h, out_w = resize_to
        if out_h <= 0 or out_w <= 0:
            raise ValueError("resize_to must be (h, w) with both > 0.")
        return cv2.resize(frame, (out_w, out_h))

    if resize_frac != 1.0:
        if resize_frac <= 0:
            raise ValueError("resize_frac must be > 0.")
        h, w = frame.shape[:2]
        new_w = int(round(w * resize_frac))
        new_h = int(round(h * resize_frac))
        new_w = max(1, new_w)
        new_h = max(1, new_h)
        return cv2.resize(frame, (new_w, new_h))

    return frame


def _compute_frame_indices(
    total_frames: int,
    offset_frames: int,
    every_n_frames: int,
    num_frames: Optional[int],
) -> Sequence[int]:
    """
    Decide which base frame indices to sample (before random jitter is applied).

    If num_frames is provided, returns `num_frames` indices uniformly spaced from
    [offset_frames, total_frames-1]. Otherwise returns range(offset_frames, ..., step every_n_frames).
    """
    if total_frames <= 0:
        return []

    start = min(max(offset_frames, 0), total_frames - 1)

    if num_frames is not None:
        if num_frames <= 0:
            raise ValueError("num_frames must be > 0 if provided.")
        # If only 1 frame requested, take the start frame.
        if num_frames == 1:
            return [start]

        end = total_frames - 1
        if start > end:
            return [end]

        # Uniform sampling inclusive of end:
        indices = np.linspace(start, end, num=num_frames)
        return [int(round(x)) for x in indices]

    # Default: every N frames
    if every_n_frames < 1:
        raise ValueError("every_n_frames must be >= 1.")
    return list(range(start, total_frames, every_n_frames))


def retrieve_frames_from_videos(
    video_paths: Iterable[PathLike],
    target_dir: PathLike = "frames",
    clear_target_dir: bool = True,
    offset_frames: int = 0,
    every_n_frames: int = 1,
    num_frames_per_video: Optional[int] = None,
    resize_frac: float = 1.0,
    resize_to: Optional[SizeHW] = None,
    crop: Optional[CropConfig] = None,
    random_frame_offset_n: int = 0,
    seed: Optional[int] = None,
) -> None:
    """
    Extract frames from videos and save them as PNG files.

    You can choose either:
      - every_n_frames sampling (default), OR
      - num_frames_per_video (fixed number of frames uniformly sampled)

    Cropping happens before resizing.

    Args:
        video_paths:            Iterable of paths to video files.
        target_dir:             Directory where extracted frames will be saved.
        clear_target_dir:       If True, remove files in target_dir before saving.
        offset_frames:          First frame index to start extraction from (>= 0).
        every_n_frames:         Extract every Nth frame (>= 1). Ignored if num_frames_per_video is set.
        num_frames_per_video:   If set, extract exactly this many frames per video, uniformly spaced.
        resize_frac:            Scale factor for resizing frames (> 0). Use 1.0 for no resize.
                                Mutually exclusive with resize_to.
        resize_to:              Resize to fixed (height, width). Mutually exclusive with resize_frac != 1.0.
        crop:                   Optional CropConfig for fixed crop or random crop.
        random_frame_offset_n:  If > 0, each sampled frame index is jittered by a random integer in
                                [-random_frame_offset_n, +random_frame_offset_n], then clamped to valid range.
        seed:                   Optional RNG seed for reproducibility (affects random crop and random frame offset).

    Raises:
        ValueError: On invalid arguments or incompatible options.
        FileNotFoundError: If any input video file does not exist.
        RuntimeError: If a video cannot be opened.
        OSError: If clearing the target directory fails for some file.
    """
    if offset_frames < 0:
        raise ValueError("offset_frames must be >= 0")
    if every_n_frames < 1:
        raise ValueError("every_n_frames must be >= 1")
    if random_frame_offset_n < 0:
        raise ValueError("random_frame_offset_n must be >= 0")

    _validate_crop_config(crop)

    target_path = Path(target_dir)
    if clear_target_dir:
        _clear_directory(target_path)
    else:
        target_path.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    for video_path in map(Path, video_paths):
        print(f"Started with video at: {video_path}")

        if not video_path.is_file():
            raise FileNotFoundError(f"Input video not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                print(f"Warning: video has no frames (or could not read frame count): {video_path}")
                continue

            stem = video_path.stem
            base_indices = _compute_frame_indices(
                total_frames=total_frames,
                offset_frames=offset_frames,
                every_n_frames=every_n_frames,
                num_frames=num_frames_per_video,
            )

            saved_count = 0
            for base_idx in base_indices:
                # Apply random frame offset (jitter) if enabled
                if random_frame_offset_n > 0:
                    jitter = int(rng.integers(-random_frame_offset_n, random_frame_offset_n + 1))
                else:
                    jitter = 0

                frame_idx = int(np.clip(base_idx + jitter, 0, total_frames - 1))

                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    # If reading fails at some point, we stop for this video.
                    break

                frame = _apply_crop(frame, crop=crop, rng=rng)
                frame = _apply_resize(frame, resize_frac=resize_frac, resize_to=resize_to)

                save_name = f"{stem}_frame_{frame_idx:06d}_sample_{saved_count:06d}.png"
                save_path = target_path / save_name
                cv2.imwrite(str(save_path), frame)
                saved_count += 1

            print(f"Saved {saved_count} frame(s) for {video_path.name}")
        finally:
            cap.release()


if __name__ == "__main__":

    # parent directory
    parent_dir = 'path_to_directory_with_videos'
    video_paths = [parent_dir + "/" + fname for fname in os.listdir(parent_dir)]

    # video_paths = [""]  # Replace with actual video paths (or use os.listdir(pathToDirectory) in order to find all paths)
    retrieve_frames_from_videos(
        video_paths=video_paths,
        target_dir="frames",
        clear_target_dir=True,
        offset_frames=10,
        # every_n_frames=5,
        num_frames_per_video=100,  # Uncomment to save a fixed number instead of every N frames
        resize_frac=1.0,
        resize_to=(640, 640),  # fixed resize (h, w) - set to None to disable
        # crop=CropConfig(random_size=(224, 224)),  # or CropConfig(fixed_box=(x, y, w, h))
        random_frame_offset_n=2,
        seed=76,
    )
