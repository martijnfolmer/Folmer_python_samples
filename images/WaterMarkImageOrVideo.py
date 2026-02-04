import cv2
import numpy as np
from typing import Literal, Tuple, Optional, Union, cast

"""
    Watermark an image or video with alpha/opacity blending.

    Author :        Martijn Folmer
    Date created :  04-02-2026
"""

# Data Types
Corner = Literal["br", "bl", "tr", "tl", "abs"]
Pos = Tuple[Corner, int, int]
ImageArray = np.ndarray


def clamp01(x: float) -> float:
    """Clamp a float into [0, 1]."""
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def resize_watermark_to_fraction_of_width(
    watermark: ImageArray,
    target_frame_width: int,
    scale: float,
) -> ImageArray:
    """
    Resize watermark such that its width becomes `scale * target_frame_width`.
    scale is clamped to [0, 1]. If scale == 0 or watermark has invalid size, returns original watermark.
    """
    s = clamp01(scale)
    if s == 0.0:
        return watermark

    wh, ww = watermark.shape[:2]
    if ww <= 0 or wh <= 0 or target_frame_width <= 0:
        return watermark

    target_w = max(1, int(round(target_frame_width * s)))
    if target_w == ww:
        return watermark

    factor = target_w / float(ww)
    target_h = max(1, int(round(wh * factor)))
    return cv2.resize(watermark, (target_w, target_h), interpolation=cv2.INTER_AREA)


def compute_position(frame_shape: Tuple[int, int, int], wm_shape: Tuple[int, int, int], pos: Pos) -> Tuple[int, int]:
    """
    frame_shape = (height, width, channels)
    wm_shape = watermark height and width
    pos: ("br"/"bl"/"tr"/"tl"/"abs", offset_x, offset_y)
    Returns (x,y) top-left.
    """
    h, w = frame_shape[:2]
    wh, ww = wm_shape[:2]
    corner, ox, oy = pos

    if corner == "abs":
        return max(0, int(ox)), max(0, int(oy))

    if corner == "br":
        x = w - ww - ox
        y = h - wh - oy
    elif corner == "bl":
        x = ox
        y = h - wh - oy
    elif corner == "tr":
        x = w - ww - ox
        y = oy
    elif corner == "tl":
        x = ox
        y = oy
    else:
        raise ValueError("pos corner must be one of: br, bl, tr, tl, abs")

    return max(0, int(x)), max(0, int(y))


def overlay_watermark(
    frame: ImageArray,
    watermark: ImageArray,
    x: int,
    y: int,
    opacity: float = 0.35,
) -> ImageArray:
    """
    Blend watermark over frame at (x,y) with given opacity.
    - watermark can be BGR or BGRA (alpha channel supported).
    - opacity is clamped to [0, 1].
    """
    if frame is None or watermark is None:
        raise ValueError("frame/watermark is None")

    op = clamp01(opacity)

    h, w = frame.shape[:2]
    wh, ww = watermark.shape[:2]

    # Clip to frame bounds
    if x >= w or y >= h:
        return frame  # watermark completely outside
    x2 = min(w, x + ww)
    y2 = min(h, y + wh)

    # Corresponding watermark region
    wm_x2 = x2 - x
    wm_y2 = y2 - y
    wm_roi = watermark[:wm_y2, :wm_x2]

    roi = frame[y:y2, x:x2]

    # Ensure ROI is BGR
    if roi.ndim == 2:
        roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)

    # Build alpha mask
    if wm_roi.ndim != 3 or wm_roi.shape[2] not in (3, 4):
        raise ValueError("Watermark must have 3 (BGR) or 4 (BGRA) channels.")

    if wm_roi.shape[2] == 4:
        wm_bgr = wm_roi[:, :, :3].astype(np.float32)
        alpha = wm_roi[:, :, 3].astype(np.float32) / 255.0
        alpha = (alpha * op)[..., None]  # (H, W, 1)
    else:
        wm_bgr = wm_roi.astype(np.float32)
        alpha = np.full((wm_roi.shape[0], wm_roi.shape[1], 1), op, dtype=np.float32)

    base = roi.astype(np.float32)
    blended = base * (1.0 - alpha) + wm_bgr * alpha
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    frame_out = frame.copy()
    frame_out[y:y2, x:x2] = blended
    return frame_out


def add_watermark_to_image(
    input_path: str,
    watermark_path: str,
    output_path: str,
    pos: Pos = ("br", 20, 20),
    scale: float = 0.2,
    opacity: float = 0.35,
) -> None:

    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {input_path}")

    wm = cv2.imread(watermark_path, cv2.IMREAD_UNCHANGED)  # keep alpha if present
    if wm is None:
        raise FileNotFoundError(f"Could not read watermark: {watermark_path}")

    h, w = img.shape[:2]
    wm_scaled = resize_watermark_to_fraction_of_width(wm, w, scale)

    x, y = compute_position(cast(Tuple[int, int, int], img.shape), cast(Tuple[int, int, int], wm_scaled.shape), pos)
    out = overlay_watermark(img, wm_scaled, x, y, opacity=opacity)

    cv2.imwrite(output_path, out)
    print("Saved:", output_path)


def add_watermark_to_video(
    input_path: str,
    watermark_path: str,
    output_path: str,
    pos: Pos = ("br", 20, 20),
    scale: float = 0.15,
    opacity: float = 0.30,
) -> None:

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {input_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    wm = cv2.imread(watermark_path, cv2.IMREAD_UNCHANGED)
    if wm is None:
        cap.release()
        raise FileNotFoundError(f"Could not read watermark: {watermark_path}")

    # Resize ONCE, outside the loop => no accidental double scaling & faster.
    wm_scaled = resize_watermark_to_fraction_of_width(wm, frame_w, scale)
    x, y = compute_position((frame_h, frame_w, 3), cast(Tuple[int, int, int], wm_scaled.shape), pos)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps if fps > 0 else 30.0, (frame_w, frame_h))

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_wm = overlay_watermark(frame, wm_scaled, x, y, opacity=opacity)
            out.write(frame_wm)
    finally:
        cap.release()
        out.release()

    print("Saved:", output_path)


if __name__ == "__main__":
    watermark_png = "images/megaman_single_filled.png"  # can be PNG with alpha, or JPG

    add_watermark_to_image(
        input_path="images/img1.jpeg",
        watermark_path=watermark_png,
        output_path="../readme_img/WatermarkImageOrVideo.jpg",
        pos=("abs", 300, 30),
        scale=0.25,
        opacity=0.35,
    )

    add_watermark_to_video(
        input_path="C:/Users/marti/Folmer_python_samples/Folmer_python_samples/video/videos/clip_5s_to_12_5s.mp4",
        watermark_path=watermark_png,
        output_path="output_watermarked.mp4",
        pos=("tr", 30, 30),
        scale=0.5,
        opacity=0.25,
    )
