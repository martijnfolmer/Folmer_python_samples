import cv2
import numpy as np
from pathlib import Path

"""
    Replace a color in an image with another color (BGR), save the result,
    and optionally show original and modified images side-by-side.

    Gradient-style replacement:
    - For pixels within the tolerance range of FROM_COLOR_BGR,
      we keep the *same per-channel offset* relative to FROM_COLOR_BGR,
      but applied around TO_COLOR_BGR.

      In other words:
      new_pixel = TO_COLOR_BGR + (old_pixel - FROM_COLOR_BGR)

    Notes:
    - OpenCV uses BGR color order by default.
    - Tolerance is applied per-channel using cv2.inRange().
    - Output is clipped to [0, 255].

    Author :        Martijn Folmer
    Date created :  31-01-2026
"""


def replace_color_bgr_with_offset(
    img_bgr: np.ndarray,
    from_color_bgr: tuple[int, int, int],
    to_color_bgr: tuple[int, int, int],
    tolerance: int = 20,
) -> np.ndarray:
    """
    Replace pixels close to from_color_bgr by shifting them around to_color_bgr,
    preserving the original pixel's offset from from_color_bgr.

    Parameters:
    - img_bgr: Input image (BGR)
    - from_color_bgr: Target color to replace (B, G, R)
    - to_color_bgr: Replacement base color (B, G, R)
    - tolerance: Per-channel tolerance (0-255). Higher = more pixels affected.

    Returns:
    - New image with gradient-style color replacement
    """
    if img_bgr is None:
        raise ValueError("img_bgr is None")

    if img_bgr.ndim == 2:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)

    img_out = img_bgr.copy()

    from_bgr_u8 = np.array(from_color_bgr, dtype=np.uint8)
    to_bgr_i16 = np.array(to_color_bgr, dtype=np.int16)

    # Build mask of pixels "close enough" to FROM color
    lower = np.clip(from_bgr_u8.astype(np.int16) - tolerance, 0, 255).astype(np.uint8)
    upper = np.clip(from_bgr_u8.astype(np.int16) + tolerance, 0, 255).astype(np.uint8)
    mask = cv2.inRange(img_bgr, lower, upper)  # 255 where pixel is in range

    # Compute offset and apply it around TO color (int16 to avoid uint8 wraparound)
    img_i16 = img_bgr.astype(np.int16)
    from_i16 = from_bgr_u8.astype(np.int16)

    # new_pixel = TO + (old - FROM)
    shifted = to_bgr_i16 + (img_i16 - from_i16)
    shifted = np.clip(shifted, 0, 255).astype(np.uint8)

    # Apply only where mask is true
    img_out[mask > 0] = shifted[mask > 0]

    return img_out


def show_side_by_side(
    img_left: np.ndarray,
    img_right: np.ndarray,
    window_name: str = "Original (left) vs Modified (right)",
    resize_to: tuple[int, int] | None = None,
) -> None:
    """
    Show two images next to each other in one window.

    Parameters:
    - img_left: Left image (BGR)
    - img_right: Right image (BGR)
    - window_name: Window title
    - resize_to: Optional (width, height) to resize both images before showing
    """
    if resize_to is not None:
        img_left = cv2.resize(img_left, resize_to, interpolation=cv2.INTER_AREA)
        img_right = cv2.resize(img_right, resize_to, interpolation=cv2.INTER_AREA)
    else:
        if img_left.shape[:2] != img_right.shape[:2]:
            h, w = img_left.shape[:2]
            img_right = cv2.resize(img_right, (w, h), interpolation=cv2.INTER_AREA)

    stacked = np.hstack([img_left, img_right])
    cv2.imshow(window_name, stacked)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":

    # Input image path
    INPUT_PATH = Path("images/megaman_single.png")

    # Output image path
    OUTPUT_PATH = Path("output_recolored.png")

    # Color to replace (B, G, R)  -> OpenCV default order
    FROM_COLOR_BGR = (229, 136, 30)  # blue

    # Replacement base color (B, G, R)
    TO_COLOR_BGR = (31, 229, 51)  # green

    # Tolerance per channel (0-255). Increase if your source color has shading/compression artifacts.
    TOLERANCE = 110

    # Show original and new image next to each other
    SHOW_PREVIEW = True

    # Optional resize for preview window (width, height) or None
    PREVIEW_RESIZE_TO = None  # e.g. (900, 600)

    img = cv2.imread(str(INPUT_PATH), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {INPUT_PATH}")

    img_recolored = replace_color_bgr_with_offset(
        img_bgr=img,
        from_color_bgr=FROM_COLOR_BGR,
        to_color_bgr=TO_COLOR_BGR,
        tolerance=TOLERANCE,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(OUTPUT_PATH), img_recolored)
    if not ok:
        raise IOError(f"Failed to write output image: {OUTPUT_PATH}")

    print(f"Saved recolored image to: {OUTPUT_PATH}")

    if SHOW_PREVIEW:
        show_side_by_side(img, img_recolored, resize_to=PREVIEW_RESIZE_TO)
