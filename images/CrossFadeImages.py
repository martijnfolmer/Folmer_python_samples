"""
    Cross-fade (alpha blend) two images.

    What this script does:
    - Loads two images with OpenCV (BGR)
    - Resizes the second image to match the first if their sizes differ
    - Blends them with a configurable mix: result = img1 * (1 - blend_ratio) + img2 * blend_ratio
    - Writes the blended image to disk

    blend_ratio 0.0 is fully the first image, 1.0 is fully the second, 0.5 is an even mix.

    Author :        Martijn Folmer
    Date created :  24-03-2026
"""

from __future__ import annotations

from pathlib import Path

import cv2


def cross_fade_images(
    image1_path: Path,
    image2_path: Path,
    output_path: Path,
    blend_ratio: float = 0.5,
) -> None:
    """
    Blend two images and save the result.

    Parameters:
    - image1_path: path to the first image (base size used if resizing is needed)
    - image2_path: path to the second image
    - output_path: where to write the blended image
    - blend_ratio: weight of the second image in [0.0, 1.0]
    """
    if not 0.0 <= blend_ratio <= 1.0:
        raise ValueError(f"blend_ratio must be between 0 and 1, got {blend_ratio}")

    img1 = cv2.imread(str(image1_path))
    if img1 is None:
        raise FileNotFoundError(f"Could not load image: {image1_path}")

    img2 = cv2.imread(str(image2_path))
    if img2 is None:
        raise FileNotFoundError(f"Could not load image: {image2_path}")

    h1, w1 = img1.shape[:2]
    if img2.shape[:2] != (h1, w1):
        img2 = cv2.resize(img2, (w1, h1), interpolation=cv2.INTER_LINEAR)

    w1_blend = 1.0 - blend_ratio
    w2_blend = blend_ratio
    blended = cv2.addWeighted(img1, w1_blend, img2, w2_blend, 0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(output_path), blended)
    if not ok:
        raise RuntimeError(f"Failed to write: {output_path}")

    print(f"Blended image saved to {output_path}")


if __name__ == "__main__":
    IMAGE1_PATH = Path(r"images") / "image_a.png"       # first image
    IMAGE2_PATH = Path(r"images") / "image_b.png"       # second image
    OUTPUT_PATH = Path(r"readme_img") / "CrossFadeImages.png"

    BLEND_RATIO = 0.5  # 0.5 = equal mix of both images

    if not IMAGE1_PATH.exists():
        raise FileNotFoundError(f"Source image does not exist: {IMAGE1_PATH}")
    if not IMAGE2_PATH.exists():
        raise FileNotFoundError(f"Source image does not exist: {IMAGE2_PATH}")

    print("Start cross-fade!!!!")

    cross_fade_images(
        image1_path=IMAGE1_PATH,
        image2_path=IMAGE2_PATH,
        output_path=OUTPUT_PATH,
        blend_ratio=BLEND_RATIO,
    )

    print("We are done!")
