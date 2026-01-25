"""
    Image letterboxer

    What this script does:
    - Reads all readable image files from a source directory
    - Resizes each image to fit inside a desired (width, height) while keeping aspect ratio
    - Fills the remaining area (letterbox) with a chosen BGR color
    - Writes the letterboxed images into a clean destination directory

    Supported formats depend on your OpenCV build, but commonly include:
    jpg, png, bmp, tif, webp

    IMPORTANT NOTE:
    - creating the directory where we store the letterboxed images is destructive, meaning that any files in it will
    be deleted. Make sure you change the directory if you want to keep the data

    Author :        Martijn Folmer
    Date created :  25-01-2026
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def ensure_empty_dir(dir_path: Path) -> None:
    """Create directory if needed and remove all files inside it."""
    dir_path.mkdir(parents=True, exist_ok=True)

    for p in dir_path.iterdir():
        if p.is_file():
            p.unlink()


def letterbox_image(
    image: np.ndarray,
    desired_size: tuple[int, int],
    color_bgr: tuple[int, int, int],
) -> tuple[np.ndarray, int, int]:
    """
    Resize an image to fit into the desired size while keeping its aspect ratio.
    The remaining areas are filled with color_bgr.

    Parameters:
    image (numpy.ndarray): Input image in BGR format.
    desired_size (tuple): Desired (width, height) for the output image.
    color_bgr (tuple): Fill color in BGR format, e.g. (112, 112, 112)

    Returns:
    tuple: (letterboxed_image, x_offset, y_offset)
    """
    # Get original dimensions
    orig_h, orig_w = image.shape[:2]
    new_w, new_h = desired_size

    # Compute scaling factor (keeping aspect ratio)
    scale = min(new_w / orig_w, new_h / orig_h)
    resized_w = int(orig_w * scale)
    resized_h = int(orig_h * scale)

    # Resize the image with the computed scale
    resized_image = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

    # Create a new image filled with the chosen color (BGR)
    letterboxed = np.full((new_h, new_w, 3), color_bgr, dtype=np.uint8)

    # Compute offsets to center the image on the background
    x_offset = (new_w - resized_w) // 2
    y_offset = (new_h - resized_h) // 2

    # Place the resized image on the background
    letterboxed[y_offset : y_offset + resized_h, x_offset : x_offset + resized_w] = resized_image

    return letterboxed, x_offset, y_offset


def letterbox_images(
    src_dir: Path,
    dst_dir: Path,
    desired_size: tuple[int, int],
    color_bgr: tuple[int, int, int],
    output_ext: str,
    verbose_n: int = 1,
) -> None:
    """
    Letterbox all images in src_dir to desired_size and write them to dst_dir.

    Parameters:
    - src_dir: directory containing source images
    - dst_dir: directory where letterboxed images are written
    - desired_size: (width, height) output size
    - color_bgr: fill color in BGR
    - output_ext: file extension without dot (e.g. 'jpg', 'png', 'webp')
    - verbose_n: how often we print progress
    """
    output_ext = output_ext.lower().lstrip(".")
    ensure_empty_dir(dst_dir)

    paths = [p for p in sorted(src_dir.iterdir()) if p.is_file()]
    n_img = len(paths)

    for i_path, src_path in enumerate(paths):
        img = cv2.imread(str(src_path))
        if img is None:
            continue  # skip non-images

        lb, _, _ = letterbox_image(
            image=img,
            desired_size=desired_size,
            color_bgr=color_bgr,
        )

        dst_path = dst_dir / f"{src_path.stem}.{output_ext}"
        ok = cv2.imwrite(str(dst_path), lb)
        if not ok:
            raise RuntimeError(f"Failed to write: {dst_path}")

        if (i_path % verbose_n) == 0:
            print(f"We are at {i_path + 1} / {n_img} files")


if __name__ == "__main__":
    SRC_DIR = Path(r"images")                 # where the images come from
    DST_DIR = Path(r"letterboxed_images")     # where the images are going

    # Output size is (width, height)
    DESIRED_SIZE = (640, 640)

    # Letterbox fill color (B, G, R)
    LETTERBOX_COLOR_BGR = (112, 112, 112)

    # Output image format (do not insert dot/period/small black circle!!!!!!)
    OUTPUT_EXT = "png"  # e.g. "jpg", "png", "webp"

    VERBOSE_N = 3  # How often we print our progress

    if not SRC_DIR.exists():
        raise FileNotFoundError(f"Source directory does not exist: {SRC_DIR}")

    print("Start the letterboxing!!!!")

    letterbox_images(
        src_dir=SRC_DIR,
        dst_dir=DST_DIR,
        desired_size=DESIRED_SIZE,
        color_bgr=LETTERBOX_COLOR_BGR,
        output_ext=OUTPUT_EXT,
        verbose_n=VERBOSE_N,
    )

    print("We are done!")
