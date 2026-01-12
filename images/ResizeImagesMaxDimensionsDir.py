"""
    Batch image resizer with aspect ratio preservation.

    What this script does:
    - Reads all readable image files from a source directory
    - Checks if width or height exceeds the maximum allowed dimensions
    - Resizes images that exceed the limits while maintaining aspect ratio
    - Writes the processed images into a clean destination directory

    Images that are already within the limits are copied unchanged.

    IMPORTANT NOTE:
    - Creating the directory where we store the resized images is destructive, meaning that any files in it will
    be deleted. Make sure you change the directory if you want to keep the data

    Author :        Martijn Folmer
    Date created :  12-01-2026
"""

from __future__ import annotations

from pathlib import Path
import cv2
import os


def ensure_empty_dir(dir_path: Path) -> None:
    """Create directory if needed and remove all files inside it."""
    dir_path.mkdir(parents=True, exist_ok=True)

    for p in dir_path.iterdir():
        if p.is_file():
            p.unlink()


def calculate_new_dimensions(
    current_width: int,
    current_height: int,
    max_width: int,
    max_height: int,
) -> tuple[int, int]:
    """
    Calculate new dimensions that fit within max bounds while preserving aspect ratio.

    Returns the new (width, height) tuple. If no resize is needed, returns the original dimensions.
    """
    if current_width <= max_width and current_height <= max_height:
        return current_width, current_height

    # Calculate scale factors for both dimensions
    width_ratio = max_width / current_width
    height_ratio = max_height / current_height

    # Use the smaller ratio to ensure both dimensions fit within limits
    scale = min(width_ratio, height_ratio)

    new_width = int(current_width * scale)
    new_height = int(current_height * scale)

    return new_width, new_height


def resize_images_max_dimensions(
    src_dir: Path,
    dst_dir: Path,
    max_width: int,
    max_height: int,
    verbose_n: int = 1,
) -> None:
    """
    Resize all images in src_dir that exceed max dimensions and write them to dst_dir.

    Parameters:
    - src_dir: directory containing source images
    - dst_dir: directory where resized images are written
    - max_width: maximum allowed width in pixels
    - max_height: maximum allowed height in pixels
    - verbose_n: print progress every N images
    """
    ensure_empty_dir(dst_dir)

    n_img = len(os.listdir(src_dir))
    resized_count = 0
    skipped_count = 0

    for i_path, src_path in enumerate(sorted(src_dir.iterdir())):
        if not src_path.is_file():
            continue

        img = cv2.imread(str(src_path))
        if img is None:
            continue  # skip non-images

        current_height, current_width = img.shape[:2]
        new_width, new_height = calculate_new_dimensions(
            current_width, current_height, max_width, max_height
        )

        # Check if resize is needed
        if new_width != current_width or new_height != current_height:
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
            resized_count += 1
        else:
            skipped_count += 1

        dst_path = dst_dir / src_path.name
        ok = cv2.imwrite(str(dst_path), img)
        if not ok:
            raise RuntimeError(f"Failed to write: {dst_path}")

        if (i_path + 1) % verbose_n == 0:
            print(f"Processed {i_path + 1} / {n_img} images")

    print(f"\nDone! Resized: {resized_count}, Already within limits: {skipped_count}")


if __name__ == "__main__":
    SRC_DIR = Path(r"images")               # where the images come from
    DST_DIR = Path(r"images_resized")       # where the resized images are going

    # Maximum allowed dimensions (images exceeding either will be resized)
    MAX_WIDTH = 1920    # maximum width in pixels
    MAX_HEIGHT = 1080   # maximum height in pixels

    VERBOSE_N = 1       # how often we print our progress

    if not SRC_DIR.exists():
        raise FileNotFoundError(f"Source directory does not exist: {SRC_DIR}")

    print(f"Resizing images larger than {MAX_WIDTH}x{MAX_HEIGHT} while preserving aspect ratio...")

    resize_images_max_dimensions(
        src_dir=SRC_DIR,
        dst_dir=DST_DIR,
        max_width=MAX_WIDTH,
        max_height=MAX_HEIGHT,
        verbose_n=VERBOSE_N,
    )

