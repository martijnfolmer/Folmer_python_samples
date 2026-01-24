"""
    Batch image format converter using OpenCV.

    What this script does:
    - Reads all readable image files from a source directory
    - Converts them to grayscale
    - Writes the converted images into a clean destination directory

    IMPORTANT NOTE:
    - creating the directory where we store the converted images is destructive, meaning that any files in it will
    be deleted. Make sure you change the directory if you want to keep the data
    - The script uses cv2, which reads images in a BGR format. This is our assumption when transforming them to
    grayscale. If you for some reason suspect it is reading it in a different format, you may need a different
    color converter

    Author :        Martijn Folmer
    Date created :  23-01-2026
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


def convert_images(
    src_dir: Path,
    dst_dir: Path,
    verbose_n: 1,
) -> None:
    """
    Convert all images in src_dir to grayscale and write them to dst_dir.

    Parameters:
    - src_dir: directory containing source images
    - dst_dir: directory where converted images are written
    """
    ensure_empty_dir(dst_dir)

    n_img = len(os.listdir(src_dir))
    for i_path, src_path in enumerate(sorted(src_dir.iterdir())):
        if not src_path.is_file():
            continue

        img = cv2.imread(str(src_path))
        if img is None:
            continue  # skip non-images

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        dst_path = dst_dir / f"{src_path.name}"
        ok = cv2.imwrite(str(dst_path), img)
        if not ok:
            raise RuntimeError(f"Failed to write: {dst_path}")

        if i_path % verbose_n == 0:
            print(f"We are at {i_path + 1} / {n_img} images")


if __name__ == "__main__":
    SRC_DIR = Path(r"images")               # where the images come from
    DST_DIR = Path(r"converted_images")     # where the converted images are going

    VERBOSE_N = 3      # How often we print our progress

    if not SRC_DIR.exists():
        raise FileNotFoundError(f"Source directory does not exist: {SRC_DIR}")

    print("Start the conversion!!!!")

    convert_images(
        src_dir=SRC_DIR,
        dst_dir=DST_DIR,
        verbose_n=VERBOSE_N,              # how often we print where we are
    )

    print("We are done!")
