"""
    Batch image format converter using OpenCV.

    What this script does:
    - Reads all readable image files from a source directory
    - Converts them to a chosen output image format (e.g. jpg, png, webp)
    - Writes the converted images into a clean destination directory

    Supported formats depend on your OpenCV build, but commonly include:
    jpg, png, bmp, tif, webp

    IMPORTANT NOTE:
    - creating the directory where we store the converted images is destructive, meaning that any files in it will
    be deleted. Make sure you change the directory if you want to keep the data

    Author :        Martijn Folmer
    Date created :  03-01-2026
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
    output_ext: str,
    verbose_n: 1,
) -> None:
    """
    Convert all images in src_dir to output_ext and write them to dst_dir.

    Parameters:
    - src_dir: directory containing source images
    - dst_dir: directory where converted images are written
    - output_ext: file extension without dot (e.g. 'jpg', 'png', 'webp')
    """
    output_ext = output_ext.lower().lstrip(".")
    ensure_empty_dir(dst_dir)

    n_img = len(os.listdir(src_dir))
    for i_path, src_path in enumerate(sorted(src_dir.iterdir())):
        if not src_path.is_file():
            continue

        img = cv2.imread(str(src_path))
        if img is None:
            continue  # skip non-images

        dst_path = dst_dir / f"{src_path.stem}.{output_ext}"
        ok = cv2.imwrite(str(dst_path), img)
        if not ok:
            raise RuntimeError(f"Failed to write: {dst_path}")

        if i_path % verbose_n == 0:
            print(f"We are at {i_path + 1} / {n_img} images")


if __name__ == "__main__":
    SRC_DIR = Path(r"images")               # where the images come from
    DST_DIR = Path(r"converted_images")     # where the converted images are going

    # Output image format (do not insert dot/period/small black circle!!!!!!)
    OUTPUT_EXT = "jpg"  # e.g. "jpg", "png", "webp"

    VERBOSE_N = 3      # How often we print our progress

    if not SRC_DIR.exists():
        raise FileNotFoundError(f"Source directory does not exist: {SRC_DIR}")

    print("Start the conversion!!!!")

    convert_images(
        src_dir=SRC_DIR,
        dst_dir=DST_DIR,
        output_ext=OUTPUT_EXT,
        verbose_n=VERBOSE_N,              # how often we print where we are
    )

    print("We are done!")
