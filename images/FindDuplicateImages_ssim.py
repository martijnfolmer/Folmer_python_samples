"""
    Find duplicate images in a directory using SSIM (Structural Similarity).

    What this script does:
    - Scans a directory for readable image files
    - Resizes decoded images to a fixed comparison size, then compares each image
      to already-kept uniques using SSIM
    - Treats pairs with SSIM at or above a threshold as duplicates (first file kept)
    - Creates a "duplicate_images" subdirectory in the input directory
    - Moves duplicate images into that subdirectory
    - Prints a summary of the results

    NOTE:
    - Requires scikit-image: pip install scikit-image
    - SSIM is perceptual, not exact: the threshold can cause false positives/negatives.
    - Resizing before SSIM helps match re-exports at different resolutions but can
      confuse very different crops or compositions.
    - Runtime grows with the number of unique images (each new file is compared to
      every kept unique)

    Author :        Martijn Folmer
    Date created :  11-05-2026
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity


def _resize_interpolation(src_w: int, src_h: int, dst_w: int, dst_h: int) -> int:
    if dst_w * dst_h < src_w * src_h:
        return cv2.INTER_AREA
    return cv2.INTER_LINEAR


def prepare_image_for_ssim(
    img: np.ndarray,
    comparison_size: tuple[int, int],
) -> np.ndarray | None:
    """
    Convert an OpenCV-loaded image to float64 in [0, 1], RGB or single-channel,
    resized to comparison_size (width, height).

    Returns None if the array cannot be used (e.g. unexpected channel count).
    """
    if img.ndim == 2:
        gray = img
        dst_w, dst_h = comparison_size
        interp = _resize_interpolation(gray.shape[1], gray.shape[0], dst_w, dst_h)
        resized = cv2.resize(gray, comparison_size, interpolation=interp)
        return (resized.astype(np.float64) / 255.0)

    if img.ndim != 3:
        return None

    channels = img.shape[2]
    if channels == 4:
        bgr = img[:, :, :3]
    elif channels == 3:
        bgr = img
    elif channels == 1:
        gray = img[:, :, 0]
        dst_w, dst_h = comparison_size
        interp = _resize_interpolation(gray.shape[1], gray.shape[0], dst_w, dst_h)
        resized = cv2.resize(gray, comparison_size, interpolation=interp)
        return (resized.astype(np.float64) / 255.0)
    else:
        return None

    dst_w, dst_h = comparison_size
    interp = _resize_interpolation(bgr.shape[1], bgr.shape[0], dst_w, dst_h)
    resized = cv2.resize(bgr, comparison_size, interpolation=interp)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float64) / 255.0


def ssim_between(a: np.ndarray, b: np.ndarray) -> float:
    """Compute SSIM between two same-shaped float images in [0, 1]."""
    if a.ndim == 2:
        return float(
            structural_similarity(a, b, data_range=1.0),
        )
    return float(
        structural_similarity(a, b, data_range=1.0, channel_axis=-1),
    )


def find_duplicates(
    image_dir: Path,
    verbose_n: int = 1,
    ssim_threshold: float = 0.99,
    comparison_size: tuple[int, int] = (256, 256),
) -> tuple[list[Path], int, int]:
    """
    Find duplicate images in a directory using SSIM.

    Parameters:
    - image_dir: directory containing images to check
    - verbose_n: how often to print progress (every N images)
    - ssim_threshold: SSIM value at or above which a file is considered a duplicate
    - comparison_size: (width, height) both images are resized to before SSIM

    Returns:
    - Tuple of (list of duplicate file paths, total images processed, unique images count)
    """
    if not image_dir.exists():
        raise FileNotFoundError(f"Directory does not exist: {image_dir}")

    duplicates: list[Path] = []
    total_processed = 0
    unique_tensors: list[np.ndarray] = []

    image_files = sorted([f for f in image_dir.iterdir() if f.is_file()])
    total_files = len(image_files)

    print(f"Scanning {total_files} files for duplicates...")

    for img_path in image_files:
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue

        tensor = prepare_image_for_ssim(img, comparison_size)
        if tensor is None:
            continue

        total_processed += 1

        is_duplicate = False
        for ref in unique_tensors:
            if ssim_between(tensor, ref) >= ssim_threshold:
                duplicates.append(img_path)
                is_duplicate = True
                break

        if not is_duplicate:
            unique_tensors.append(tensor)

        if verbose_n > 0 and total_processed % verbose_n == 0:
            print(f"Processed {total_processed} images")

    unique_count = len(unique_tensors)
    return duplicates, total_processed, unique_count


def move_duplicates(image_dir: Path, duplicate_paths: list[Path]) -> None:
    """
    Move duplicate images to a subdirectory.

    Parameters:
    - image_dir: directory containing the images
    - duplicate_paths: list of duplicate file paths to move
    """
    if not duplicate_paths:
        return

    duplicates_dir = image_dir / "duplicate_images"
    duplicates_dir.mkdir(exist_ok=True)

    for dup_path in duplicate_paths:
        dest_path = duplicates_dir / dup_path.name

        # Handle name conflicts if duplicate_images already contains a file with same name
        counter = 1
        while dest_path.exists():
            stem = dup_path.stem
            suffix = dup_path.suffix
            dest_path = duplicates_dir / f"{stem}_{counter}{suffix}"
            counter += 1

        dup_path.rename(dest_path)


if __name__ == "__main__":

    IMAGE_DIR = Path(r"C:/Users/martijn.folmer/Downloads/testSomeImg")  # directory containing images to check
    VERBOSE_N = 10  # How often we print our progress

    # SSIM at or above this value (against some already-kept unique) marks a duplicate.
    # Lower values are looser (more files flagged as duplicates).
    SSIM_THRESHOLD = 0.99

    # Both images are resized to this (width, height) before SSIM. Change if you need
    # more or less detail in the comparison (larger is slower but can separate similar shots).
    COMPARISON_SIZE = (256, 256)

    if not IMAGE_DIR.exists():
        raise FileNotFoundError(f"Directory does not exist: {IMAGE_DIR}")

    print("Starting duplicate detection (SSIM)...")

    duplicate_paths, total_processed, unique_count = find_duplicates(
        image_dir=IMAGE_DIR,
        verbose_n=VERBOSE_N,
        ssim_threshold=SSIM_THRESHOLD,
        comparison_size=COMPARISON_SIZE,
    )

    if duplicate_paths:
        print(f"\nMoving {len(duplicate_paths)} duplicate(s) to duplicate_images subdirectory...")
        move_duplicates(IMAGE_DIR, duplicate_paths)
    else:
        print("\nNo duplicates found.")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    duplicates_count = len(duplicate_paths)

    print(f"Total images processed: {total_processed}")
    print(f"Unique images found: {unique_count}")
    print(f"Duplicates found: {duplicates_count}")

    if duplicate_paths:
        print(f"\nDuplicate files moved:")
        for dup_path in duplicate_paths:
            print(f"  - {dup_path.name}")

    print("=" * 60)
    print("Done!")
