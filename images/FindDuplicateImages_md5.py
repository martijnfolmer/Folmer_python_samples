"""
    Find duplicate images in a directory using MD5 hashes.

    What this script does:
    - Scans a directory for readable image files
    - Computes an MD5 hash per image (by default: of the raw file bytes)
    - Treats images with identical MD5 as duplicates (same content, different name allowed)
    - Creates a 'duplicate_images' subdirectory in the input directory
    - Moves all duplicate images to that subdirectory, keeping the first occurrence of each unique hash
    - Prints a summary of the results

    NOTE:
    - Using MD5 on *file bytes* detects exact file duplicates only.
      If the same image is re-encoded (e.g. different JPEG quality), the MD5 will differ.
    - If you want to detect "same pixels but different encoding", enable CANONICALIZE_PIXELS
      to hash a canonical pixel representation instead of file bytes.

    Author :        Martijn Folmer
    Date created :  21-02-2026
"""

from __future__ import annotations

from pathlib import Path
import hashlib

import cv2


def md5_file_bytes(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """
    Compute MD5 hash of a file's raw bytes.

    Parameters:
    - path: file to hash
    - chunk_size: bytes per read

    Returns:
    - hex digest string
    """
    hasher = hashlib.md5()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def md5_image_pixels(path: Path) -> str | None:
    """
    Compute MD5 hash of an image's decoded pixel content (canonicalized).

    Canonicalization approach:
    - Decode with OpenCV
    - Include shape + dtype + raw bytes in the hash

    Returns:
    - hex digest string, or None if image can't be read
    """
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None

    hasher = hashlib.md5()
    # Include metadata so different shapes don't collide even if raw bytes match coincidentally
    hasher.update(str(img.shape).encode("utf-8"))
    hasher.update(str(img.dtype).encode("utf-8"))
    hasher.update(img.tobytes())
    return hasher.hexdigest()


def find_duplicates(image_dir: Path, verbose_n: int = 1) -> tuple[list[Path], int, int]:
    """
    Find duplicate images in a directory using MD5.

    Parameters:
    - image_dir: directory containing images to check
    - verbose_n: how often to print progress (every N images)

    Returns:
    - Tuple of (list of duplicate file paths, total images processed, unique images count)
    """
    if not image_dir.exists():
        raise FileNotFoundError(f"Directory does not exist: {image_dir}")

    duplicates: list[Path] = []
    total_processed = 0

    # key = md5 hash, value = first seen file path
    first_seen_by_hash: dict[str, Path] = {}

    image_files = sorted([f for f in image_dir.iterdir() if f.is_file()])
    total_files = len(image_files)

    print(f"Scanning {total_files} files for duplicates...")

    for i, img_path in enumerate(image_files, start=1):
        # Compute hash (skip non-images when CANONICALIZE_PIXELS=True)
        if CANONICALIZE_PIXELS:
            md5_hash = md5_image_pixels(img_path)
            if md5_hash is None:
                continue  # not a readable image
        else:
            # Raw bytes hashing does not guarantee it's an image, so we quickly verify it's readable
            # to match the behavior of the original script (skip non-images).
            test_img = cv2.imread(str(img_path))
            if test_img is None:
                continue
            md5_hash = md5_file_bytes(img_path)

        total_processed += 1

        if md5_hash in first_seen_by_hash:
            duplicates.append(img_path)
        else:
            first_seen_by_hash[md5_hash] = img_path

        if verbose_n > 0 and total_processed % verbose_n == 0:
            print(f"Processed {total_processed} images")

    unique_count = len(first_seen_by_hash)
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

    IMAGE_DIR = Path(r"C:/Users/marti/Downloads/images")  # directory containing images to check
    VERBOSE_N = 10  # How often we print our progress

    # If True: compute MD5 of a canonical pixel buffer (resize disabled; uses decoded pixels).
    # If False: compute MD5 of raw file bytes (fastest, strictest "exact file" duplicate detection).
    CANONICALIZE_PIXELS = False

    if not IMAGE_DIR.exists():
        raise FileNotFoundError(f"Directory does not exist: {IMAGE_DIR}")

    print("Starting duplicate detection (MD5)...")

    duplicate_paths, total_processed, unique_count = find_duplicates(
        image_dir=IMAGE_DIR,
        verbose_n=VERBOSE_N,
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