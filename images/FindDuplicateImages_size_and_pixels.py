"""
    Find duplicate images in a directory.

    What this script does:
    - Reads all readable image files from a directory
    - Checks if images are duplicates by comparing resolution and pixel content
    - Creates a 'duplicate_images' subdirectory in the input directory
    - Moves all duplicate images to that subdirectory, keeping the first occurrence of each unique image
    - Prints a summary of the results

    NOTE: this only works on images that are the exact same, but have a different name. So images that are for
    example cropped or augmented are not considered duplicates

    Author :        Martijn Folmer
    Date created :  17-02-2026
"""

from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np


def find_duplicates(image_dir: Path, verbose_n: int = 1) -> tuple[list[Path], int, int]:
    """
    Find duplicate images in a directory.

    Parameters:
    - image_dir: directory containing images to check
    - verbose_n: how often to print progress (every N images)

    Returns:
    - Tuple of (list of duplicate file paths, total images processed, unique images count)
    """
    if not image_dir.exists():
        raise FileNotFoundError(f"Directory does not exist: {image_dir}")

    # Dictionary to store unique images grouped by dimensions: key = (width, height), value = list of (img_array, file_path)
    unique_images = {}
    duplicates = []
    total_processed = 0

    # Get all image files
    image_files = sorted([f for f in image_dir.iterdir() if f.is_file()])
    total_files = len(image_files)

    print(f"Scanning {total_files} files for duplicates...")

    for i, img_path in enumerate(image_files):
        # Try to load image
        img = cv2.imread(str(img_path))
        if img is None:
            continue  # skip non-images

        total_processed += 1
        height, width = img.shape[:2]
        dim_key = (width, height)

        # Check against all unique images with same dimensions
        is_duplicate = False
        if dim_key in unique_images:
            # Check against all images with same dimensions
            for img_array, first_path in unique_images[dim_key]:
                if np.array_equal(img, img_array):
                    # Duplicate found!
                    duplicates.append(img_path)
                    is_duplicate = True
                    break

        if not is_duplicate:
            # This is a unique image, store it
            if dim_key not in unique_images:
                unique_images[dim_key] = []
            unique_images[dim_key].append((img, img_path))

        if total_processed % verbose_n == 0:
            print(f"Processed {total_processed} images")

    # Count unique images (sum of all lists in unique_images)
    unique_count = sum(len(images) for images in unique_images.values())
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

    # Create duplicate_images subdirectory
    duplicates_dir = image_dir / "duplicate_images"
    duplicates_dir.mkdir(exist_ok=True)

    # Move duplicates
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


    IMAGE_DIR = Path(r"C:/Users/martijn.folmer/Downloads/newImages/fatbikes")  # directory containing images to check

    VERBOSE_N = 10  # How often we print our progress

    if not IMAGE_DIR.exists():
        raise FileNotFoundError(f"Directory does not exist: {IMAGE_DIR}")

    print("Starting duplicate detection...")

    # Find duplicates
    duplicate_paths, total_processed, unique_count = find_duplicates(
        image_dir=IMAGE_DIR,
        verbose_n=VERBOSE_N,
    )

    # Move duplicates
    if duplicate_paths:
        print(f"\nMoving {len(duplicate_paths)} duplicate(s) to duplicate_images subdirectory...")
        move_duplicates(IMAGE_DIR, duplicate_paths)
    else:
        print("\nNo duplicates found.")

    # Print summary
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

