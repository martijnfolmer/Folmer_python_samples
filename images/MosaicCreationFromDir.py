import os
import random
import numpy as np
import cv2
from pathlib import Path

"""

    What this script does:
    - Reads all readable image files from a source directory
    - Selects random ones and resizes them to put them in a grid, mosaic style
    - saves the mosaics in an output directory

    IMPORTANT NOTE:
    - creating the directory where we store the created images is destructive, meaning that any files in it will
    be deleted. Make sure you change the directory if you want to keep the data


    Author :        Martijn Folmer
    Date created :  24-01-2026
"""


def ensure_empty_dir(dir_path: Path) -> None:
    """Create directory if needed and remove all files inside it."""
    dir_path.mkdir(parents=True, exist_ok=True)

    for p in dir_path.iterdir():
        if p.is_file():
            p.unlink()


def load_images_from_folder(folder):
    return [
        os.path.join(folder, fname)
        for fname in os.listdir(folder)
        if fname.lower().endswith((".jpg", ".jpeg", ".png"))
    ]


def create_mosaic(image_paths, grid_w, grid_h, img_size):
    tot_needed = grid_w * grid_h
    assert tot_needed <= len(image_paths), (
        f"Need at least {tot_needed} images, got {len(image_paths)}"
    )

    random.shuffle(image_paths)
    img_idx = 0

    for y in range(grid_h):
        for x in range(grid_w):
            img = cv2.imread(image_paths[img_idx])
            img = cv2.resize(img, img_size)
            img = cv2.rectangle(
                img,
                (0, 0),
                (img.shape[1], img.shape[0]),
                (255, 255, 255),
                5,
            )

            row = img if x == 0 else np.concatenate([row, img], axis=1)
            img_idx += 1

        mosaic = row if y == 0 else np.concatenate([mosaic, row], axis=0)

    return mosaic


if __name__ == "__main__":

    PATH_TO_IMAGES = "images"
    OUTPUT_DIR = "mosaics"

    GRID_W = 4
    GRID_H = 4
    IMG_SIZE = (640 * 2, 640)

    NUM_MOSAICS = 5  # how many mosaics to generate

    ensure_empty_dir(Path(OUTPUT_DIR))

    all_images = load_images_from_folder(PATH_TO_IMAGES)

    for i in range(NUM_MOSAICS):
        mosaic = create_mosaic(
            image_paths=all_images.copy(),
            grid_w=GRID_W,
            grid_h=GRID_H,
            img_size=IMG_SIZE,
        )

        out_path = os.path.join(OUTPUT_DIR, f"mosaic_{i:03d}.png")
        cv2.imwrite(out_path, mosaic)

        print(f"Saved {out_path} ({mosaic.shape[1]} x {mosaic.shape[0]})")
