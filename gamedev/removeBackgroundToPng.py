from pathlib import Path
import cv2
import numpy as np

"""
    Takes a directory of images, and removes a given background by using a flood fill. Saves the result with the 
    transparant background as a png. Only removes if it is connected to the image border (flood fill)

    This script does not clear the target directory

    Author :        Martijn Folmer
    Date created :  25-01-26
"""


def removeBackgroundPng(
    img: np.ndarray,
    background_rgb: tuple[int, int, int],
    tolerance: int
) -> np.ndarray:
    """
    Removes background color only if it's connected to the image border (edge-flood behavior).
    Returns BGRA image.
    """

    bg_r, bg_g, bg_b = background_rgb

    # Normalize to BGR + alpha
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if img.shape[2] == 3:
        bgr = img
        alpha = np.full((img.shape[0], img.shape[1]), 255, dtype=np.uint8)
    elif img.shape[2] == 4:
        bgr = img[:, :, :3]
        alpha = img[:, :, 3].copy()
    else:
        raise ValueError(f"Unsupported channel count: {img.shape[2]}")

    # Color range in OpenCV BGR order
    lower = np.array(
        [max(0, bg_b - tolerance),
         max(0, bg_g - tolerance),
         max(0, bg_r - tolerance)],
        dtype=np.uint8
    )
    upper = np.array(
        [min(255, bg_b + tolerance),
         min(255, bg_g + tolerance),
         min(255, bg_r + tolerance)],
        dtype=np.uint8
    )

    # Candidate background pixels
    mask_bg = cv2.inRange(bgr, lower, upper)

    # Connected components on candidate mask
    # labels: 0 = non-bg, 1..N = bg components
    num_labels, labels = cv2.connectedComponents(mask_bg, connectivity=8)

    if num_labels <= 1:
        # No background candidates found
        return np.dstack((bgr, alpha))

    h, w = mask_bg.shape

    # Collect labels that touch any border where mask_bg is 255
    top = labels[0, :][mask_bg[0, :] == 255]
    bottom = labels[h - 1, :][mask_bg[h - 1, :] == 255]
    left = labels[:, 0][mask_bg[:, 0] == 255]
    right = labels[:, w - 1][mask_bg[:, w - 1] == 255]

    border_labels = np.unique(np.concatenate([top, bottom, left, right]))
    border_labels = border_labels[border_labels != 0]  # exclude non-bg

    if border_labels.size == 0:
        # Background color exists only inside, not connected to edge -> remove nothing
        return np.dstack((bgr, alpha))

    # Pixels belonging to edge-connected background components
    edge_bg = np.isin(labels, border_labels)

    # Make only those pixels transparent
    alpha[edge_bg] = 0

    return np.dstack((bgr, alpha))


if __name__ == "__main__":

    input_dir = Path("images")              # path with all of the images
    output_dir = Path("output_images")      # path where we want to store our results

    background_rgb = (255, 255, 255)  # (R, G, B)
    tolerance = 5                       # tolerance, how far the color can be off by this and still be removed, usefull for removing jpg artifacting

    supported_exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}

    # create target output if it does not already exist
    output_dir.mkdir(parents=True, exist_ok=True)

    for img_path in input_dir.iterdir():
        if not img_path.is_file():
            continue
        if img_path.suffix.lower() not in supported_exts:
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Failed to read: {img_path.name}")
            continue

        try:
            out = removeBackgroundPng(img, background_rgb, tolerance)
            out_path = output_dir / (img_path.stem + ".png")
            ok = cv2.imwrite(str(out_path), out)
            if ok:
                print(f"Processed: {img_path.name}")
            else:
                print(f"Failed to write: {out_path.name}")

        except Exception as e:
            print(f"Failed: {img_path.name} ({e})")

    print("Done.")
