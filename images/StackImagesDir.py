import cv2
import numpy as np
import os
from pathlib import Path

"""
    Stack images from multiple directories based on matching filenames.

    What this script does:
    - Takes multiple directories containing images
    - For images that share the same filename across directories, stacks them together
    - Can stack horizontally (left-to-right) or vertically (top-to-bottom)
    - Adds the directory name as a label above each image
    - Resizes images to a common size if needed
    - Saves the stacked result to an output directory
    
    Special case:
    - If only one directory is provided, stacks ALL images in that directory together

    IMPORTANT NOTE:
    - Creating the output directory is destructive, meaning any existing files in it will
      be deleted. Make sure you change the directory if you want to keep the data

    Author :        Martijn Folmer
    Date created :  06-01-2026
"""


def ensure_empty_dir(dir_path: Path) -> None:
    """Create directory if needed and remove all files inside it."""
    dir_path.mkdir(parents=True, exist_ok=True)

    for p in dir_path.iterdir():
        if p.is_file():
            p.unlink()


def add_label_to_image(
    img: np.ndarray,
    label: str,
    font_scale: float = 0.8,
    font_thickness: int = 2,
    label_height: int = 40,
    bg_color: tuple = (40, 40, 40),
    text_color: tuple = (255, 255, 255),
) -> np.ndarray:
    """
    Add a label bar above the image with the given text.
    
    Parameters:
    - img: The image to add the label to
    - label: Text to display
    - font_scale: Size of the font
    - font_thickness: Thickness of the font
    - label_height: Height of the label bar in pixels
    - bg_color: Background color of the label bar (BGR)
    - text_color: Color of the text (BGR)
    
    Returns:
    - Image with label bar on top
    """
    h, w = img.shape[:2]
    
    # Create label bar
    label_bar = np.full((label_height, w, 3), bg_color, dtype=np.uint8)
    
    # Calculate text position (centered)
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
    text_x = (w - text_w) // 2
    text_y = (label_height + text_h) // 2
    
    # Draw the text
    cv2.putText(label_bar, label, (text_x, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
    
    # Stack label on top of image
    return np.vstack([label_bar, img])


def get_image_files(directory: Path) -> dict[str, Path]:
    """
    Get all image files from a directory.
    
    Returns:
    - Dictionary mapping filename (stem + extension) to full path
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
    images = {}
    
    for p in directory.iterdir():
        if p.is_file() and p.suffix.lower() in image_extensions:
            images[p.name.lower()] = p
    
    return images


def stack_images_from_directories(
    input_dirs: list[str | Path],
    output_dir: str | Path,
    direction: str = "horizontal",
    resize_to: tuple[int, int] | None = None,
    add_labels: bool = True,
    label_height: int = 40,
    pad_color: tuple[int, int, int] = (0, 0, 0),
    verbose_n: int = 1,
) -> None:
    """
    Stack images from multiple directories based on matching filenames.
    
    Parameters:
    - input_dirs: List of directories containing images
    - output_dir: Directory where stacked images will be saved
    - direction: "horizontal" or "vertical"
    - resize_to: (width, height) to resize each image to, or None for auto
    - add_labels: Whether to add directory name labels above each image
    - label_height: Height of the label bar in pixels
    - pad_color: BGR color for padding if needed
    - verbose_n: Print progress every N images
    """
    if direction not in ("horizontal", "vertical"):
        raise ValueError("direction must be 'horizontal' or 'vertical'")
    
    if not input_dirs:
        raise ValueError("input_dirs is empty")
    
    # Convert to Path objects
    input_dirs = [Path(d) for d in input_dirs]
    output_dir = Path(output_dir)
    
    # Validate directories exist
    for d in input_dirs:
        if not d.exists():
            raise FileNotFoundError(f"Directory does not exist: {d}")
    
    # Get images from each directory
    dir_images = {}
    for d in input_dirs:
        dir_images[d] = get_image_files(d)
        if not dir_images[d]:
            print(f"Warning: No images found in {d}")
    
    # Handle single directory case: stack all images together
    if len(input_dirs) == 1:
        single_dir = input_dirs[0]
        images = dir_images[single_dir]
        
        if len(images) < 2:
            raise ValueError("Need at least 2 images to stack when using a single directory")
        
        ensure_empty_dir(output_dir)
        
        # Load all images
        loaded_images = []
        for name, path in sorted(images.items()):
            img = cv2.imread(str(path))
            if img is None:
                print(f"Warning: Could not read image: {path}")
                continue
            
            # Use filename as label
            if add_labels:
                img = add_label_to_image(img, path.stem, label_height=label_height)
            
            loaded_images.append(img)
        
        if not loaded_images:
            raise ValueError("No valid images found to stack")
        
        # Determine common size
        if resize_to is None:
            min_w = min(img.shape[1] for img in loaded_images)
            min_h = min(img.shape[0] for img in loaded_images)
            resize_to = (min_w, min_h)
        
        # Resize all images
        resized = []
        for img in loaded_images:
            if (img.shape[1], img.shape[0]) != resize_to:
                img = cv2.resize(img, resize_to, interpolation=cv2.INTER_AREA)
            resized.append(img)
        
        # Stack
        if direction == "horizontal":
            stacked = np.hstack(resized)
        else:
            stacked = np.vstack(resized)
        
        # Save
        output_path = output_dir / "stacked_all.png"
        cv2.imwrite(str(output_path), stacked)
        print(f"Saved stacked image with {len(resized)} images to: {output_path}")
        return
    
    # Multiple directories: find common filenames
    all_filenames = set()
    for images in dir_images.values():
        all_filenames.update(images.keys())
    
    # Find filenames that exist in all directories
    common_filenames = all_filenames.copy()
    for images in dir_images.values():
        common_filenames &= set(images.keys())
    
    if not common_filenames:
        print("Warning: No common filenames found across all directories.")
        print("Processing files that exist in at least 2 directories...")
        
        # Find files that exist in at least 2 directories
        filename_counts = {}
        for images in dir_images.values():
            for name in images:
                filename_counts[name] = filename_counts.get(name, 0) + 1
        
        common_filenames = {name for name, count in filename_counts.items() if count >= 2}
        
        if not common_filenames:
            raise ValueError("No images found that exist in multiple directories")
    
    ensure_empty_dir(output_dir)
    
    n_images = len(common_filenames)
    print(f"Processing {n_images} images...")
    
    for i_img, filename in enumerate(sorted(common_filenames)):
        images_to_stack = []
        
        for d in input_dirs:
            if filename not in dir_images[d]:
                continue
            
            path = dir_images[d][filename]
            img = cv2.imread(str(path))
            
            if img is None:
                print(f"Warning: Could not read image: {path}")
                continue
            
            # Ensure 3-channel BGR
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            
            # Add directory label
            if add_labels:
                label = d.name
                img = add_label_to_image(img, label, label_height=label_height)
            
            images_to_stack.append(img)
        
        if len(images_to_stack) < 2:
            continue
        
        # Determine common size for this set
        if resize_to is None:
            target_w = min(img.shape[1] for img in images_to_stack)
            target_h = min(img.shape[0] for img in images_to_stack)
        else:
            target_w, target_h = resize_to
        
        # Resize all to common size
        resized = []
        for img in images_to_stack:
            if (img.shape[1], img.shape[0]) != (target_w, target_h):
                img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
            resized.append(img)
        
        # Stack images
        if direction == "horizontal":
            stacked = np.hstack(resized)
        else:
            stacked = np.vstack(resized)
        
        # Save the stacked image
        output_path = output_dir / filename
        ok = cv2.imwrite(str(output_path), stacked)
        if not ok:
            print(f"Warning: Failed to write: {output_path}")
        
        if i_img % verbose_n == 0:
            print(f"Processed {i_img + 1} / {n_images}: {filename}")
    
    print(f"Done! Stacked images saved to: {output_dir}")


if __name__ == "__main__":
    
    # List of directories containing images to stack
    # Images with matching filenames across directories will be stacked together
    # If only ONE directory is provided, ALL images in that directory will be stacked
    INPUT_DIRS = [
        "images_dir1",
        "images_dir2",
        "images_dir3",
    ]
    
    # Where to save the stacked images
    OUTPUT_DIR = "stacked_output"
    
    # "horizontal" (left-to-right) or "vertical" (top-to-bottom)
    DIRECTION = "horizontal"
    
    # Resize each image to this size before stacking, or None to auto-resize to smallest common size
    RESIZE_TO = None  # e.g. (640, 480) or None
    
    # Whether to add directory name labels above each image
    ADD_LABELS = True
    
    # Height of the label bar in pixels
    LABEL_HEIGHT = 40
    
    # Padding color (B, G, R)
    PAD_COLOR = (0, 0, 0)
    
    # Print progress every N images
    VERBOSE_N = 1
    
    stack_images_from_directories(
        input_dirs=INPUT_DIRS,
        output_dir=OUTPUT_DIR,
        direction=DIRECTION,
        resize_to=RESIZE_TO,
        add_labels=ADD_LABELS,
        label_height=LABEL_HEIGHT,
        pad_color=PAD_COLOR,
        verbose_n=VERBOSE_N,
    )

