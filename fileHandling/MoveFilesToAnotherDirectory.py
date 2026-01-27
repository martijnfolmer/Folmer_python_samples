import os
import shutil

"""
    This script moves all files and subdirectories
    from a source directory to a destination directory.
    
    The source directory will be kept as an empty directory at its original location

    Author :        Martijn Folmer
    Date created :  24-01-2026
"""


def move_directory_contents(source_dir, destination_dir):
    if not os.path.isdir(source_dir):
        raise NotADirectoryError(f"Source directory does not exist: {source_dir}")

    if os.path.abspath(source_dir) == os.path.abspath(destination_dir):
        raise ValueError("Source and destination directories must be different.")

    # Ensure destination directory exists
    os.makedirs(destination_dir, exist_ok=True)

    moved_items = []
    for item_name in os.listdir(source_dir):
        source_path = os.path.join(source_dir, item_name)
        destination_path = os.path.join(destination_dir, item_name)

        # Move file or directory
        shutil.move(source_path, destination_path)
        moved_items.append(destination_path)

    # Print the summary of what we have done
    print("Directory move complete")
    print(f"Source directory:      {source_dir}")
    print(f"Destination directory: {destination_dir}")
    print(f"Items moved:           {len(moved_items)}")

    if moved_items:
        for path in moved_items:
            print(f"  MOVED -> {path}")
    else:
        print("No items found to move.")


if __name__ == "__main__":
    SOURCE_DIR = "images"
    DESTINATION_DIR = "images_moved"

    move_directory_contents(
        source_dir=SOURCE_DIR,
        destination_dir=DESTINATION_DIR
    )
