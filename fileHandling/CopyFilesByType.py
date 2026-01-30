import os
import shutil
from pathlib import Path
from typing import List

"""
    This script will find all files with a certain extension (given in a list), and copy those to a new location
    It will crawl through the given directory, and check all of its sub-directories and sub-sub-directories
    and sub-sub-sub-directories and ... etc.
    
    Matching file extensions is performed in a case-insensitive way

    Author :        Martijn Folmer
    Date created :  30-01-2026
"""


def find_files_by_type(source_dir: str, file_extensions: list, destination_dir: str) \
        -> List[str]:
    """
    Crawl through all files in a directory (including subdirectories) and copy
    files of a certain type to a new directory.

    Args:
        source_dir: The directory to search in
        file_extensions: A list with the file extension to search for (e.g., '.mp4', '.jpg', '.txt')
                       Can include or exclude the dot
        destination_dir: The directory where matching files will be copied

    Returns:
        List of paths to the copied files
    """
    source_path = Path(source_dir)
    dest_path = Path(destination_dir)

    # Normalize file extension (ensure it starts with a dot)
    for i_ext, extension in enumerate(file_extensions):
        if not extension.startswith('.'):
            extension = '.' + extension
        extension = extension.lower()
        file_extensions[i_ext] = extension

    # Create destination directory if it doesn't exist
    dest_path.mkdir(parents=True, exist_ok=True)

    copied_files = []

    # Walk through all files in source directory
    for root, dirs, files in os.walk(source_path):
        root_path = Path(root)

        for file in files:
            file_path = root_path / file

            # Check if file has the desired extension (case-insensitive)
            if file_path.suffix.lower() in file_extensions:
                # Copy to destination root, handle name conflicts
                dest_file_path = dest_path / file_path.name
                counter = 1
                original_dest = dest_file_path
                while dest_file_path.exists():
                    stem = original_dest.stem
                    suffix = original_dest.suffix
                    dest_file_path = dest_path / f"{stem}_{counter}{suffix}"
                    counter += 1

                # Copy the file
                try:
                    shutil.copy2(file_path, dest_file_path)
                    copied_files.append(str(dest_file_path))
                    print(f"Copied: {file_path} -> {dest_file_path}")
                except Exception as e:
                    print(f"Error copying {file_path}: {str(e)}")

    return copied_files


def main():

    SOURCE_DIRECTORY = "C:/Users/martijn.folmer/Folmer_python_samples"  # Specify path
    FILE_EXTENSIONS = ["md", ".png"]  # file types we want to find
    DESTINATION_DIRECTORY = "copied_files"  # Directory where files will be copied to

    print(f"Searching for {FILE_EXTENSIONS} files in: {SOURCE_DIRECTORY}")
    print(f"Destination directory: {DESTINATION_DIRECTORY}")
    print("-" * 60)

    copied_files = find_files_by_type(
        source_dir=SOURCE_DIRECTORY,
        file_extensions=FILE_EXTENSIONS,
        destination_dir=DESTINATION_DIRECTORY,
    )

    print("-" * 60)
    print(f"\nTotal files copied: {len(copied_files)}")

    if copied_files:
        print("\nCopied files:")
        for file in copied_files:
            print(f"  - {file}")

if __name__ == "__main__":
    main()
