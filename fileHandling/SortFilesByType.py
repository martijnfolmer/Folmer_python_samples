import os
import shutil
from pathlib import Path
from typing import List

"""
    This script will find all files and copy them to new directories based on their file_extension. So a new
    directory for .mp4, for .png, for .jpg, etc.
    
    It will crawl through the given directory, and check all of its sub-directories and sub-sub-directories
    and sub-sub-sub-directories and ... etc.
    
    Files which don't have a file extension should be copied to a subdirectory called 'empty'

    Author :        Martijn Folmer
    Date created :  21-02-2026
"""


def sort_files_by_type(source_dir: str, destination_dir: str) \
        -> List[str]:
    """
    Crawl through all files in a directory (including subdirectories) and copy
    files to sorted subdirectories of that type

    Args:
        source_dir: The directory to search in
        destination_dir: The directory where matching files will be copied

    Returns:
        List of paths to the copied files
    """
    source_path = Path(source_dir)
    dest_path = Path(destination_dir)

    # Create destination directory if it doesn't exist
    dest_path.mkdir(parents=True, exist_ok=True)
    copied_files = []

    # Walk through all files in source directory
    for root, dirs, files in os.walk(source_path):
        root_path = Path(root)

        for file in files:
            file_path = root_path / file

            # continue if there is not path
            suffixPath = file_path.suffix.lower()
            if len(suffixPath)==0 or suffixPath=="" or suffixPath == " ":
                suffixPath = 'empty'

            # create the subdirectory we sort it in, if needed.
            subdir_dest_path = dest_path / suffixPath
            if not subdir_dest_path.exists():
                subdir_dest_path.mkdir(parents=True, exist_ok=True)

            # Copy to destination root, handle name conflicts
            dest_file_path = subdir_dest_path / file_path.name
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


if __name__ == "__main__":

    SOURCE_DIRECTORY = "C:/Users/marti/Folmer_python_samples/Folmer_python_samples"  # Specify path
    DESTINATION_DIRECTORY = "C:/Users/marti/Downloads/Folmer_python_samples/sorted_files"  # Directory where files will be copied to

    print(f"Destination directory: {DESTINATION_DIRECTORY}")
    print("-" * 60)

    copied_files = sort_files_by_type(
        source_dir=SOURCE_DIRECTORY,
        destination_dir=DESTINATION_DIRECTORY,
    )

    print("-" * 60)
    print(f"\nTotal files copied: {len(copied_files)}")

    if copied_files:
        print("\nCopied files:")
        for file in copied_files:
            print(f"  - {file}")
