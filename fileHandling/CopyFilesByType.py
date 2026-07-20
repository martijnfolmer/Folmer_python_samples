import os
import shutil
import threading
import concurrent.futures
from pathlib import Path
from typing import List, Tuple, Optional

"""
    This script will find all files with a certain extension (given in a list), and copy those to a new location
    It will crawl through the given directory, and check all of its sub-directories and sub-sub-directories
    and sub-sub-sub-directories and ... etc.

    Matching file extensions is performed in a case-insensitive way

    Author :        Martijn Folmer
    Date created :  30-01-2026
"""


def get_matching_files(source_dir: str, file_extensions: list) -> List[str]:
    """
    Crawls through a directory and returns a list of paths to files that match
    the extensions, which case-insensitive

    Args:
        source_dir: The directory to search in
        file_extensions: A list of file extensions to search for

    Returns:
        List of full paths to the matching files
    """
    # Normalize file extensions
    ext_tuple = tuple(
        ext.lower() if ext.startswith('.') else f".{ext.lower()}"
        for ext in file_extensions
    )

    matched_files = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith(ext_tuple):
                src_file = os.path.join(root, file)
                matched_files.append(src_file)

    return matched_files


def _copy_worker(src_file: str, filename: str, dest_path: Path, used_names: set, name_lock: threading.Lock) -> Tuple[
    Optional[str], Optional[str]]:
    """Module-level worker function to handle thread-safe naming and file copying."""
    with name_lock:
        dest_name = filename
        if dest_name in used_names:
            stem, suffix = os.path.splitext(filename)
            counter = 1
            dest_name = f"{stem}_{counter}{suffix}"
            while dest_name in used_names:
                counter += 1
                dest_name = f"{stem}_{counter}{suffix}"
        used_names.add(dest_name)

    dest_file_path = dest_path / dest_name

    try:
        shutil.copy2(src_file, dest_file_path)
        return str(dest_file_path), None
    except Exception as e:
        return None, f"Error copying {src_file}: {str(e)}"


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
    dest_path = Path(destination_dir)

    # Create destination directory if it doesn't exist
    dest_path.mkdir(parents=True, exist_ok=True)
    used_names = {f.name for f in dest_path.iterdir() if f.is_file()}
    name_lock = threading.Lock()
    copied_files = []

    # Get the list of all matching file paths
    matching_files = get_matching_files(source_dir, file_extensions)

    # Run copies concurrently using a ThreadPool
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(_copy_worker, src, os.path.basename(src), dest_path, used_names, name_lock)
            for src in matching_files
        ]

        for future in concurrent.futures.as_completed(futures):
            dest_result, error = future.result()
            if error:
                print(error)
            else:
                copied_files.append(dest_result)

    return copied_files


if __name__ == "__main__":

    SOURCE_DIRECTORY = "Path_to_directory_we_want_to_crawl_through"  # Specify path
    FILE_EXTENSIONS = ["md", ".png"]  # file types we want to find (case insensitive)
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