import os
import zipfile

from typing import List

"""
    This script will zip all files with a certain file extension found in a directory into a single zip file

    Author :        Martijn Folmer
    Date created :  20-07-2026
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
    # Normalize file extensions, for case_insensitivity
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


def zipFilesOfType(_pathToDirToScan, _pathToZip, _fileTypes):
    """zip all found files with certain file extension insto a single"""

    all_files = get_matching_files(_pathToDirToScan, _fileTypes)
    print(f"Number of files found : {len(all_files)}")

    with zipfile.ZipFile(_pathToZip, "w") as zipf:
        for fname in all_files:
            zipf.write(fname, fname.split("/")[-1])

if __name__ == "__main__":

    allFileTypes = [".png", ".jpg", ".jpeg"] # example of filetypes, if we want to zip all image related ones
    dirToScan = "Directory_to_scan"
    zipName = "Location_and_name_of_where_zip_file_lives.zip"

    zipFilesOfType(dirToScan, zipName, allFileTypes)
