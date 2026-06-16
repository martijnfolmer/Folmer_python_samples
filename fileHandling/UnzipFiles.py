import zipfile
from pathlib import Path
import os

"""
    This script will unzip all zipped files in a given directory

    It will optionally delete the original zipped files depending on the flag

    Author :        Martijn Folmer
    Date created :  30-01-2026
"""


def unzip_all_files(_pathToDir):
    """Unzip all zip files in the current directory."""
    current_dir = Path(_pathToDir)
    zip_files = list(current_dir.glob('*.zip'))

    if not zip_files:
        print("No zip files found in the current directory.")
        return

    print(f"Found {len(zip_files)} zip file(s) to extract.")

    allSuccesUnzips = []
    allFailedUnzips = []
    for zip_file in zip_files:
        try:
            print(f"Extracting: {zip_file.name}")
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:

                # FIXED: Use the '/' operator to join pathlib.Path objects
                extract_to = current_dir / zip_file.stem
                zip_ref.extractall(extract_to)
                print(f"  [OK] Successfully extracted to: {extract_to}")
                allSuccesUnzips.append(str(zipfile))

        except zipfile.BadZipFile:
            print(f"  [ERROR] {zip_file.name} is not a valid zip file or is corrupted.")
            allFailedUnzips.append(zip_file.name)
        except Exception as e:
            print(f"  [ERROR] Error extracting {zip_file.name}: {str(e)}")
            allFailedUnzips.append(zip_file.name)

    if allFailedUnzips:
        print("\nWe failed to unzip the following directories: ")
        for failed_zip in allFailedUnzips:
            print(f"    {failed_zip}")

    print("\nExtraction complete!")


if __name__ == "__main__":

    pathToDir = '<insert path here, with the zip files we want to unzip>'
    unzip_all_files(pathToDir)