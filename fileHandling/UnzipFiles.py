import zipfile
from pathlib import Path

"""
    This script will unzip all zipped files in a given directory
    
    It will not delete the original zipped files

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

    for zip_file in zip_files:
        try:
            print(f"Extracting: {zip_file.name}")
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                # Extract to a folder with the same name as the zip file (without .zip extension)
                extract_to = zip_file.stem
                zip_ref.extractall(extract_to)
                print(f"  [OK] Successfully extracted to: {extract_to}")
        except zipfile.BadZipFile:
            print(f"  [ERROR] {zip_file.name} is not a valid zip file or is corrupted.")
        except Exception as e:
            print(f"  [ERROR] Error extracting {zip_file.name}: {str(e)}")

    print("\nExtraction complete!")


if __name__ == "__main__":
    pathToDir = 'Insert_the_path_to_unizip_here'
    unzip_all_files(pathToDir)
