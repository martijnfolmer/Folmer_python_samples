import os
import shutil

"""
    This script removes or moves files from a directory if their filename contains a specified string.
    It will skip any subdirectories it finds in the main directory

    Features:
    - Optional case-sensitive or case-insensitive matching
    - Optional safe mode: move files instead of deleting them

    IMPORTANT NOTE: when not using safe-mode, it will delete the files permanently, making it destructive

    Author :        Martijn Folmer
    Date created :  23-01-2026
"""


def clean_directory_by_name(
    target_dir,
    name_substring,
    case_sensitive=False,
    move_instead_of_delete=True,
    move_dir='deleted'
):
    if not os.path.isdir(target_dir):
        raise NotADirectoryError(f"Target directory does not exist: {target_dir}")

    if not name_substring:
        raise ValueError("name_substring must be a non-empty string.")

    # Normalize substring for comparison
    if not case_sensitive:
        name_substring_cmp = name_substring.lower()
    else:
        name_substring_cmp = name_substring

    # Create directory where we move the files to
    if move_instead_of_delete:
        os.makedirs(move_dir, exist_ok=True)

    deleted_files = []
    moved_files = []

    # Iterate through directory contents
    for filename in os.listdir(target_dir):
        file_path = os.path.join(target_dir, filename)

        # Skip subdirectories
        if not os.path.isfile(file_path):
            continue

        filename_cmp = filename if case_sensitive else filename.lower()

        if name_substring_cmp in filename_cmp:
            if move_instead_of_delete:
                destination = os.path.join(move_dir, filename)
                shutil.move(file_path, destination)
                moved_files.append(destination)
            else:
                os.remove(file_path)
                deleted_files.append(file_path)

    # Print the summary
    print("Directory cleanup complete")
    print(f"Target directory: {target_dir}")
    print(f"Search string:    '{name_substring}'")
    print(f"Case sensitive:   {case_sensitive}")
    print(f"Files matched:    {len(deleted_files) + len(moved_files)}")

    if move_instead_of_delete:
        print(f"Files moved to:   {move_dir}")
        for path in moved_files:
            print(f"  MOVED   -> {path}")
    else:
        for path in deleted_files:
            print(f"  DELETED -> {path}")

    if not deleted_files and not moved_files:
        print("No matching files found.")


if __name__ == "__main__":
    TARGET_DIR = "name_of_target_directory"
    NAME_SUBSTRING = "000"
    CASE_SENSITIVE = False

    MOVE_INSTEAD_OF_DELETE = True
    MOVE_DIR = "delete"

    clean_directory_by_name(
        target_dir=TARGET_DIR,
        name_substring=NAME_SUBSTRING,
        case_sensitive=CASE_SENSITIVE,
        move_instead_of_delete=MOVE_INSTEAD_OF_DELETE,
        move_dir=MOVE_DIR
    )
