import os
import shutil
from tqdm import tqdm
import filecmp

"""
    This script will synchronize a folder to a backup folder
    
    If delete is set to true, we also delete any files in the target location that are not in the source location
    
    created by : Martijn Folmer
    on 15-06-26
"""

# Check if the source and target directories exist
def check_directories(_source_directory, _target_directory):
    # make sure the source directory exists
    if not os.path.exists(_source_directory):
        print("ERROR: Source directory does not exists, so we can't back it up")

    # make sure we have a target directory to backup to, if not, create it
    os.makedirs(_target_directory, exist_ok=True)

# Get the paths of all files and directories in a given directory (does not include that given directory)
def get_all_files_and_directories_in_path(_directory_path):
    all_files = []
    with tqdm(desc="Going through everything in source directory") as pbar:
        for root, dirs, files in os.walk(_directory_path):
            for directory in dirs:
                all_files.append(os.path.join(root, directory))
            for file in files:
                all_files.append(os.path.join(root, file))
            pbar.update(1)

    return all_files

# If we set delete to true, we will delete any file in the target directory which does not have an equivalent in the source
def delete_new_files(_source_directory, _target_directory):
    all_files_in_target_directory = get_all_files_and_directories_in_path(_target_directory)
    with tqdm(total=len(all_files_in_target_directory), desc="Deleting files", unit="file") as pbar:
        for target_path in all_files_in_target_directory:
            source_path = os.path.join(_source_directory, os.path.relpath(target_path, _target_directory))
            if not os.path.exists(source_path):
                # Set the description of the progress bar
                pbar.set_description(f"Processing '{target_path}'")
                print(f"\nDeleting {target_path}")

                # is directory, shutil.rmtree
                if os.path.isdir(target_path):
                    shutil.rmtree(target_path)
                # is file, os.remove
                else:
                    os.remove(target_path)

            # Update the progress bar
            pbar.update(1)

def sync(_source_directory, _target_directory, _delete):

    # make sure directories exists
    check_directories(_source_directory, _target_directory)

    all_source_files = get_all_files_and_directories_in_path(_source_directory)
    total_files = len(all_source_files)

    with tqdm(total = total_files, desc="Going over each file", unit="file_n") as pbar:
        for source_path in all_source_files:
            # Equivalent path in the target directory
            target_path = os.path.join(_target_directory, os.path.relpath(source_path, _source_directory))

            # if it is a directory, and does not exist in target, create it
            if os.path.isdir(source_path):
                os.makedirs(target_path, exist_ok=True)
            # if it is a file, and does not exists in target, we copy it there
            else:
                if not os.path.exists(target_path) or not filecmp.cmp(source_path, target_path, shallow=False):
                    shutil.copy2(source_path, target_path)

            pbar.update(1)

    if delete:
        delete_new_files(_source_directory, _target_directory)

if __name__=="__main__":

    delete = True # if set to true, we delete any files in target directory which don't have an equivalent in previous
    source_directory = "path_to_directory_to_backup" # the directory to backup from (will make script exit if it does not exist)
    target_directory = "path_to_directory_to_store_the_backup" # the directory to backup to

    sync(source_directory, target_directory, delete)
