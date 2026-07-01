"""
    This script will find all files in a directory and its subdirectories, and its subsubdirectories, etc.
    and return their paths as a list

    Author :        Martijn Folmer
    Date created :  01-07-2026
"""


import os


def listAllFiles(source_path, listToIgnore):
    allFiles = []
    for root, dirs, files in os.walk(source_path):
        dirs[:] = [d for d in dirs if d not in listToIgnore]
        for file in files:
            file_path = os.path.join(root, file)
            path_parts = os.path.normpath(file_path).split(os.sep)
            if any(ignore in path_parts for ignore in listToIgnore):
                continue
            allFiles.append(file_path)
    return allFiles


if __name__=="__main__":

    # This is the location we want to list all files from
    dirToCheck = "C:/Users/martijn.folmer/Folmer_python_samples"
    listToIgnore = [".git", "__pycache__", ".idea"] # list of directories to ignore

    # run the function
    allFiles = listAllFiles(dirToCheck, listToIgnore)

    # print all files found
    for file in allFiles:
        print(file)
    
    print(f"Found {len(allFiles)} files in: {dirToCheck}")
