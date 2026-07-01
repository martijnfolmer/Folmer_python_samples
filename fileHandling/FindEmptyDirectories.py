"""
    This script will find all empty directories and subdirectories and subsubdirectories, return their paths and
    optionally delete them

    IMPORTANT : deleting them in this way is destructive, recovery may not be possible, proceed with caution

    Author :        Martijn Folmer
    Date created :  01-07-2026
"""


import os

from sympy import false


def findEmptyDir(source_path, deleteEmptyDir = false):
    allEmptyDir = []
    for root, dirs, files in os.walk(source_path):
        if os.path.isdir(root):
            if os.path.isdir(root) and len(os.listdir(root)) == 0:
                allEmptyDir.append(root)

                # delete empty directory if we have set that
                if deleteEmptyDir:
                    os.rmdir(root)
    return allEmptyDir


if __name__=="__main__":

    # This is the location we want to check for empty directories
    dirToCheck = "C:/Users/martijn.folmer/Folmer_python_samples"
    toDelete = False

    # run the function
    allEmptyDir = findEmptyDir(dirToCheck, toDelete)

    print("Found the following empty directories :")
    for dir in allEmptyDir:
        print(dir)

    if toDelete:
        print("We deleted all of the empty directories")