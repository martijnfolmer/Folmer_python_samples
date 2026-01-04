import cv2
import os

"""
    Batch image size converter

    What this script does:
    - Reads all readable image files from a source directory
    - Resizes them to desired size
    - Writes the converted images into a clean destination directory

    IMPORTANT NOTE:
    - creating the directory where we store the converted images is destructive, meaning that any files in it will
    be deleted. Make sure you change the directory if you want to keep the data

    Author :        Martijn Folmer
    Date created :  04-01-2026
"""


def resize_all_images(_path_to_directory, _path_to_output, _size, _verbose_n):

    if not os.path.exists(_path_to_directory):
        raise RuntimeError(f"Directory does not exists {_path_to_directory}")

    # create and clear the directory
    os.makedirs(_path_to_output, exist_ok=True)
    for fname in os.listdir(_path_to_output):
        os.remove(_path_to_output + "/" + fname)

    # Loop over all images
    allImg = os.listdir(_path_to_directory)
    nImg = len(allImg)
    for i_img, imgPath in enumerate(allImg):

        img = cv2.imread(_path_to_directory + "/" + imgPath)
        img = cv2.resize(img, _size)
        dst_path = _path_to_output + "/" + imgPath

        ok = cv2.imwrite(str(dst_path), img)
        if not ok:
            raise RuntimeError(f"Failed to write: {dst_path}")

        if i_img%_verbose_n == 0:
            print(f"We are at {i_img + 1} / {nImg}")


if __name__ == "__main__":

    PATHTOIMAGES = 'images'     # where the images are stored
    PATHTOOUTPUT = 'output'     # where we want to save the stored output
    SIZE = (1920, 1080)         # size to change the images to
    VERBOSE_N = 1               # how often we print where we are

    resize_all_images(PATHTOIMAGES, PATHTOOUTPUT, SIZE, VERBOSE_N)



