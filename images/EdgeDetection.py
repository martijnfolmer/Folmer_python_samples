import cv2

"""
    Use simple canny edge detection in order to find the edges on an image. And edge detected image can be usefull
    for pattern recognition, such as using hough circles to find circle shaped objects in an image

    Author :        Martijn Folmer
    Date created :  10-01-2026
"""


def detect_edges(input_path, output_path, low_threshold, high_threshold):
    # Load image in grayscale
    image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {input_path}")

    # Perform Canny edge detection
    edges = cv2.Canny(image, low_threshold, high_threshold)

    # Save result
    cv2.imwrite(output_path, edges)


if __name__ == "__main__":
    INPUT_IMAGE_PATH = "input_image.png"
    OUTPUT_IMAGE_PATH = "output_image.png"
    LOW_THRESHOLD = 50
    HIGH_THRESHOLD = 150

    detect_edges(
        input_path=INPUT_IMAGE_PATH,
        output_path=OUTPUT_IMAGE_PATH,
        low_threshold=LOW_THRESHOLD,
        high_threshold=HIGH_THRESHOLD
    )
