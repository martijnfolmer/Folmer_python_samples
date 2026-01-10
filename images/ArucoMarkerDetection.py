"""
    Find aruco markers in an image.
    Aruco markers are a sort of QR codes, but they don't contain information except for their identification. Usefull
    for robotics and computer vision application, in depth measurement, or mapping, or autonomous driving.

    Important note, I've noticed aruco markers don't work so well when the marker is bend, or partially visible. So
    make sure in your application that it can be visible

    More information about aruco markers can be found here :
    https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html

    Author :        Martijn Folmer
    Date created :  10-01-2026
"""


import cv2


def detect_aruco_markers(image_path, output_path, dict_name):
    # Load image
    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    # ---- ArUco dictionary selection ----
    aruco = cv2.aruco
    if dict_name == "DICT_4X4_50":
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    elif dict_name == "DICT_4X4_100":
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_100)
    elif dict_name == "DICT_5X5_100":
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_100)
    elif dict_name == "DICT_6X6_250":
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    elif dict_name == "DICT_7X7_1000":
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_7X7_1000)
    else:
        raise ValueError(
            f"Unknown dict_name={dict_name}. "
            "Try one of: DICT_4X4_50, DICT_4X4_100, DICT_5X5_100, DICT_6X6_250, DICT_7X7_1000"
        )

    # Detector parameters (OpenCV 4.7+ style)
    params = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, params)

    # Detect markers
    corners, ids, rejected = detector.detectMarkers(image_bgr)

    # Draw detections + print IDs
    vis = image_bgr.copy()
    if ids is not None and len(ids) > 0:
        aruco.drawDetectedMarkers(vis, corners, ids)
        print(f"Detected {len(ids)} marker(s): {ids.flatten().tolist()}")
    else:
        print("No markers detected.")

    # Save annotated image
    if not cv2.imwrite(output_path, vis):
        raise IOError(f"Failed to write output image: {output_path}")


if __name__ == "__main__":
    IMAGE_PATH = "singlemarkersoriginal.jpg"     # location of the input image
    OUTPUT_PATH = "arucoMarkerDetection.png"  # where we store our result
    DICT_NAME = "DICT_6X6_250"        # must match your printed markers
    MARKER_LENGTH_METERS = 0.05      # only used for pose estimation (if enabled)

    detect_aruco_markers(
        image_path=IMAGE_PATH,
        output_path=OUTPUT_PATH,
        dict_name=DICT_NAME,
    )
