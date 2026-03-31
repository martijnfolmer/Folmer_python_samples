import cv2

from ultralytics import YOLO

"""
    Open the default (or chosen) webcam, run Ultralytics YOLO pose estimation on each frame, and show the image with
    keypoints and skeleton drawn. Press 'q' to quit.

    On first run the chosen weights file is downloaded if it is not already cached (e.g. yolo11n-pose.pt).

    NOTE: requires ultralytics (and OpenCV), same as ML_trainMnistClassificationUltralytics.py for the framework.

    Author :        Martijn Folmer
    Date created :  31-03-2026
"""


def run_webcam_pose(model_variant: str = "yolo11n-pose.pt", camera_index: int = 0, imgsz: int = 640) -> None:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open webcam (camera_index={camera_index}). "
            "Try another index or check that the camera is not in use."
        )

    model = YOLO(model_variant)

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                print("Failed to read frame from webcam; exiting.")
                break

            results = model.predict(frame_bgr, imgsz=imgsz, verbose=False)
            annotated_bgr = results[0].plot()

            cv2.imshow("Ultralytics pose (q to quit)", annotated_bgr)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    modelVariant = "yolo11n-pose.pt"
    cameraIndex = 0
    imgSz = 640

    run_webcam_pose(model_variant=modelVariant, camera_index=cameraIndex, imgsz=imgSz)
