"""
    This script concatenates multiple video files into a single output video,
    appending them one after another in the order provided.

    It uses cv2 to:
    - Read each input video frame-by-frame
    - Optionally resize frames to a target resolution
    - Write all frames into a single output video file

    A tqdm-based progress bar is used to show progress

    Requirements:
    - All input videos should ideally have the same FPS and codec (different FPS especially messes things up)
    - Resolution mismatches can be handled via resizing in __main__

    Author :        Martijn Folmer
    Date created :  03-01-2026
"""

import os
import cv2
from tqdm import tqdm


def concatenate_videos(
    video_paths,
    output_path,
    resize=False,
    target_size=(1280, 720),
):
    if not video_paths:
        raise ValueError("No videos provided")

    # Open first video to get base properties (FPS and such, this will result in problems if it is not the same for each
    # video)
    first_cap = cv2.VideoCapture(video_paths[0])
    if not first_cap.isOpened():
        raise IOError(f"Cannot open video: {video_paths[0]}")

    fps = first_cap.get(cv2.CAP_PROP_FPS)

    if resize:
        width, height = target_size
    else:
        width = int(first_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(first_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    first_cap.release()

    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    try:
        for video_path in video_paths:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise IOError(f"Cannot open video: {video_path}")

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            for _ in tqdm(
                range(frame_count),
                desc=f"Processing {os.path.basename(video_path)}",
            ):
                ret, frame = cap.read()
                if not ret:
                    break

                if resize:
                    frame = cv2.resize(frame, (width, height))
                else:
                    # Safety resize if dimensions mismatch
                    if frame.shape[1] != width or frame.shape[0] != height:
                        frame = cv2.resize(frame, (width, height))

                writer.write(frame)

            cap.release()
    finally:
        writer.release()


if __name__ == "__main__":
    # Base directory for videos
    BASE_PATH = "videos"

    # Videos to concatenate (order matters, first goes first, last goes last)
    videos = [
        "recording_0.mp4",
        "recording_1.mp4",
    ]

    # Optional resize settings
    ENABLE_RESIZE = True
    TARGET_SIZE = (1920, 1080)  # width, height (used only if ENABLE_RESIZE=True)

    video_paths = [os.path.join(BASE_PATH, v) for v in videos]
    output_video = os.path.join(BASE_PATH, "combined.mp4")

    print("Start the video concatenation")

    concatenate_videos(
        video_paths,
        output_video,
        resize=ENABLE_RESIZE,
        target_size=TARGET_SIZE,
    )

    print("Video concatenation complete!")
