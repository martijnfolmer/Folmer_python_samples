"""
    This script repeats a single video multiple times, appending it to itself
    to create a looped output video.

    - Read the input video frame-by-frame
    - Optionally crop frames to a specified region
    - Optionally trim frames from the beginning and/or end
    - Write all frames (repeated N times) into a single output video file

    A tqdm-based progress bar is used to show progress.

    Author :        Martijn Folmer
    Date created :  13-01-2026
"""

import os
import cv2
from tqdm import tqdm


def repeat_video(
    video_path: str,
    output_path: str,
    repeat_count: int = 3,
    crop: tuple[int, int, int, int] | None = None,
    trim_start: int = 0,
    trim_end: int = 0,
    codec: str = "mp4v",
) -> None:
    """
    Repeat a video multiple times into a single output video.

    Args:
        video_path: Path to the input video file
        output_path: Path for the output video file
        repeat_count: Number of times to repeat the video (default: 3)
        crop: Optional crop region as (x, y, width, height). If None, no cropping is applied
        trim_start: Number of frames to skip from the beginning (default: 0)
        trim_end: Number of frames to skip from the end (default: 0)
        codec: Video codec to use (default: "mp4v")
    """
    if repeat_count < 1:
        raise ValueError("repeat_count must be at least 1")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    try:
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps <= 0:
            fps = 30.0  # Fallback if metadata is missing

        # Calculate effective frame range after trimming
        start_frame = trim_start
        end_frame = frame_count - trim_end
        if start_frame >= end_frame:
            raise ValueError(f"Trim values leave no frames (total: {frame_count}, trim_start: {trim_start}, trim_end: {trim_end})")
        effective_frame_count = end_frame - start_frame

        # Determine output dimensions (after cropping if specified)
        if crop is not None:
            crop_x, crop_y, crop_w, crop_h = crop
            # Validate crop region
            if crop_x < 0 or crop_y < 0:
                raise ValueError("Crop x and y must be non-negative")
            if crop_x + crop_w > original_width:
                raise ValueError(f"Crop region exceeds video width ({original_width})")
            if crop_y + crop_h > original_height:
                raise ValueError(f"Crop region exceeds video height ({original_height})")
            output_width, output_height = crop_w, crop_h
        else:
            output_width, output_height = original_width, original_height

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))

        if not writer.isOpened():
            raise RuntimeError(f"Could not open VideoWriter for: {output_path}")

        # Process and write frames for each repetition
        for i in range(repeat_count):
            # Reset video to start frame for each repetition
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            for _ in tqdm(
                range(effective_frame_count),
                desc=f"Repetition {i + 1}/{repeat_count}",
                position=0,
                leave=True,
            ):
                ret, frame = cap.read()
                if not ret:
                    break

                # Apply crop if specified
                if crop is not None:
                    frame = frame[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]

                writer.write(frame)

    finally:
        cap.release()
        try:
            writer.release()
        except Exception:
            pass


if __name__ == "__main__":
    # Path to the input video
    VIDEO_PATH = "original_video.mp4"

    # Path for the output video
    OUTPUT_PATH = "repeated_video.mp4"

    # Number of times to repeat the video
    REPEAT_COUNT = 3

    # Optional crop region: (x, y, width, height)
    # Set to None to disable cropping
    # Example: CROP = (100, 50, 640, 480) crops a 640x480 region starting at position (100, 50)
    CROP = None
    # CROP = (100, 50, 640, 480)

    # Trim frames from the beginning and end, set to 0 to disable trimming
    TRIM_START = 0  # Number of frames to skip from the beginning
    TRIM_END = 0    # Number of frames to skip from the end
    CODEC = "mp4v"

    # Ensure output directory exists
    out_dir = os.path.dirname(OUTPUT_PATH)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    print(f"Repeating video {REPEAT_COUNT} times...")
    if CROP:
        print(f"Cropping to region: x={CROP[0]}, y={CROP[1]}, w={CROP[2]}, h={CROP[3]}")
    if TRIM_START or TRIM_END:
        print(f"Trimming: {TRIM_START} frames from start, {TRIM_END} frames from end")

    repeat_video(
        video_path=VIDEO_PATH,
        output_path=OUTPUT_PATH,
        repeat_count=REPEAT_COUNT,
        crop=CROP,
        trim_start=TRIM_START,
        trim_end=TRIM_END,
        codec=CODEC,
    )

    print(f"Saved repeated video to: {OUTPUT_PATH}")
