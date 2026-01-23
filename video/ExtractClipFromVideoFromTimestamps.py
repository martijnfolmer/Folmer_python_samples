import os
import cv2

"""
    This script extracts a video clip between two timestamps
    and saves it to a new video file using OpenCV (cv2).

    Timestamps are given in seconds (float or int).
    The script seeks to the start timestamp, then writes frames
    until the end timestamp is reached.

    Author :        Martijn Folmer
    Date created :  23-01-2026
"""


def ExtractClipFromVideoFromTimestamps(video_path, start_time_s, end_time_s, output_path):
    if start_time_s < 0 or end_time_s <= start_time_s:
        raise ValueError("Invalid timestamps: ensure 0 <= start_time_s < end_time_s.")

    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Input video not found: {video_path}")

    # Open input video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    # Read video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        raise RuntimeError("Could not determine FPS for the input video.")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Convert timestamps to frame indices
    start_frame = int(round(start_time_s * fps))
    end_frame = int(round(end_time_s * fps))

    # Clamp to video bounds
    start_frame = max(0, min(start_frame, total_frames - 1))
    end_frame = max(0, min(end_frame, total_frames))

    if end_frame <= start_frame:
        cap.release()
        raise ValueError("Timestamps resolve to an empty clip (check video length / fps).")

    # Ensure output directory exists
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    if not out.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create output video: {output_path}")

    # set video to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames_written = 0
    current_frame = start_frame

    # Write frames until we reach end_frame
    while current_frame < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        out.write(frame)
        frames_written += 1
        current_frame += 1
        print(f"We are at frame {frames_written} / {end_frame}, we have written {frames_written} frames")

    cap.release()
    out.release()

    if frames_written == 0:
        raise RuntimeError("No frames were written. Check timestamps and video format.")

    print("Clip extraction complete!")
    print(f"Input video:     {video_path}")
    print(f"Start time (s):  {start_time_s}  -> frame {start_frame}")
    print(f"End time (s):    {end_time_s}    -> frame {end_frame}")
    print(f"Frames written:  {frames_written}")
    print(f"Saved to:        {output_path}")


if __name__ == "__main__":
    VIDEOPATH = "videos/recording_0.mp4"
    START_TIME_S = 5.0
    END_TIME_S = 12.5
    OUTPUTPATH = "videos/clip_5s_to_12_5s.mp4"

    ExtractClipFromVideoFromTimestamps(
        video_path=VIDEOPATH,
        start_time_s=START_TIME_S,
        end_time_s=END_TIME_S,
        output_path=OUTPUTPATH
    )
