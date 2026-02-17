import os
import cv2
from tqdm import tqdm


"""
    This script splits a single video into N equal-length clips based on frame count.

    A tqdm-based progress bar is used to show progress for each clip.

    Author :        Martijn Folmer
    Date created :  20-01-2026
"""


def split_video_into_clips(
    video_path: str,
    output_dir: str,
    num_clips: int = 3,
    codec: str = "mp4v",
) -> None:
    """
    Split a video into N equal-length clips based on frame count.

    Args:
        video_path: Path to the input video file
        output_dir: Directory to save the output clips
        num_clips: Number of clips to create (default: 3)
        codec: Video codec to use (default: "mp4v")
    """
    if num_clips < 1:
        raise ValueError("num_clips must be at least 1")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    try:
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps <= 0:
            fps = 30.0  # Fallback if metadata is missing

        if total_frames < num_clips:
            raise ValueError(f"Video has {total_frames} frames, which is less than the requested {num_clips} clips")

        # Calculate frames per clip
        frames_per_clip = total_frames // num_clips
        remainder_frames = total_frames % num_clips

        # Generate base filename from input video
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        extension = os.path.splitext(video_path)[1]

        # Ensure output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Process each clip
        for clip_idx in range(num_clips):
            # Calculate frame range for this clip
            start_frame = clip_idx * frames_per_clip
            # Last clip gets any remainder frames
            if clip_idx == num_clips - 1:
                end_frame = total_frames
            else:
                end_frame = start_frame + frames_per_clip

            clip_frame_count = end_frame - start_frame

            # Generate output filename
            output_filename = f"{base_name}_clip_{clip_idx + 1}{extension}"
            output_path = os.path.join(output_dir, output_filename)

            # Create video writer for this clip
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            if not writer.isOpened():
                raise RuntimeError(f"Could not open VideoWriter for: {output_path}")

            # Seek to start frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            # Write frames for this clip
            for _ in tqdm(
                range(clip_frame_count),
                desc=f"Clip {clip_idx + 1}/{num_clips}",
                position=0,
                leave=True,
            ):
                ret, frame = cap.read()
                if not ret:
                    break

                writer.write(frame)

            writer.release()

    finally:
        cap.release()


if __name__ == "__main__":
    # Path to the input video
    VIDEO_PATH = "original_video.mp4"

    # Directory to save the output clips
    OUTPUT_DIR = "output_clips"

    # Number of clips to create
    NUM_CLIPS = 3

    # Video codec to use
    CODEC = "mp4v"

    print(f"Splitting video into {NUM_CLIPS} clips...")
    print(f"Output directory: {OUTPUT_DIR}")

    split_video_into_clips(
        video_path=VIDEO_PATH,
        output_dir=OUTPUT_DIR,
        num_clips=NUM_CLIPS,
        codec=CODEC,
    )

    print(f"Saved {NUM_CLIPS} clips to: {OUTPUT_DIR}")


