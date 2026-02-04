import cv2
from tqdm import tqdm
import math
import os

"""
    This script changes the FPS of a video, while keeping its length
    
    - if desired FPS > FPS of original video, we duplicate frames
    - if desired FPS < FPS of original video, we remove frames
    
    There is no interpolation or in between frame splicing, just purely duplication and removal of frames

    Author :        Martijn Folmer
    Date created :  04-02-2026
"""


def change_fps_by_frame_mapping(
    input_path: str,
    output_path: str,
    target_fps: float,
    codec: str = "mp4v",
) -> None:
    """
    Change FPS by dropping or duplicating frames
    Works by mapping each output frame time to a source frame index

    src_index = floor(i * src_fps / target_fps)

    This:
      - duplicates frames when target_fps > src_fps
      - drops frames when target_fps < src_fps

    Output duration is approximately preserved.
    """
    if target_fps <= 0:
        raise ValueError("target_fps must be > 0")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open input video: {input_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if not src_fps or src_fps <= 0:
        cap.release()
        raise RuntimeError("Could not read source FPS (CAP_PROP_FPS returned <= 0).")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_count <= 0:
        # need to know framecount, framecount can not be 0 or less than 0, but this sometimes happens with weird
        # encodings
        cap.release()
        raise RuntimeError(
            "Source frame count is unknown (<=0). "
        )

    duration_s = frame_count / src_fps
    out_count = max(1, int(round(duration_s * target_fps)))

    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(output_path, fourcc, float(target_fps), (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(
            f"Could not open output video for writing: {output_path}\n"
            f"Try another codec (e.g., 'avc1' on macOS, 'XVID' for AVI)."
        )

    # How many times we'll need each source frame (can be 0,1,2,...)
    # counts[j] = number of output frames that map to source frame j
    counts = [0] * frame_count
    for i in range(out_count):
        j = int(math.floor(i * src_fps / target_fps))
        if j < 0:
            j = 0
        elif j >= frame_count:
            j = frame_count - 1
        counts[j] += 1

    # Read each source frame once, write it counts[j] times
    pbar = tqdm(total=out_count, desc="Writing frames", unit="frame")
    for j in range(frame_count):
        read, frame = cap.read()

        # break if end of file
        if not read:
            break

        c = counts[j]
        if c <= 0:
            continue

        for _ in range(c):
            writer.write(frame)
        pbar.update(c)

    pbar.close()
    writer.release()
    cap.release()


if __name__ == "__main__":

    INPUT_VIDEO = "videos/clip_5s_to_12_5s.mp4"  # original video
    OUTPUT_VIDEO = "videos/output_5fps.mp4"      # location of new video
    TARGET_FPS = 5                               # FPS of new video
    CODEC = "mp4v"

    if not os.path.isfile(INPUT_VIDEO):
        raise SystemExit(f"Input file not found: {INPUT_VIDEO}")

    change_fps_by_frame_mapping(
        input_path=INPUT_VIDEO,
        output_path=OUTPUT_VIDEO,
        target_fps=TARGET_FPS,
        codec=CODEC,
    )

    print("Done writing the video.")
