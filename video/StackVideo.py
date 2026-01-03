import cv2
import numpy as np
from pathlib import Path

"""
    Stack multiple videos into a single output video.

    # Direction
    - direction="horizontal": videos are placed left-to-right
    - direction="vertical": videos are placed top-to-bottom
    
    # Resizing
    - resize_to=(w,h): resize every frame to exactly that size before stacking
    - If resize_to is None, videos are resized to a common size (min width & min height across inputs)
      to keep things consistent and avoid mismatched shapes.

    Notes:
    - Output FPS is the minimum FPS across inputs.
    
    Author :        Martijn Folmer
    Date created :  03-01-2026
"""


def stack_videos(
    video_paths: list[str],                         # list of the paths to the videos we want to stack
    output_path: str,                               # Where we store the output video
    direction: str = "horizontal",                  # "horizontal" or "vertical"
    resize_to: tuple[int, int] = (1920, 1080),      # (width, height) or None
    pad_color: tuple[int, int, int] = (0, 0, 0),    # BGR
    codec: str = "mp4v",
) -> None:

    if direction not in ("horizontal", "vertical"):
        raise ValueError("direction must be 'horizontal' or 'vertical'")

    if not video_paths:
        raise ValueError("video_paths is empty")

    # Open captures
    caps = []
    for p in video_paths:
        cap = cv2.VideoCapture(p)
        if not cap.isOpened():
            # Clean up anything already opened
            for c in caps:
                c.release()
            raise FileNotFoundError(f"Could not open video: {p}")
        caps.append(cap)

    try:
        # Gather properties
        widths, heights, fps_list = [], [], []
        for cap in caps:
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps is None or fps <= 0:
                fps = 30.0  # fallback if metadata is missing
            widths.append(w)
            heights.append(h)
            fps_list.append(float(fps))

        out_fps = min(fps_list)

        # Decide per-frame target size
        if resize_to is not None:
            target_w, target_h = resize_to
            if target_w <= 0 or target_h <= 0:
                raise ValueError("resize_to must be positive (width, height)")
        else:
            # Use smallest common size to avoid upscaling by default
            target_w, target_h = min(widths), min(heights)

        # Output dimensions after stacking
        n = len(caps)
        if direction == "horizontal":
            out_w, out_h = target_w * n, target_h
        else:
            out_w, out_h = target_w, target_h * n

        # Create writer
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(output_path, fourcc, out_fps, (out_w, out_h))
        if not writer.isOpened():
            raise RuntimeError(f"Could not open VideoWriter for: {output_path}")

        # Processing loop: stop when any capture ends
        while True:
            frames = []
            for cap in caps:
                ok, frame = cap.read()
                if not ok:
                    return  # done (min-length behavior)

                # Ensure 3-channel BGR
                if frame is None:
                    return
                if frame.ndim == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

                # Resize to common size
                if (frame.shape[1], frame.shape[0]) != (target_w, target_h):
                    frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)

                frames.append(frame)

            # Stack
            if direction == "horizontal":
                stacked = np.hstack(frames)
            else:
                stacked = np.vstack(frames)

            # In case something odd happens, pad/crop to exact output size
            if stacked.shape[1] != out_w or stacked.shape[0] != out_h:
                canvas = np.full((out_h, out_w, 3), pad_color, dtype=np.uint8)
                h = min(out_h, stacked.shape[0])
                w = min(out_w, stacked.shape[1])
                canvas[:h, :w] = stacked[:h, :w]
                stacked = canvas

            writer.write(stacked)

    finally:
        for cap in caps:
            cap.release()
        # writer might not exist if we error early, so guard
        try:
            writer.release()  # type: ignore[name-defined]
        except Exception:
            pass


if __name__ == "__main__":

    VIDEO_PATHS = [
        "videos/recording_0.mp4",
        "videos/recording_1.mp4",
    ]

    OUTPUT_PATH = "videos/stacked_output.mp4"

    # "horizontal" or "vertical"
    DIRECTION = "vertical"

    # set to (width, height) to force a fixed size for each input
    # Set to None to auto-resize to smallest common size among inputs
    RESIZE_TO = (640, 360)  # e.g. (640, 360) or None
    # RESIZE_TO = None

    # Padding color (B, G, R) used only if an odd mismatch occurs
    PAD_COLOR = (0, 0, 0)

    # Codec (mp4v is commonly available; try 'avc1' or 'H264' if supported)
    CODEC = "mp4v"

    # Ensure output directory exists (if specified)
    out_dir = Path(OUTPUT_PATH).parent
    if str(out_dir) not in ("", ".") and not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)

    stack_videos(
        video_paths=VIDEO_PATHS,
        output_path=OUTPUT_PATH,
        direction=DIRECTION,
        resize_to=RESIZE_TO,
        pad_color=PAD_COLOR,
        codec=CODEC,
    )

    print(f"Saved stacked video to: {OUTPUT_PATH}")
