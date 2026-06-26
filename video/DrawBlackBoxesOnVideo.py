"""
    Draw black rectangles on a video.

    Shows the middle frame for interactive drawing, then writes a copy of the full video with filled black boxes on every frame.
    This can be usefull when creating training data for ML purposes, and parts of the video are not relevant for what
    you are trying to do

    WARNING: if you pass the same path file as an already existing file as the output, this will overwrite that. This
    is a destructive action

    Instructions on how to draw rectangles:
    Left mouse drag: add a rectangle
    Enter / Space:   finish and write output video
    u:               undo last rectangle
    c:               clear all rectangles
    Esc:             cancel without writing

    Author :        Martijn Folmer
    Date created :  26-06-2026
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

Rect = Tuple[int, int, int, int]  # (x0, y0, x1, y1) in frame coordinates (top left to bottom right)

WINDOW_NAME = "Draw black boxes"
MAX_PREVIEW_DIMENSION = 1280

HELP_LINES = [
    "Drag: add rectangle",
    "Enter/Space: apply to video",
    "u: undo  c: clear  Esc: cancel",
]


def _normalize_rect(x0: int, y0: int, x1: int, y1: int) -> Optional[Rect]:
    """Make sure the rectangle is in (x,y)-top left and (x,y)-bottom right order"""
    xa, xb = sorted((x0, x1))
    ya, yb = sorted((y0, y1))
    if xb <= xa or yb <= ya:
        return None
    return xa, ya, xb, yb


def _preview_geometry(frame_w: int, frame_h: int) -> Tuple[float, int, int]:
    """Return (scale, preview_w, preview_h) fitting within MAX_PREVIEW_DIMENSION."""
    max_dim = max(frame_w, frame_h)
    if max_dim <= MAX_PREVIEW_DIMENSION:
        return 1.0, frame_w, frame_h

    scale = MAX_PREVIEW_DIMENSION / float(max_dim)
    preview_w = max(1, int(round(frame_w * scale)))
    preview_h = max(1, int(round(frame_h * scale)))
    return scale, preview_w, preview_h


def _display_to_frame_xy(x: int, y: int, scale: float) -> Tuple[int, int]:
    if scale == 1.0:
        return x, y
    return int(round(x / scale)), int(round(y / scale))


def _read_middle_frame(cap: cv2.VideoCapture) -> Tuple[np.ndarray, int]:
    """Get the middle frame of the video, the first and last frame can be fading in and out, so middle seems most
    representative"""
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames > 0:
        middle_idx = total_frames // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, middle_idx)
        ret, frame = cap.read()
        if ret:
            return frame, middle_idx

    cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 0.5)
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Could not read middle frame from video.")

    middle_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    return frame, middle_idx


def _draw_help_overlay(image: np.ndarray, rect_count: int) -> np.ndarray:
    """Draw the instructions with an outline for drawing on the gui"""
    out = image.copy()
    y = 24
    for line in HELP_LINES:
        cv2.putText(
            out,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        y += 22

    count_text = f"Rectangles: {rect_count}"
    cv2.putText(
        out,
        count_text,
        (10, y + 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        out,
        count_text,
        (10, y + 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    return out


def _render_preview(
    frame: np.ndarray,
    rectangles: List[Rect],
    scale: float,
    preview_w: int,
    preview_h: int,
    drag_start: Optional[Tuple[int, int]] = None,
    drag_end: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    if scale == 1.0:
        preview = frame.copy()
    else:
        preview = cv2.resize(frame, (preview_w, preview_h), interpolation=cv2.INTER_AREA)

    for x0, y0, x1, y1 in rectangles:
        if scale == 1.0:
            px0, py0, px1, py1 = x0, y0, x1, y1
        else:
            px0 = int(round(x0 * scale))
            py0 = int(round(y0 * scale))
            px1 = int(round(x1 * scale))
            py1 = int(round(y1 * scale))
        cv2.rectangle(preview, (px0, py0), (px1, py1), (0, 0, 0), thickness=-1)

    if drag_start is not None and drag_end is not None:
        cv2.rectangle(preview, drag_start, drag_end, (0, 0, 0), thickness=2)

    return _draw_help_overlay(preview, len(rectangles))


def _collect_rectangles_on_frame(frame: np.ndarray) -> Optional[List[Rect]]:
    """Use events to listen to the buttons and mouse actions and draw or delete rectangles"""
    frame_h, frame_w = frame.shape[:2]
    scale, preview_w, preview_h = _preview_geometry(frame_w, frame_h)

    rectangles: List[Rect] = []
    drag_start: Optional[Tuple[int, int]] = None
    drag_end: Optional[Tuple[int, int]] = None
    drawing = False

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        nonlocal drag_start, drag_end, drawing

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            drag_start = (x, y)
            drag_end = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            drag_end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and drawing:
            drawing = False
            if drag_start is None or drag_end is None:
                return

            fx0, fy0 = _display_to_frame_xy(drag_start[0], drag_start[1], scale)
            fx1, fy1 = _display_to_frame_xy(drag_end[0], drag_end[1], scale)
            rect = _normalize_rect(fx0, fy0, fx1, fy1)
            if rect is not None:
                rectangles.append(rect)

            drag_start = None
            drag_end = None

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    try:
        while True:
            preview = _render_preview(
                frame,
                rectangles,
                scale,
                preview_w,
                preview_h,
                drag_start=drag_start,
                drag_end=drag_end,
            )
            cv2.imshow(WINDOW_NAME, preview)

            key = cv2.waitKey(20) & 0xFF
            if key in (13, 32):  # Enter or Space
                if not rectangles:
                    print("Draw at least one rectangle before confirming.")
                    continue
                return rectangles
            if key == ord("u"):
                if rectangles:
                    rectangles.pop()
            elif key == ord("c"):
                rectangles.clear()
            elif key == 27:  # Esc
                return None
    finally:
        cv2.destroyWindow(WINDOW_NAME)


def _apply_black_rectangles(frame: np.ndarray, rectangles: List[Rect]) -> np.ndarray:
    out = frame.copy()
    for x0, y0, x1, y1 in rectangles:
        cv2.rectangle(out, (x0, y0), (x1, y1), (0, 0, 0), thickness=-1)
    return out


def draw_black_boxes_on_video(
    video_path: str,
    output_path: str,
    codec: str = "mp4v",
) -> None:
    """
    Show the middle frame for rectangle drawing, then write a video copy
    with filled black rectangles on every frame.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Input video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        middle_frame, middle_idx = _read_middle_frame(cap)
        print(f"Showing middle frame {middle_idx} for rectangle drawing.")

        rectangles = _collect_rectangles_on_frame(middle_frame)
        if rectangles is None:
            print("Cancelled. No output video written.")
            return

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Could not create output video: {output_path}")

        frames_written = 0
        progress_total = total_frames if total_frames > 0 else None

        try:
            with tqdm(total=progress_total, desc="Writing video", unit="frame") as pbar:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_out = _apply_black_rectangles(frame, rectangles)
                    writer.write(frame_out)
                    frames_written += 1
                    pbar.update(1)
        finally:
            writer.release()

        if frames_written == 0:
            raise RuntimeError("No frames were written. Check input video format.")

        print("Black box video complete!")
        print(f"Input video:      {video_path}")
        print(f"Rectangles drawn: {len(rectangles)}")
        print(f"Frames written:   {frames_written}")
        print(f"Saved to:         {output_path}")
    finally:
        cap.release()


if __name__ == "__main__":
    VIDEO_PATH = "PATH TO VIDEO WE WANT TO DRAW BLACK BOXES ON"
    OUTPUT_PATH = "FILEPATH TO WHERE WE WANT TO STORE IT"
    CODEC = "mp4v"

    if not os.path.isfile(VIDEO_PATH):
        raise SystemExit(f"Input file not found: {VIDEO_PATH}")

    draw_black_boxes_on_video(
        video_path=VIDEO_PATH,
        output_path=OUTPUT_PATH,
        codec=CODEC,
    )
