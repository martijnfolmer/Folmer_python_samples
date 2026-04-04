"""
    Black box tool. This script takes an directory, opens a random image from it, and allows you to draw black rectangles on it.
    After you have drawn the rectangles, you can apply them to all the images in the directory.

    Usefull for removing parts of the image when preparing data for machine learning (like timestamps, or logos)

    Results are saved in a new directory named <source_folder_name>_black_box

    Author :        Martijn Folmer
    Date created :  04-04-2026
"""

import os
import random
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageDraw, ImageTk


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".tif", ".webp"}
CAPTURES_DIR = os.path.dirname(os.path.abspath(__file__))

# Map extension -> PIL save format (uppercase where needed)
_EXT_TO_FORMAT = {
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".png": "PNG",
    ".bmp": "BMP",
    ".gif": "GIF",
    ".tiff": "TIFF",
    ".tif": "TIFF",
    ".webp": "WEBP",
}


def _letterbox_geometry(
    img_w: int, img_h: int, canvas_w: int, canvas_h: int
) -> tuple[int, int, int, int]:
    """Return (disp_w, disp_h, offset_x, offset_y) for centered fit."""
    cw = max(canvas_w, 1)
    ch = max(canvas_h, 1)
    img_ratio = img_w / img_h
    win_ratio = cw / ch
    if img_ratio > win_ratio:
        disp_w = cw
        disp_h = int(cw / img_ratio)
    else:
        disp_h = ch
        disp_w = int(ch * img_ratio)
    ox = (cw - disp_w) // 2
    oy = (ch - disp_h) // 2
    return disp_w, disp_h, ox, oy


def _canvas_to_image_xy(
    cx: float,
    cy: float,
    ref_w: int,
    ref_h: int,
    disp_w: int,
    disp_h: int,
    ox: int,
    oy: int,
) -> tuple[float, float] | None:
    lx, ly = cx - ox, cy - oy
    if disp_w <= 0 or disp_h <= 0:
        return None
    if lx < 0 or ly < 0 or lx > disp_w or ly > disp_h:
        return None
    ix = lx * ref_w / disp_w
    iy = ly * ref_h / disp_h
    return ix, iy


def _norm_rect_from_image(
    x0: float, y0: float, x1: float, y1: float, ref_w: int, ref_h: int
) -> tuple[float, float, float, float]:
    """Normalize to [0,1] using reference size"""
    if ref_w <= 0 or ref_h <= 0:
        return (0.0, 0.0, 0.0, 0.0)
    xa, xb = sorted((x0, x1))
    ya, yb = sorted((y0, y1))
    return (
        max(0.0, min(1.0, xa / ref_w)),
        max(0.0, min(1.0, ya / ref_h)),
        max(0.0, min(1.0, xb / ref_w)),
        max(0.0, min(1.0, yb / ref_h)),
    )


def _pixel_rect_from_norm(
    nx0: float, ny0: float, nx1: float, ny1: float, w: int, h: int
) -> tuple[int, int, int, int]:
    x0 = int(round(nx0 * w))
    y0 = int(round(ny0 * h))
    x1 = int(round(nx1 * w))
    y1 = int(round(ny1 * h))
    x0 = max(0, min(w, x0))
    x1 = max(0, min(w, x1))
    y0 = max(0, min(h, y0))
    y1 = max(0, min(h, y1))
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    return x0, y0, x1, y1


def _apply_black_rectangles(img: Image.Image, norm_rects: list[tuple[float, float, float, float]]) -> Image.Image:
    w, h = img.size
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGBA" if "A" in img.getbands() else "RGB")
    draw = ImageDraw.Draw(img)
    fill = (0, 0, 0, 255) if img.mode == "RGBA" else (0, 0, 0)
    for nr in norm_rects:
        box = _pixel_rect_from_norm(*nr, w, h)
        if box[2] > box[0] and box[3] > box[1]:
            draw.rectangle(box, fill=fill)
    return img


def _save_image(img: Image.Image, dest_path: str, ext: str) -> None:
    fmt = _EXT_TO_FORMAT.get(ext.lower(), "PNG")
    save_kw: dict = {}
    if fmt == "JPEG":
        save_kw["quality"] = 95
        if img.mode == "RGBA":
            bg = Image.new("RGB", img.size, (0, 0, 0))
            bg.paste(img, mask=img.split()[3])
            img = bg
    elif fmt == "WEBP":
        save_kw["quality"] = 90
    img.save(dest_path, format=fmt, **save_kw)


class BlackBoxTool:
    def __init__(self, root: tk.Tk, source_dir: str) -> None:
        self.root = root
        self.source_dir = source_dir
        self.output_dir = source_dir.rstrip("/\\") + "_black_box"

        self.images = sorted(
            f for f in os.listdir(source_dir)
            if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
        )
        if not self.images:
            messagebox.showerror("No images", f"No supported images found in:\n{source_dir}")
            root.destroy()
            return

        ref_name = random.choice(self.images)
        ref_path = os.path.join(source_dir, ref_name)
        try:
            self._ref_pil = Image.open(ref_path).convert("RGBA")
        except Exception as exc:
            messagebox.showerror("Error", f"Cannot open reference image:\n{ref_path}\n\n{exc}")
            root.destroy()
            return

        self._ref_w, self._ref_h = self._ref_pil.size
        self._ref_name = ref_name
        # Normalized rectangles: (nx0, ny0, nx1, ny1) in 0..1
        self._norm_rects: list[tuple[float, float, float, float]] = []
        self._drag_start: tuple[float, float] | None = None
        self._preview_rect_id: int | None = None
        self._rect_item_ids: list[int] = []

        dir_name = os.path.basename(source_dir.rstrip("/\\"))
        root.title(f"Black box — {dir_name}")
        root.configure(bg="#1e1e1e")
        try:
            root.state("zoomed")
        except tk.TclError:
            pass

        self._build_ui()
        self._canvas.bind("<Configure>", self._on_canvas_configure)
        self._canvas.bind("<ButtonPress-1>", self._on_press)
        self._canvas.bind("<B1-Motion>", self._on_motion)
        self._canvas.bind("<ButtonRelease-1>", self._on_release)

    def _build_ui(self) -> None:
        top = tk.Frame(self.root, bg="#2d2d2d", pady=8)
        top.pack(side=tk.TOP, fill=tk.X)

        tk.Label(
            top,
            text="Drag on the image to add black rectangles. Clear removes all.",
            bg="#2d2d2d",
            fg="#e0e0e0",
            font=("Segoe UI", 11),
        ).pack(side=tk.LEFT, padx=12)

        tk.Button(
            top,
            text="Clear rectangles",
            command=self._on_clear,
            bg="#3c3c3c",
            fg="#e0e0e0",
            activebackground="#505050",
            font=("Segoe UI", 10),
        ).pack(side=tk.LEFT, padx=8)

        self._out_var = tk.StringVar(value=f"Output: {self.output_dir}")
        tk.Label(
            top,
            textvariable=self._out_var,
            bg="#2d2d2d",
            fg="#888888",
            font=("Segoe UI", 9),
        ).pack(side=tk.RIGHT, padx=12)

        self._ref_var = tk.StringVar(value=f"Reference (random): {self._ref_name}")
        tk.Label(
            self.root,
            textvariable=self._ref_var,
            bg="#1e1e1e",
            fg="#aaaaaa",
            font=("Segoe UI", 10),
        ).pack(side=tk.TOP, pady=(4, 0))

        self._canvas = tk.Canvas(self.root, bg="#1e1e1e", highlightthickness=0)
        self._canvas.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

        bottom = tk.Frame(self.root, bg="#1e1e1e", pady=8)
        bottom.pack(side=tk.BOTTOM, fill=tk.X)
        tk.Button(
            bottom,
            text="Apply to all images",
            command=self._on_apply_all,
            bg="#0e639c",
            fg="#ffffff",
            activebackground="#1177bb",
            font=("Segoe UI", 12),
            padx=16,
            pady=6,
        ).pack()

        self._photo: ImageTk.PhotoImage | None = None
        self._image_id: int | None = None
        self._disp_geom: tuple[int, int, int, int] | None = None

    def _on_canvas_configure(self, _event: tk.Event) -> None:
        self._redraw_canvas()

    def _redraw_canvas(self) -> None:
        cw = max(self._canvas.winfo_width(), 100)
        ch = max(self._canvas.winfo_height(), 100)
        dw, dh, ox, oy = _letterbox_geometry(self._ref_w, self._ref_h, cw, ch)
        self._disp_geom = (dw, dh, ox, oy)

        resized = self._ref_pil.resize((dw, dh), Image.LANCZOS)
        self._photo = ImageTk.PhotoImage(resized)
        self._canvas.delete("all")
        self._image_id = self._canvas.create_image(ox + dw // 2, oy + dh // 2, image=self._photo)
        self._rect_item_ids.clear()
        self._preview_rect_id = None

        for nr in self._norm_rects:
            rid = self._draw_norm_rect_on_canvas(nr)
            self._rect_item_ids.append(rid)

    def _norm_to_canvas_rect(
        self, nr: tuple[float, float, float, float]
    ) -> tuple[int, int, int, int] | None:
        if self._disp_geom is None:
            return None
        dw, dh, ox, oy = self._disp_geom
        nx0, ny0, nx1, ny1 = nr
        x0 = ox + int(nx0 * dw)
        y0 = oy + int(ny0 * dh)
        x1 = ox + int(nx1 * dw)
        y1 = oy + int(ny1 * dh)
        return (x0, y0, x1, y1)

    def _draw_norm_rect_on_canvas(self, nr: tuple[float, float, float, float]) -> int:
        box = self._norm_to_canvas_rect(nr)
        if box is None:
            return self._canvas.create_rectangle(0, 0, 0, 0, outline="", width=0)
        x0, y0, x1, y1 = box
        return self._canvas.create_rectangle(
            x0,
            y0,
            x1,
            y1,
            outline="#ffcc00",
            width=2,
            stipple="gray50",
        )

    def _on_press(self, event: tk.Event) -> None:
        self._drag_start = (event.x, event.y)
        if self._preview_rect_id is not None:
            self._canvas.delete(self._preview_rect_id)
            self._preview_rect_id = None

    def _on_motion(self, event: tk.Event) -> None:
        if self._drag_start is None or self._disp_geom is None:
            return
        x0, y0 = self._drag_start
        x1, y1 = event.x, event.y
        if self._preview_rect_id is not None:
            self._canvas.delete(self._preview_rect_id)
        self._preview_rect_id = self._canvas.create_rectangle(
            x0, y0, x1, y1, outline="#ffcc00", width=2, dash=(4, 4)
        )

    def _on_release(self, event: tk.Event) -> None:
        if self._preview_rect_id is not None:
            self._canvas.delete(self._preview_rect_id)
            self._preview_rect_id = None
        if self._drag_start is None or self._disp_geom is None:
            return
        sx, sy = self._drag_start
        self._drag_start = None
        dw, dh, ox, oy = self._disp_geom

        p0 = _canvas_to_image_xy(sx, sy, self._ref_w, self._ref_h, dw, dh, ox, oy)
        p1 = _canvas_to_image_xy(event.x, event.y, self._ref_w, self._ref_h, dw, dh, ox, oy)
        if p0 is None or p1 is None:
            return
        nr = _norm_rect_from_image(p0[0], p0[1], p1[0], p1[1], self._ref_w, self._ref_h)
        if nr[2] - nr[0] < 1e-6 or nr[3] - nr[1] < 1e-6:
            return
        self._norm_rects.append(nr)
        rid = self._draw_norm_rect_on_canvas(nr)
        self._rect_item_ids.append(rid)

    def _on_clear(self) -> None:
        self._norm_rects.clear()
        self._redraw_canvas()

    def _on_apply_all(self) -> None:
        if not self._norm_rects:
            messagebox.showwarning("No rectangles", "Draw at least one rectangle first.")
            return
        os.makedirs(self.output_dir, exist_ok=True)
        ok = 0
        errors: list[str] = []
        for name in self.images:
            src = os.path.join(self.source_dir, name)
            ext = os.path.splitext(name)[1].lower()
            dst = os.path.join(self.output_dir, name)
            try:
                with Image.open(src) as im:
                    im = im.copy()
                out = _apply_black_rectangles(im, self._norm_rects)
                if ext == ".gif" and getattr(out, "is_animated", False):
                    out.seek(0)
                    base, _ = os.path.splitext(name)
                    dst = os.path.join(self.output_dir, base + ".png")
                    out = out.convert("RGBA")
                    _save_image(out, dst, ".png")
                else:
                    _save_image(out, dst, ext)
                ok += 1
            except Exception as exc:
                errors.append(f"{name}: {exc}")

        msg = f"Processed {ok} of {len(self.images)} image(s).\nSaved to:\n{self.output_dir}"
        if errors:
            msg += "\n\nErrors:\n" + "\n".join(errors[:8])
            if len(errors) > 8:
                msg += f"\n... and {len(errors) - 8} more."
            messagebox.showwarning("Done with errors", msg)
        else:
            messagebox.showinfo("Done", msg)


if __name__ == "__main__":
    root = tk.Tk()
    source_dir = "<insert folder name here>"
    if not source_dir:
        root.withdraw()
        source_dir = filedialog.askdirectory(
            title="Choose folder with images",
            initialdir=CAPTURES_DIR,
        )
        if not source_dir:
            root.destroy()
            return
        root.deiconify()

    source_dir = os.path.abspath(source_dir)
    if not os.path.isdir(source_dir):
        messagebox.showerror("Invalid path", f"Not a directory:\n{source_dir}")
        root.destroy()
        return

    BlackBoxTool(root, source_dir)
    root.mainloop()
