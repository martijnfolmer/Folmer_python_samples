import os
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox
from typing import Optional
from PIL import Image, ImageTk

"""
    Interactive GUI to step through images in one folder. Use the left key to
    skip or the right key to copy the current file into a new folder named
    <source_folder_name>_selected

    This will not delete the original images

    Author :        Martijn Folmer
    Date created :  04-04-2026
"""

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".gif",
    ".tiff",
    ".tif",
    ".webp",
}


class ImageSelector:
    """Tkinter UI to view images and copy chosen files to an output folder."""

    def __init__(self, root: tk.Tk, source_dir: str) -> None:
        self.root = root
        self.source_dir = source_dir
        self.output_dir = source_dir.rstrip("/\\") + "_selected"

        self.images = sorted(
            f
            for f in os.listdir(source_dir)
            if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
        )
        if not self.images:
            messagebox.showerror(
                "No images",
                f"No supported images found in:\n{source_dir}",
            )
            root.destroy()
            return

        self.total = len(self.images)
        self.index = 0
        self.selected_count = 0

        dir_name = os.path.basename(source_dir.rstrip("/\\"))
        root.title(f"Image Selector — {dir_name}")
        root.configure(bg="#1e1e1e")
        root.state("zoomed")

        self._build_ui()
        self._show_current()

        root.bind("<Left>", self._on_skip)
        root.bind("<Right>", self._on_select)
        root.focus_force()

    def _build_ui(self) -> None:
        """Create header, progress text, filename label, and image area."""
        top = tk.Frame(self.root, bg="#2d2d2d", pady=8)
        top.pack(side=tk.TOP, fill=tk.X)

        self.progress_var = tk.StringVar()
        progress_label = tk.Label(
            top,
            textvariable=self.progress_var,
            bg="#2d2d2d",
            fg="#e0e0e0",
            font=("Segoe UI", 13),
        )
        progress_label.pack(side=tk.LEFT, padx=20)

        hint = tk.Label(
            top,
            text="← Skip       Select & Copy →",
            bg="#2d2d2d",
            fg="#888888",
            font=("Segoe UI", 11),
        )
        hint.pack(side=tk.RIGHT, padx=20)

        self.filename_var = tk.StringVar()
        filename_label = tk.Label(
            self.root,
            textvariable=self.filename_var,
            bg="#1e1e1e",
            fg="#aaaaaa",
            font=("Segoe UI", 10),
        )
        filename_label.pack(side=tk.TOP, pady=(4, 0))

        self.canvas = tk.Label(self.root, bg="#1e1e1e")
        self.canvas.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)
        self.root.bind("<Configure>", self._on_resize)

    def _show_current(self) -> None:
        """Update labels and show the image at the current index, or finish."""
        if self.index >= self.total:
            self._finish()
            return

        remaining = self.total - self.index
        self.progress_var.set(
            f"Image {self.index + 1} / {self.total}   |   "
            f"Selected: {self.selected_count}   |   "
            f"Remaining: {remaining}"
        )
        self.filename_var.set(self.images[self.index])
        self._render_image()

    def _render_image(self) -> None:
        """Load the current file from disk; on failure, show error and skip."""
        path = os.path.join(self.source_dir, self.images[self.index])
        try:
            img = Image.open(path)
        except OSError as exc:
            messagebox.showerror(
                "Error",
                f"Cannot open image:\n{path}\n\n{exc}",
            )
            self._advance()
            return

        self._current_pil = img
        self._resize_and_display()

    def _resize_and_display(self) -> None:
        """Scale the PIL image to fit the canvas while preserving aspect ratio."""
        img = self._current_pil
        cw = max(self.canvas.winfo_width(), 100)
        ch = max(self.canvas.winfo_height(), 100)

        img_ratio = img.width / img.height
        win_ratio = cw / ch

        if img_ratio > win_ratio:
            new_w = cw
            new_h = int(cw / img_ratio)
        else:
            new_h = ch
            new_w = int(ch * img_ratio)

        resized = img.resize((new_w, new_h), Image.LANCZOS)
        self._photo = ImageTk.PhotoImage(resized)
        self.canvas.configure(image=self._photo)

    def _on_resize(self, _event: tk.Event) -> None:
        """Re-fit the image when the window is resized."""
        if hasattr(self, "_current_pil"):
            self._resize_and_display()

    def _on_skip(self, _event: Optional[tk.Event] = None) -> None:
        """Skip the current image without copying."""
        self._advance()

    def _on_select(self, _event: Optional[tk.Event] = None) -> None:
        """Copy the current file to the output directory and advance."""
        os.makedirs(self.output_dir, exist_ok=True)
        src = os.path.join(self.source_dir, self.images[self.index])
        dst = os.path.join(self.output_dir, self.images[self.index])
        shutil.copy2(src, dst)
        self.selected_count += 1
        self._advance()

    def _advance(self) -> None:
        """Move to the next image."""
        self.index += 1
        self._show_current()

    def _finish(self) -> None:
        """Clear the preview and show a short summary when all images are done."""
        self.canvas.configure(image="")
        self.filename_var.set("")
        self.progress_var.set(
            f"Done!   {self.total} images reviewed   |   "
            f"{self.selected_count} selected   |   "
            f"Saved to: {self.output_dir}"
        )
        messagebox.showinfo(
            "All done!",
            f"Reviewed all {self.total} images.\n"
            f"Selected: {self.selected_count}\n"
            f"Saved to: {self.output_dir}",
        )

if __name__ == "__main__":
    """Open a folder picker, then run the selector on the chosen directory"""
    INITIAL_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

    root = tk.Tk()
    root.withdraw()

    source_dir = filedialog.askdirectory(
        title="Choose a captures subdirectory",
        initialdir=INITIAL_DIRECTORY,
    )

    if not source_dir:
        root.destroy()
        return

    root.deiconify()
    ImageSelector(root, source_dir)
    root.mainloop()
