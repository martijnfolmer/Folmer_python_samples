import os
import tkinter as tk
from tkinter import messagebox

from PIL import Image, ImageDraw, ImageTk

"""
    Draw black rectangles on top of images and save copies of them in an output directory

    left mouse button to drag a black box
    spacebar to move onto the next image and save the current one
    

    Author :        Martijn Folmer
    Date created :  01-04-2026
"""

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif", ".tiff", ".tif")

def collect_image_paths(source_dirs):
    """Return sorted file paths under source_dirs that have the proper suffixes"""
    paths = []
    for directory in source_dirs:
        if not os.path.isdir(directory):
            continue
        for name in sorted(os.listdir(directory)):
            full = os.path.join(directory, name)
            if not os.path.isfile(full):
                continue
            lower = name.lower()
            if lower.endswith(IMAGE_SUFFIXES):
                paths.append(full)
    return paths


class ImageBoxDrawer:
    """Load images one-by-one, drag left mouse to draw filled black boxes, Space = next."""

    def __init__(self, master, image_paths, output_dir):
        self.master = master
        self.master.title("Image Box Drawer")

        self.image_paths = list(image_paths)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.canvas = tk.Canvas(master, cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        menu = tk.Menu(master)
        master.config(menu=menu)
        file_menu = tk.Menu(menu, tearoff=0)
        menu.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Save Image", command=self.save_image)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=master.quit)

        self.image = None
        self.tk_image = None
        self.start_x = None
        self.start_y = None
        self.rect = None
        self.draw = None

        self.img_idx = -1

        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_move_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)
        self.master.bind("<space>", self.on_space_press)

        self.get_next_image()

    def get_next_image(self):
        if self.img_idx >= 0:
            self.save_image()

        self.img_idx += 1
        total = len(self.image_paths)
        print(f"Image {self.img_idx + 1} / {total}")

        if self.img_idx >= total:
            messagebox.showinfo("Done", "All images processed.")
            self.master.quit()
            return

        self.open_image(self.image_paths[self.img_idx])

    def on_space_press(self, _event):
        self.get_next_image()

    def open_image(self, file_path):
        self.canvas.delete("all")
        self.image = Image.open(file_path).convert("RGB")
        self.tk_image = ImageTk.PhotoImage(self.image)
        self.canvas.config(width=self.tk_image.width(), height=self.tk_image.height())
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)
        self.draw = ImageDraw.Draw(self.image)
        self.rect = None

    def save_image(self):
        if self.image is None or self.img_idx < 0:
            return
        basename = os.path.basename(self.image_paths[self.img_idx])
        out_path = os.path.join(self.output_dir, basename)
        self.image.save(out_path)

    def on_button_press(self, event):
        self.start_x = event.x
        self.start_y = event.y
        self.rect = self.canvas.create_rectangle(
            self.start_x,
            self.start_y,
            self.start_x,
            self.start_y,
            outline="black",
        )

    def on_move_press(self, event):
        if self.rect is None:
            return
        self.canvas.coords(self.rect, self.start_x, self.start_y, event.x, event.y)

    def on_button_release(self, event):
        if self.draw is None or self.start_x is None:
            return
        end_x, end_y = event.x, event.y
        self.draw.rectangle(
            [self.start_x, self.start_y, end_x, end_y],
            fill="black",
        )
        self.canvas.delete("all")
        self.tk_image = ImageTk.PhotoImage(self.image)
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)

if __name__ == "__main__":

    # folders with images
    input_dir = ["img"]
    # where we store the new images with black boxes on them
    output_dir = "img_with_bars"

    root = tk.Tk()
    paths = collect_image_paths(input_dir)
    print(paths)
    if not paths:
        messagebox.showerror(
            "No images",
            "No image files found. Check input_dir exist and contain images.",
        )
        root.destroy()
        exit()

    ImageBoxDrawer(root, paths, output_dir)
    root.mainloop()
