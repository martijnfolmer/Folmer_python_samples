import tkinter as tk
from tkinter import font, filedialog, messagebox
import sys
import os
from PIL import Image, ImageDraw, ImageFont, ImageTk

"""
    Draws a piece of text with all installed fonts on the system, so we can pick one for our purposes

    Can optionally add a list of custom fonts. That must be a .txt file that has the following format:
    
    path/to/custom/font1
    path/to/custom/font2
    path/to/custom/font3

    Author :        Martijn Folmer
    Date created :  26-07-26
"""



class FontViewerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("System & Custom Font Viewer")
        self.root.geometry("1200x800")

        self.custom_font_paths = []
        self.preview_widgets = []

        top_frame = tk.Frame(root, padx=10, pady=10, bg="#f0f0f0")
        top_frame.pack(fill=tk.X)

        tk.Label(top_frame, text="Enter text:", bg="#f0f0f0").pack(side=tk.LEFT)

        self.text_var = tk.StringVar()
        self.text_var.set("The quick brown fox jumps over the lazy dog")
        self.entry = tk.Entry(top_frame, textvariable=self.text_var, width=40, font=("Helvetica", 12))
        self.entry.pack(side=tk.LEFT, padx=10)

        tk.Button(top_frame, text="Update Text", command=self.update_previews).pack(side=tk.LEFT, padx=5)
        tk.Button(top_frame, text="Load Custom Fonts (.txt)", command=self.load_custom_fonts_file).pack(side=tk.LEFT,
                                                                                                        padx=5)
        self.root.bind('<Return>', self.update_previews)

        self.canvas = tk.Canvas(root, borderwidth=0, highlightthickness=0)
        self.scrollbar = tk.Scrollbar(root, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas_window = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        self.canvas.bind("<Configure>", self.on_canvas_configure)
        self.root.bind_all("<MouseWheel>", self._on_mousewheel)
        self.root.bind_all("<Button-4>", self._on_mousewheel_linux)
        self.root.bind_all("<Button-5>", self._on_mousewheel_linux)

        # Load system fonts on startup
        self.build_font_list()

    def on_canvas_configure(self, event):
        self.canvas.itemconfig(self.canvas_window, width=event.width)

    def _on_mousewheel(self, event):
        delta = -1 if sys.platform == 'darwin' else -1 * (event.delta // 120)
        self.canvas.yview_scroll(delta, "units")

    def _on_mousewheel_linux(self, event):
        if event.num == 4:
            self.canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            self.canvas.yview_scroll(1, "units")

    def load_custom_fonts_file(self):
        # get a .txt file with the custom font paths
        filepath = filedialog.askopenfilename(
            title="Select Custom Fonts List",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
        )

        if not filepath:
            return

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                paths = [line.strip() for line in f if line.strip()]

            self.custom_font_paths = paths
            messagebox.showinfo("Success", f"Loaded {len(paths)} custom font paths. Rebuilding list...")
            self.build_font_list()
        except Exception as e:
            messagebox.showerror("Error", f"Could not read file:\n{e}")

    def build_font_list(self):
        """Clears the current list and rebuilds it with system + custom fonts."""
        # Clear existing widgets in the scrollable frame
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()

        self.preview_widgets.clear()

        # System Fonts
        available_fonts = list(set(font.families()))
        available_fonts.sort()

        for f in available_fonts:
            self.create_font_row(font_type="system", name=f)

        # Custom Fonts from File
        for path in self.custom_font_paths:
            if os.path.exists(path):
                self.create_font_row(font_type="custom", path=path)
            else:
                print(f"Warning: Font file not found at {path}")

        # Draw the initial text
        self.update_previews()

    def create_font_row(self, font_type, name="", path=""):
        container = tk.Frame(self.scrollable_frame, pady=5, padx=10)
        container.pack(fill=tk.X)

        if font_type == "system":
            display_name = f"{name} (System Font)"
        else:
            filename = os.path.basename(path)
            display_name = f"{filename} (Custom File)"

        name_lbl = tk.Label(container, text=display_name, font=("Helvetica", 10, "bold"), fg="#333333", anchor="w")
        name_lbl.pack(fill=tk.X)

        prev_lbl = tk.Label(container, anchor="w", bg="white", relief="solid", borderwidth=1, padx=8, pady=8)
        prev_lbl.pack(fill=tk.X)

        # Save widget data so update_previews knows how to render it
        self.preview_widgets.append({
            'label': prev_lbl,
            'type': font_type,
            'name': name,
            'path': path
        })

    def update_previews(self, event=None):
        text = self.text_var.get()
        if not text:
            text = " "

        for item in self.preview_widgets:
            lbl = item['label']

            if item['type'] == "system":
                lbl.config(text=text, font=(item['name'], 16), image="")

            elif item['type'] == "custom":
                try:
                    # Load font file
                    pil_font = ImageFont.truetype(item['path'], 24)

                    # Calculate image size needed
                    bbox = pil_font.getbbox(text)
                    width = bbox[2] - bbox[0] + 20
                    height = bbox[3] - bbox[1] + 20

                    # Create blank white image
                    img = Image.new('RGB', (width, max(height, 40)), color='white')
                    draw = ImageDraw.Draw(img)

                    draw.text((10, 10), text, font=pil_font, fill='black')

                    photo = ImageTk.PhotoImage(img)
                    lbl.config(image=photo, text="")

                    lbl.image = photo

                except Exception as e:
                    lbl.config(text=f"Error rendering font: {e}", font=("Helvetica", 10), image="")


if __name__ == "__main__":
    root = tk.Tk()
    app = FontViewerApp(root)
    root.mainloop()

