import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
import numpy as np
from skimage import exposure, io, util

class HistAdjustApp:
    """
    This class implements a user interface for image processing with histogram adjustment functionalities.
    """
  
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processing Laboratory - Histogram Adjustment App")

        # Image variables
        self.original_image = None
        self.modified_image = None

        # Image selection components
        self.image_label = ttk.Label(self.root, text="Choose an image:")
        self.image_label.grid(row=0, column=0, columnspan=3)

        self.image_selection = tk.StringVar(value="tires")
        self.radio_tires = ttk.Radiobutton(self.root, text="Tires", variable=self.image_selection, value="tires")
        self.radio_tires.grid(row=0, column=3)
        self.radio_bubbles = ttk.Radiobutton(self.root, text="Bubbles", variable=self.image_selection, value="bubbles")
        self.radio_bubbles.grid(row=0, column=4)

        self.load_image_button = ttk.Button(self.root, text="Load Image", command=self.load_image)
        self.load_image_button.grid(row=0, column=5)

        self.fig, self.ax = plt.subplots(2, 2, figsize=(12, 8))

        # Radio button components
        self.processing_option = tk.StringVar(value="None")
        self.radio_gamma = ttk.Radiobutton(self.root, text="Gamma", variable=self.processing_option, value="Gamma", command=self.select_gamma)
        self.radio_gamma.grid(row=1, column=0)
        self.radio_gamma.config(state="disabled")
        self.radio_contrast = ttk.Radiobutton(self.root, text="Contrast", variable=self.processing_option, value="Contrast", command=self.select_contrast)
        self.radio_contrast.grid(row=2, column=0)
        self.radio_contrast.config(state="disabled")
        self.radio_brightness = ttk.Radiobutton(self.root, text="Brightness", variable=self.processing_option, value="Brightness", command=self.select_brightness)
        self.radio_brightness.grid(row=3, column=0)
        self.radio_brightness.config(state="disabled")
        self.radio_hist_eq = ttk.Radiobutton(self.root, text="Histogram Equalization", variable=self.processing_option, value="HistEq", command=self.select_histeq)
        self.radio_hist_eq.grid(row=4, column=0)
        self.radio_hist_eq.config(state="disabled")

        # Scroller components
        self.gamma_value = tk.DoubleVar()
        self.gamma_scroll = ttk.Scale(self.root, from_=0.1, to=5.0, orient=tk.HORIZONTAL, length=200, variable=self.gamma_value, command=self.update_gamma_label)
        self.gamma_scroll.grid(row=1, column=1, columnspan=4)
        self.gamma_label = ttk.Label(self.root, text="Gamma: 1.0")
        self.gamma_label.grid(row=1, column=5)

        self.contrast_value = tk.DoubleVar()
        self.contrast_scroll = ttk.Scale(self.root, from_=0.1, to=5.0, orient=tk.HORIZONTAL, length=200, variable=self.contrast_value, command=self.update_contrast_label)
        self.contrast_scroll.grid(row=2, column=1, columnspan=4)
        self.contrast_label = ttk.Label(self.root, text="Contrast: 1.0")
        self.contrast_label.grid(row=2, column=5)

        self.brightness_value = tk.DoubleVar()
        self.brightness_scroll = ttk.Scale(self.root, from_=0.0, to=1.0, orient=tk.HORIZONTAL, length=200, variable=self.brightness_value, command=self.update_brightness_label)
        self.brightness_scroll.grid(row=3, column=1, columnspan=4)
        self.brightness_label = ttk.Label(self.root, text="Brightness: 0.5")
        self.brightness_label.grid(row=3, column=5)

        # Apply button
        self.apply_button = ttk.Button(self.root, text="Apply", command=self.apply_processing)
        self.apply_button.grid(row=5, column=0)

        # Reset button
        self.reset_button = ttk.Button(self.root, text="Reset", command=self.reset)
        self.reset_button.grid(row=5, column=1)

        # Image display
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=6, column=0, columnspan=6)

        # Initialize variables
        self.gamma_value.set(1.0)
        self.contrast_value.set(1.0)
        self.brightness_value.set(0.5)

        self.disable_scrollers()

    def select_gamma(self):
        self.disable_scrollers()
        self.gamma_scroll.config(state="normal")
        self.gamma_label.config(state="normal")

    def select_contrast(self):
        self.disable_scrollers()
        self.contrast_scroll.config(state="normal")
        self.contrast_label.config(state="normal")

    def select_brightness(self):
        self.disable_scrollers()
        self.brightness_scroll.config(state="normal")
        self.brightness_label.config(state="normal")

    def select_histeq(self):
        self.disable_scrollers()

    def disable_scrollers(self):
        self.gamma_scroll.config(state="disabled")
        self.contrast_scroll.config(state="disabled")
        self.brightness_scroll.config(state="disabled")
        self.gamma_label.config(state="disabled")
        self.contrast_label.config(state="disabled")
        self.brightness_label.config(state="disabled")

    def apply_processing(self):
        if self.processing_option.get() == "Gamma":
            self.apply_gamma()
        elif self.processing_option.get() == "Contrast":
            self.apply_contrast()
        elif self.processing_option.get() == "Brightness":
            self.apply_brightness()
        elif self.processing_option.get() == "HistEq":
            self.histogram_equalization()

    def load_image(self):
        selected_image = self.image_selection.get()
        image_path = os.path.join("images", f"{selected_image}.jpg")
        self.original_image = io.imread(image_path)
        self.modified_image = self.original_image.copy()
        self.display_images()
        self.radio_gamma.config(state="normal")
        self.radio_contrast.config(state="normal")
        self.radio_brightness.config(state="normal")
        self.radio_hist_eq.config(state="normal")

    def display_images(self):
        self.ax[0, 0].cla()
        self.ax[0, 1].cla()
        self.ax[1, 0].cla()
        self.ax[1, 1].cla()

        self.ax[0, 0].imshow(self.original_image, cmap='gray', vmin=0, vmax=255)
        self.ax[0, 0].set_title("Original Image")

        self.ax[0, 1].hist(np.array(self.original_image).ravel(), bins=256, color='gray', alpha=0.7)
        self.ax[0, 1].set_title("Original Image Histogram")
        self.ax[0, 1].set_xlim(left=0, right=255)

        self.ax[1, 0].imshow(self.modified_image, cmap='gray', vmin=0, vmax=255)
        self.ax[1, 0].set_title("Modified Image")

        self.ax[1, 1].hist(np.array(self.modified_image).ravel(), bins=256, color='gray', alpha=0.7)
        self.ax[1, 1].set_title("Modified Image Histogram")
        self.ax[1, 1].set_xlim(left=0, right=255)

        self.canvas.draw()

    def apply_gamma(self):
        gamma_value = self.gamma_value.get()
        self.modified_image = np.array(self.original_image).astype(float)
        self.modified_image = exposure.adjust_gamma(self.original_image, gamma=gamma_value)
        self.display_images()

    def apply_contrast(self):
        contrast_value = self.contrast_value.get()
        self.modified_image = np.array(self.original_image).astype(float)
        self.modified_image = (self.original_image - 0.5) * contrast_value + 0.5
        self.display_images()

    def apply_brightness(self):
        brightness_value = self.brightness_value.get()
        self.modified_image = np.array(self.original_image).astype(float)
        self.modified_image = np.clip(self.modified_image + int((brightness_value - 0.5) * 400), 0, 255)
        self.display_images()

    def histogram_equalization(self):
        self.modified_image = exposure.equalize_hist(self.modified_image)
        self.modified_image = util.img_as_ubyte(self.modified_image) # convert back to uint8
        self.display_images()

    def reset(self):
        self.processing_option.set("None")
        self.gamma_scroll.set(1.0)
        self.contrast_scroll.set(1.0)
        self.brightness_scroll.set(0.5)
        self.modified_image = self.original_image.copy()
        self.display_images()
        self.disable_scrollers()

    def update_gamma_label(self, value):
        self.gamma_label.config(text=f"Gamma: {float(value):.2f}")

    def update_contrast_label(self, value):
        self.contrast_label.config(text=f"Contrast: {float(value):.2f}")

    def update_brightness_label(self, value):
        self.brightness_label.config(text=f"Brightness: {float(value):.2f}")


if __name__ == "__main__":
    root = tk.Tk()
    app = HistAdjustApp(root)
    root.mainloop()
