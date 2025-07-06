import tkinter as tk
from tkinter import filedialog, messagebox, ttk # ttk for potentially better looking widgets if needed
from PIL import Image, ImageTk
import cv2
import numpy as np
from skimage import measure
from skimage.morphology import remove_small_objects

class AcneDetectorApp:
    def __init__(self, master):
        self.master = master
        master.title("Acne Detection Pro")
        master.geometry("960x720") # Give a default size

        # --- Color Palette ---
        self.COLOR_BG_PRIMARY = "#2C3E50"    # Dark Slate Blue
        self.COLOR_BG_SECONDARY = "#34495E"  # Wet Asphalt (for panels)
        self.COLOR_BG_CONTENT = "#ECF0F1"    # Clouds (for content areas within panels)
        self.COLOR_ACCENT_PRIMARY = "#3498DB" # Peter River (for primary buttons)
        self.COLOR_ACCENT_SUCCESS = "#2ECC71" # Emerald (for process button)
        self.COLOR_ACCENT_WARNING = "#F1C40F" # Sunflower (for reset)
        self.COLOR_ACCENT_DANGER = "#E74C3C"  # Alizarin (for exit)
        self.COLOR_TEXT_LIGHT = "#FFFFFF"
        self.COLOR_TEXT_DARK = "#2C3E50"
        self.COLOR_PLACEHOLDER_BG = "#BDC3C7" # Silver

        master.configure(bg=self.COLOR_BG_PRIMARY)

        self.original_cv_image = None
        self.original_cv_image_rgb = None

        self.img_display_width = 200 # Slightly smaller for more space
        self.img_display_height = 200

        self.placeholder_pil_img = Image.new('RGB', (self.img_display_width, self.img_display_height), color=self.COLOR_PLACEHOLDER_BG)
        self.placeholder_photo = ImageTk.PhotoImage(self.placeholder_pil_img, master=self.master)

        # --- Main Content Frame ---
        main_frame = tk.Frame(master, bg=self.COLOR_BG_PRIMARY, padx=20, pady=20)
        main_frame.pack(expand=True, fill=tk.BOTH)

        # --- Title ---
        self.title_label = tk.Label(main_frame, text="Acne Detection Suite", font=("Segoe UI", 28, "bold"),
                                   bg=self.COLOR_BG_PRIMARY, fg=self.COLOR_TEXT_LIGHT)
        self.title_label.pack(pady=(0, 25), anchor="w")

        # --- Main Application Area (Controls + Results) ---
        app_area_frame = tk.Frame(main_frame, bg=self.COLOR_BG_SECONDARY)
        app_area_frame.pack(expand=True, fill=tk.BOTH, padx=5, pady=5) # Add some padding around this area
        app_area_frame.columnconfigure(1, weight=1) # Make right panel expandable

        # --- Left Panel (Controls) ---
        left_panel = tk.Frame(app_area_frame, bg=self.COLOR_BG_CONTENT, padx=20, pady=20,
                              borderwidth=1, relief=tk.SOLID) # tk.RIDGE or tk.GROOVE
        left_panel.grid(row=0, column=0, sticky="nswe", padx=(0,10), pady=0) # Sticky to fill height

        tk.Label(left_panel, text="Original Image", font=("Segoe UI", 14, "bold"),
                 bg=self.COLOR_BG_CONTENT, fg=self.COLOR_TEXT_DARK).pack(pady=(0,10), anchor="w")

        self.original_image_display = tk.Label(left_panel, image=self.placeholder_photo,
                                               bg=self.COLOR_PLACEHOLDER_BG, width=self.img_display_width,
                                               height=self.img_display_height,
                                               borderwidth=2, relief=tk.SUNKEN)
        self.original_image_display.image = self.placeholder_photo
        self.original_image_display.pack(pady=(0, 20))

        button_font = ("Segoe UI", 11, "bold")
        button_width = 18
        button_ipady = 7
        button_pady = 7

        self.btn_upload = tk.Button(left_panel, text="Upload Image", command=self.upload_image,
                                   bg=self.COLOR_ACCENT_PRIMARY, fg=self.COLOR_TEXT_LIGHT, font=button_font,
                                   width=button_width, relief=tk.FLAT, borderwidth=0,
                                   activebackground="#2980B9", activeforeground=self.COLOR_TEXT_LIGHT)
        self.btn_upload.pack(pady=button_pady, fill=tk.X, ipady=button_ipady)

        self.btn_process = tk.Button(left_panel, text="Process Image", command=self.process_image,
                                    bg=self.COLOR_ACCENT_SUCCESS, fg=self.COLOR_TEXT_LIGHT, font=button_font,
                                    width=button_width, relief=tk.FLAT, borderwidth=0, state=tk.DISABLED,
                                    activebackground="#27AE60", activeforeground=self.COLOR_TEXT_LIGHT)
        self.btn_process.pack(pady=button_pady, fill=tk.X, ipady=button_ipady)

        self.btn_reset = tk.Button(left_panel, text="Reset", command=self.reset_app,
                                  bg=self.COLOR_ACCENT_WARNING, fg=self.COLOR_TEXT_DARK, font=button_font,
                                  width=button_width, relief=tk.FLAT, borderwidth=0,
                                  activebackground="#F39C12", activeforeground=self.COLOR_TEXT_DARK)
        self.btn_reset.pack(pady=button_pady, fill=tk.X, ipady=button_ipady)

        self.btn_exit = tk.Button(left_panel, text="Exit Application", command=master.quit,
                                 bg=self.COLOR_ACCENT_DANGER, fg=self.COLOR_TEXT_LIGHT, font=button_font,
                                 width=button_width, relief=tk.FLAT, borderwidth=0,
                                 activebackground="#C0392B", activeforeground=self.COLOR_TEXT_LIGHT)
        self.btn_exit.pack(pady=(button_pady*2, button_pady), fill=tk.X, ipady=button_ipady) # More space before exit

        # --- Right Panel (Processing Stages) ---
        right_panel_container = tk.Frame(app_area_frame, bg=self.COLOR_BG_SECONDARY, padx=0, pady=0)
        right_panel_container.grid(row=0, column=1, sticky="nswe")
        right_panel_container.rowconfigure(0, weight=1) # Make internal grid expandable
        right_panel_container.columnconfigure(0, weight=1)

        right_panel = tk.Frame(right_panel_container, bg=self.COLOR_BG_CONTENT, padx=15, pady=15,
                                borderwidth=1, relief=tk.SOLID)
        right_panel.grid(row=0, column=0, sticky="nswe", padx=0, pady=0)

        self.image_displays = {}
        stages = [
            ("Normalization", "norm_image_display"),
            ("Edge Detection", "edge_image_display"),
            ("Filtered Spots", "filtered_image_display"),
            ("Detection Result", "result_image_display")
        ]

        label_font = ("Segoe UI", 12, "bold")
        num_cols = 2 # Number of columns for results
        for i, (text, key) in enumerate(stages):
            row, col = divmod(i, num_cols)

            # Make grid cells in right_panel expand
            right_panel.rowconfigure(row*2, weight=0) # For label
            right_panel.rowconfigure(row*2 + 1, weight=1) # For image
            right_panel.columnconfigure(col, weight=1)

            stage_frame = tk.Frame(right_panel, bg=self.COLOR_BG_CONTENT)
            stage_frame.grid(row=row*2, column=col, rowspan=2, padx=10, pady=10, sticky="nswe") # rowspan to combine label and image area

            text_label = tk.Label(stage_frame, text=text, font=label_font,
                                  bg=self.COLOR_BG_CONTENT, fg=self.COLOR_TEXT_DARK)
            text_label.pack(pady=(0,8), anchor="center")

            img_label = tk.Label(stage_frame, image=self.placeholder_photo, bg=self.COLOR_PLACEHOLDER_BG,
                                 width=self.img_display_width, height=self.img_display_height,
                                 borderwidth=2, relief=tk.SUNKEN)
            img_label.image = self.placeholder_photo
            img_label.pack(expand=True, fill=tk.BOTH) # Allow image label to expand
            self.image_displays[key] = img_label

    def _update_image_display(self, tk_label, cv_image_data, is_gray=False):
        if cv_image_data is None:
            tk_label.config(image=self.placeholder_photo, width=self.img_display_width, height=self.img_display_height)
            tk_label.image = self.placeholder_photo
            return

        if is_gray:
            if len(cv_image_data.shape) == 3 and cv_image_data.shape[2] == 1:
                pil_img = Image.fromarray(cv_image_data[:,:,0], 'L')
            elif len(cv_image_data.shape) == 2:
                pil_img = Image.fromarray(cv_image_data, 'L')
            else: # Fallback for unexpected gray formats
                if cv_image_data.dtype == np.float32 or cv_image_data.dtype == np.float64:
                    cv_image_data = (cv_image_data * 255).astype(np.uint8)
                pil_img = Image.fromarray(cv_image_data, 'L')
        else:
            pil_img = Image.fromarray(cv_image_data, 'RGB')

        # Aspect ratio preserving resize
        original_w, original_h = pil_img.size
        target_w, target_h = tk_label.winfo_width(), tk_label.winfo_height()
        if target_w < 10 or target_h < 10: # if widget not drawn yet, use default
            target_w, target_h = self.img_display_width, self.img_display_height

        ratio = min(target_w / original_w, target_h / original_h)
        new_w = int(original_w * ratio)
        new_h = int(original_h * ratio)

        pil_img_resized = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # Create a new image with placeholder background and paste resized image onto it
        final_pil_img = Image.new('RGB', (target_w, target_h), self.COLOR_PLACEHOLDER_BG)
        paste_x = (target_w - new_w) // 2
        paste_y = (target_h - new_h) // 2
        final_pil_img.paste(pil_img_resized, (paste_x, paste_y))

        photo_img = ImageTk.PhotoImage(final_pil_img, master=self.master)
        tk_label.config(image=photo_img, width=target_w, height=target_h)
        tk_label.image = photo_img


    def upload_image(self):
        filepath = filedialog.askopenfilename(
            title="Select an Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("All files", "*.*")]
        )
        if not filepath:
            return
        try:
            self.original_cv_image = cv2.imread(filepath)
            if self.original_cv_image is None:
                raise ValueError(f"OpenCV could not read image: {filepath}")
            self.original_cv_image_rgb = cv2.cvtColor(self.original_cv_image, cv2.COLOR_BGR2RGB)

            # Update display after widgets are drawn to get actual size
            self.master.update_idletasks()
            self._update_image_display(self.original_image_display, self.original_cv_image_rgb)
            self.btn_process.config(state=tk.NORMAL, bg=self.COLOR_ACCENT_SUCCESS, fg=self.COLOR_TEXT_LIGHT)
            self.reset_processing_displays()
        except Exception as e:
            messagebox.showerror("Image Load Error", f"Failed to load image: {filepath}\n{e}")
            self.original_cv_image = None
            self.original_cv_image_rgb = None
            self.btn_process.config(state=tk.DISABLED, bg=self.COLOR_BG_SECONDARY, fg=self.COLOR_TEXT_LIGHT)


    def process_image(self):
        if self.original_cv_image_rgb is None:
            messagebox.showerror("Error", "Please upload an image first.")
            return

        self.btn_process.config(text="Processing...", state=tk.DISABLED)
        self.master.update_idletasks() # Update UI to show "Processing..."

        try:
            im = self.original_cv_image_rgb.copy()

            # --- Parameters ---
            BTh = 0.55        # Binary Threshold for goff
            AreaMin = 30      # Minimum area for connected components
            AreaMax = 6000    # Maximum area for connected components (to exclude very large regions)

            # 1. Grayscale and Equalize
            imG = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
            imG_eq = cv2.equalizeHist(imG) # Equalize histogram

            # 2. Normalize Equalized Grayscale Image (imN)
            Maxim = np.max(imG_eq)
            Minim = np.min(imG_eq)
            imN_float = (imG_eq.astype(np.float32) - Minim) / (Maxim - Minim + 1e-6) # Add epsilon for stability
            imN_display = (imN_float * 255).astype(np.uint8)
            self._update_image_display(self.image_displays["norm_image_display"], imN_display, is_gray=True)

            # 3. HSV V-Channel Normalization
            imHsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
            V = imHsv[:, :, 2]
            V_float = (V.astype(np.float32) - np.min(V)) / (np.max(V) - np.min(V) + 1e-6)

            # 4. Difference Image (imGoff)
            imGoff = V_float - imN_float
            imGoff = cv2.normalize(imGoff, None, 0, 1, cv2.NORM_MINMAX) # Normalize to 0-1
            imGoff_blur = cv2.GaussianBlur(imGoff, (3, 3), 0) # Small blur

            # 5. Binarize imGoff
            imGoff_binary = (imGoff_blur > BTh).astype(np.uint8)

            # 6. Edge Detection on Binarized Goff (Optional display, Canny might be too aggressive here)
            # For display, let's show the binarized Goff as it's more relevant to the next steps
            # edges = cv2.Canny(imGoff_binary * 255, 100, 200)
            self._update_image_display(self.image_displays["edge_image_display"], (imGoff_binary * 255).astype(np.uint8), is_gray=True)

            # 7. Morphological Operations & Region Filtering
            imB = imGoff_binary > 0 # Boolean mask

            # Remove small objects
            imB_label = measure.label(imB.astype(np.uint8), connectivity=1) # Use connectivity=1 (4-connectivity)
            imB_clean = remove_small_objects(imB_label, min_size=AreaMin, connectivity=1)

            # Filter by Max Area
            regions_filtered_area = measure.regionprops(imB_clean)
            BW2 = np.zeros_like(imB_clean, dtype=bool)
            for region in regions_filtered_area:
                if region.area < AreaMax: # Check max area
                    for coord in region.coords:
                         BW2[coord[0], coord[1]] = True

            BW2_display = (BW2 * 255).astype(np.uint8)
            self._update_image_display(self.image_displays["filtered_image_display"], BW2_display, is_gray=True)

            # 8. Draw Bounding Boxes on Original Image
            result_im_with_boxes = im.copy()
            final_labeled_BW2 = measure.label(BW2.astype(np.uint8), connectivity=1)
            regions_final = measure.regionprops(final_labeled_BW2)

            num_acne = 0
            for region in regions_final:
                # bbox is (min_row, min_col, max_row, max_col)
                minr, minc, maxr, maxc = region.bbox
                cv2.rectangle(result_im_with_boxes, (minc, minr), (maxc, maxr), (255, 0, 0), 2) # Red BBoxes
                num_acne += 1

            self._update_image_display(self.image_displays["result_image_display"], result_im_with_boxes)

        except Exception as e:
            messagebox.showerror("Processing Error", f"An error occurred during image processing:\n{e}")
            import traceback
            traceback.print_exc()
        finally:
            self.btn_process.config(text="Process Image", state=tk.NORMAL)


    def reset_processing_displays(self):
        for key in self.image_displays:
            # self.master.update_idletasks() # Ensure widget dimensions are known
            self._update_image_display(self.image_displays[key], None)

    def reset_app(self):
        self.original_cv_image = None
        self.original_cv_image_rgb = None
        # self.master.update_idletasks()
        self._update_image_display(self.original_image_display, None)
        self.reset_processing_displays()
        self.btn_process.config(state=tk.DISABLED, bg=self.COLOR_BG_SECONDARY, fg=self.COLOR_TEXT_LIGHT, text="Process Image")
        self.btn_upload.config(bg=self.COLOR_ACCENT_PRIMARY) # Reset color just in case


if __name__ == "__main__":
    root = tk.Tk()
    app = AcneDetectorApp(root)
    root.mainloop()