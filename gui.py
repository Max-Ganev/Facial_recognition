import os
import sys
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

try:
    from app import identify, load_vector
    from preprocess import clean_and_resize
    from utils import preprocess_image
except ImportError:
    messagebox.showerror("Error", "Make sure app.py, preprocess.py, and utils.py are in the same folder!")
    sys.exit()


def _get_cascade_path():
    bundled_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
    if os.path.exists(bundled_path):
        return bundled_path
    raise FileNotFoundError(
        "Could not find haarcascade_frontalface_default.xml bundled with opencv-python. "
        "Make sure opencv-python (not just opencv-python-headless) is installed."
    )


class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("From-Scratch Dual Facial Recognition Engine")
        self.root.geometry("650x650")
        self.root.configure(bg="#2d2d2d")

        # These were referenced in toggle_mode() before ever being set --
        # caused an AttributeError on the very first mode switch.
        self.cap = None
        self.selected_image_path = None

        self.title_label = tk.Label(
            root, text="Facial Recognition Software",
            font=("Arial", 18, "bold"), fg="#ffffff", bg="#2d2d2d"
        )
        self.title_label.pack(pady=15)

        self.mode_frame = tk.Frame(root, bg="#2d2d2d")
        self.mode_frame.pack(pady=10)

        self.mode_var = tk.StringVar(value="Static")

        self.static_radio = tk.Radiobutton(
            self.mode_frame, text="Static File Uploader", variable=self.mode_var,
            value="Static", command=self.toggle_mode, font=("Arial", 11, "bold"),
            fg="#ffffff", bg="#2d2d2d", selectcolor="#1e1e1e", activebackground="#2d2d2d"
        )
        self.static_radio.pack(side="left", padx=(0, 20))

        self.webcam_radio = tk.Radiobutton(
            self.mode_frame, text="Live Continuous Camera", variable=self.mode_var,
            value="Webcam", command=self.toggle_mode, font=("Arial", 11, "bold"),
            fg="#ffffff", bg="#2d2d2d", selectcolor="#1e1e1e", activebackground="#2d2d2d"
        )
        self.webcam_radio.pack(side="left")

        self.image_frame = tk.Frame(root, width=320, height=320, bg="#1e1e1e", bd=2, relief="groove")
        self.image_frame.pack_propagate(False)
        self.image_frame.pack(pady=10)

        self.image_label = tk.Label(self.image_frame, text="No Image Selected", fg="#888888", bg="#1e1e1e")
        self.image_label.pack(expand=True)

        self.btn_frame = tk.Frame(root, bg="#2d2d2d")
        self.btn_frame.pack(pady=15)

        self.upload_btn = tk.Button(
            self.btn_frame, text="Upload Test Image", command=self.upload_image,
            font=("Arial", 11, "bold"), fg="#ffffff", bg="#007acc", activebackground="#005999",
            padx=15, pady=8, bd=0
        )
        self.upload_btn.pack(side="left", padx=10)

        self.scan_btn = tk.Button(
            self.btn_frame, text="Scan Static Face", command=self.process_static_scan,
            font=("Arial", 11, "bold"), fg="#ffffff", bg="#4caf50", activebackground="#388e3c",
            padx=15, pady=8, bd=0, state="disabled"
        )
        self.scan_btn.pack(side="left", padx=10)

        self.result_label = tk.Label(
            root, text="Result: Waiting for Input...",
            font=("Arial", 14, "bold"), fg="#ffb74d", bg="#2d2d2d"
        )
        self.result_label.pack(pady=10)  # this was missing -- label existed but never showed

        cascade_path = _get_cascade_path()
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            print(f"\n[!] Warning: Could not load cascade from {cascade_path}. Bounding boxes are offline.")

    def toggle_mode(self):
        current_mode = self.mode_var.get()

        if current_mode == "Static":
            if self.cap and self.cap.isOpened():
                self.cap.release()
            self.cap = None
            self.upload_btn.configure(state="normal")
            self.scan_btn.configure(state="disabled" if not self.selected_image_path else "normal")
            self.image_label.configure(image="", text="No Image Selected")
            self.result_label.configure(text="Result: Waiting for Input...", fg="#ffb74d")

        elif current_mode == "Webcam":
            self.upload_btn.configure(state="disabled")
            self.scan_btn.configure(state="disabled")
            self.image_label.configure(text="")
            self.result_label.configure(text="Result: Initializing Camera...", fg="#ffb74d")

            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            # Give the UI a moment to draw before the feed loop starts.
            # (previously this was called both via after() AND immediately,
            # starting two overlapping polling loops -- just schedule once.)
            self.root.after(100, self.run_continuous_webcam)

    def upload_image(self):
        file_path = filedialog.askopenfilename(
            title="Select an Image",
            filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")]
        )
        if file_path:
            self.selected_image_path = file_path
            img = Image.open(file_path)
            img.thumbnail((300, 300))

            self.img_tk = ImageTk.PhotoImage(img)
            self.image_label.configure(image=self.img_tk, text="")
            self.scan_btn.configure(state="normal")
            self.result_label.configure(text="Result: Image loaded. Ready to scan!", fg="#ffb74d")

    def process_static_scan(self):
        if not self.selected_image_path:
            return
        try:
            # Use the SAME preprocessing as training/webcam inference --
            # hand-rolling .convert('L').resize(...) here used a different
            # resampling filter than preprocess_image(), which silently
            # degrades match accuracy (see utils.py's docstring).
            processed = preprocess_image(self.selected_image_path)
            processed.save("test.png")
            matched_name = identify("test.png")
            self.result_label.configure(text=f"Match Found! Identity: {matched_name}", fg="#81c784")
            messagebox.showinfo("Analysis Complete", f"Static scan match: {matched_name}")
        except Exception as e:
            messagebox.showerror("Error", f"Computation failed:\n{str(e)}")

    def run_continuous_webcam(self):
        if self.mode_var.get() != "Webcam" or self.cap is None:
            return

        ret, frame = self.cap.read()
        if ret:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            matched_name = "Scanning..."

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                try:
                    face_crop = gray_frame[y:y + h, x:x + w]
                    face_pil = Image.fromarray(face_crop)
                    processed = preprocess_image(face_pil)
                    processed.save("test.png")
                    matched_name = identify("test.png")
                    cv2.putText(
                        frame, matched_name, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
                    )
                except Exception:
                    pass

            # Moved OUTSIDE the for loop -- previously this only ran when at
            # least one face was detected, so an empty frame (nobody in
            # view yet) never got drawn at all, leaving a black square.
            cv2_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2_image)
            img.thumbnail((320, 320))
            self.img_tk = ImageTk.PhotoImage(img)
            self.image_label.configure(image=self.img_tk)
            self.result_label.configure(text=f"Live Scan: {matched_name}", fg="#81c784")

        self.root.after(50, self.run_continuous_webcam)


if __name__ == "__main__":
    window = tk.Tk()
    app = FaceRecognitionApp(window)
    window.mainloop()