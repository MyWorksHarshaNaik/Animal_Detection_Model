import tkinter as tk
from tkinter import filedialog, Label, Button
import cv2
from PIL import Image, ImageTk
import torch
from ultralytics import YOLO

class AnimalPredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Animal Detection Model")

        self.model = YOLO('./Models/best1.pt')  # Load the YOLO model

        # GUI Elements
        self.label = Label(root, text="Choose an image or video file")
        self.label.pack(pady=10)

        self.img_button = Button(root, text="Select Image", command=self.predict_image)
        self.img_button.pack(pady=5)

        self.vid_button = Button(root, text="Select Video", command=self.predict_video)
        self.vid_button.pack(pady=5)

        self.canvas = tk.Canvas(root, width=800, height=600)
        self.canvas.pack()

    def predict_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if file_path:
            results = self.model(file_path)
            annotated_frame = results[0].plot()

            # Convert annotated image to display in Tkinter
            img = Image.fromarray(annotated_frame)
            img = img.resize((800, 600))
            img_tk = ImageTk.PhotoImage(img)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
            self.canvas.image = img_tk  # Prevent garbage collection

    def predict_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
        if file_path:
            cap = cv2.VideoCapture(file_path)
            out = None

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                results = self.model(frame)
                annotated_frame = results[0].plot()

                # Convert BGR to RGB for display
                frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                img = img.resize((800, 600))
                img_tk = ImageTk.PhotoImage(img)

                self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
                self.canvas.image = img_tk
                self.root.update()

            cap.release()

if __name__ == "__main__":
    root = tk.Tk()
    app = AnimalPredictorApp(root)
    root.mainloop()
