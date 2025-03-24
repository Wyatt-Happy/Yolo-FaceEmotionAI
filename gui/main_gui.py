import tkinter as tk
from tkinter import filedialog
import cv2
from function.model_loader import load_models
from function.image_processor import process_image
from function.video_processor import process_video


class YOLOObjectDetectionGUI:
    def __init__(self):
        self.seg_model, self.emotion_model = load_models()

        # 创建主窗口
        self.root = tk.Tk()
        self.root.title("YOLO Object Detection")
        self.root.geometry("320x320")

        # 创建一个外层框架用于垂直居中
        self.outer_frame = tk.Frame(self.root)
        self.outer_frame.pack(expand=True)

        # 创建一个内层框架用于放置按钮
        self.button_frame = tk.Frame(self.outer_frame)
        self.button_frame.pack()

        # 创建按钮
        self.create_buttons()

    def detect_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png;*.jpeg")])
        if file_path:
            processed_image, id_emotions = process_image(file_path, self.seg_model, self.emotion_model)
            # 可以在这里对 id_emotions 进行进一步处理
            print(id_emotions)

    def detect_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi")])
        if file_path:
            cap = cv2.VideoCapture(file_path)
            all_results, all_id_emotions = process_video(cap, self.seg_model, self.emotion_model)
            # 可以在这里对 all_id_emotions 进行进一步处理
            print(all_id_emotions)

    def detect_camera(self):
        cap = cv2.VideoCapture(0)
        all_results, all_id_emotions = process_video(cap, self.seg_model, self.emotion_model)
        # 可以在这里对 all_id_emotions 进行进一步处理
        print(all_id_emotions)

    def create_buttons(self):
        image_button = tk.Button(self.button_frame, text="Select Image", command=self.detect_image, width=20, height=2)
        image_button.pack(pady=10)

        video_button = tk.Button(self.button_frame, text="Select Video", command=self.detect_video, width=20, height=2)
        video_button.pack(pady=10)

        camera_button = tk.Button(self.button_frame, text="Start Camera", command=self.detect_camera, width=20, height=2)
        camera_button.pack(pady=10)

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = YOLOObjectDetectionGUI()
    app.run()