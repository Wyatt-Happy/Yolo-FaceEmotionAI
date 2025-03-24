import tkinter as tk
import cv2
from tkinter import filedialog
from ultralytics import YOLO
import numpy as np
import sys
import os
from deep_sort_realtime.deepsort_tracker import DeepSort


def get_colors(num_colors):
    """
    生成指定数量的颜色列表
    :param num_colors: 所需颜色的数量
    :return: 颜色列表，每个颜色为 (B, G, R) 格式的元组
    """
    colors = []
    for i in range(num_colors):
        # 使用HSV颜色空间生成不同颜色，然后转换为BGR颜色空间
        hue = int(i * (180 / num_colors))
        hsv_color = np.uint8([[[hue, 255, 255]]])
        bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)
        color = tuple(int(c) for c in bgr_color[0][0])
        colors.append(color)
    return colors


def track_objects(frame, detections, colors, tracker):
    """
    使用 DeepSORT 进行多目标跟踪并绘制结果
    :param frame: 当前帧图像
    :param detections: 检测结果列表，每个元素为 (bounding_box, confidence, class_label)
    :param colors: 颜色列表
    :param tracker: DeepSORT 跟踪器
    :return: 处理后的帧图像
    """
    tracks = tracker.update_tracks(detections, frame=frame)
    person_count = 0
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        bbox = track.to_ltrb()
        x1, y1, x2, y2 = map(int, bbox)
        emotion_label = track.det_class

        # 选择颜色
        color = colors[person_count % len(colors)]
        person_count += 1

        # 绘制边界框
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # 计算标签的大小
        label_text = f"ID: {track_id} - {emotion_label}"
        # 减小字体大小
        font_scale = 0.6
        font_thickness = 1
        (label_width, label_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)

        # 绘制标签背景
        cv2.rectangle(frame, (x1, y1), (x1 + label_width, y1 + label_height), color, -1)

        # 绘制标签文本，使用调整后的字体大小和厚度
        cv2.putText(frame, label_text, (x1, y1 + label_height), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)

    return frame


class YOLOObjectDetectionGUI:
    def __init__(self):
        # 检查是否是打包后的环境
        if getattr(sys, 'frozen', False):
            # 如果是打包后的环境，使用 sys._MEIPASS 获取临时目录
            self.base_path = sys._MEIPASS
        else:
            # 如果是开发环境，使用当前目录
            self.base_path = os.path.dirname(os.path.abspath(__file__))

        # 加载实例分割模型
        seg_model_path = os.path.join(self.base_path, 'yolo11n-seg.pt')
        try:
            self.seg_model = YOLO(seg_model_path)
        except Exception as e:
            print(f"实例分割模型加载失败: {e}")
            sys.exit(1)

        # 加载情绪识别模型
        emotion_model_path = os.path.join(self.base_path, 'trained_model.pt')
        try:
            self.emotion_model = YOLO(emotion_model_path)
        except Exception as e:
            print(f"情绪识别模型加载失败: {e}")
            sys.exit(1)

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

    # 图像缩放函数
    def resize_image(self, image, target_width=640):
        height, width = image.shape[:2]
        aspect_ratio = height / width
        target_height = int(target_width * aspect_ratio)
        dim = (target_width, target_height)
        return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    def process_frame(self, frame):
        # 实例分割
        seg_results = self.seg_model(frame)
        detections = []
        person_count = 0
        # 假设最多需要 10 种颜色，可根据实际情况调整
        colors = get_colors(10)

        for r in seg_results:
            for j, mask in enumerate(r.masks.xy):
                class_id = int(r.boxes.cls[j].item())
                if class_id == 0:
                    x1, y1, x2, y2 = map(int, r.boxes.xyxy[j].cpu().numpy())
                    person_image = frame[y1:y2, x1:x2]

                    if person_image.size > 0:
                        person_image = cv2.resize(person_image, (640, 640))
                        emotion_results = self.emotion_model(person_image)

                        if len(emotion_results[0].boxes) > 0:
                            emotion_class_id = int(emotion_results[0].boxes.cls[0].item())
                            emotion_names = self.emotion_model.names
                            emotion_label = emotion_names[emotion_class_id]
                        else:
                            emotion_label = 'Unknown'

                        # 检测结果添加到 detections 列表中
                        detections.append(([x1, y1, x2 - x1, y2 - y1], 1.0, emotion_label))

        return detections

    def setup_tracking(self):
        """
        设置跟踪所需的参数，包括跟踪器和颜色列表
        """
        tracker = DeepSort(max_age=5, n_init=2)
        colors = get_colors(10)
        return tracker, colors

    def show_frame(self, frame):
        """
        显示处理后的帧
        """
        resized_frame = self.resize_image(frame)
        cv2.imshow('YOLO Object Detection', resized_frame)
        return resized_frame

    def process_video(self, cap):
        all_results = []
        all_id_emotions = []  # 新增列表用于存储每帧的 ID 和情绪信息
        try:
            tracker, colors = self.setup_tracking()
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                detections = self.process_frame(frame)
                tracks = tracker.update_tracks(detections, frame=frame)
                frame_id_emotions = []  # 存储当前帧的 ID 和情绪信息
                for track in tracks:
                    if track.is_confirmed():
                        track_id = track.track_id
                        emotion_label = track.det_class
                        frame_id_emotions.append((track_id, emotion_label))

                all_id_emotions.append(frame_id_emotions)

                processed_frame = track_objects(frame, detections, colors, tracker)
                resized_frame = self.show_frame(processed_frame)
                all_results.append(resized_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except Exception as e:
            print(f"处理视频时出错: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
        return all_results, all_id_emotions

    def process_image(self, file_path):
        try:
            image = cv2.imread(file_path)
            if image is None:
                raise ValueError("无法读取图片文件")

            tracker, colors = self.setup_tracking()
            detections = self.process_frame(image)
            tracks = tracker.update_tracks(detections, frame=image)
            id_emotions = []  # 存储图片中目标的 ID 和情绪信息
            for track in tracks:
                if track.is_confirmed():
                    track_id = track.track_id
                    emotion_label = track.det_class
                    id_emotions.append((track_id, emotion_label))

            processed_image = track_objects(image, detections, colors, tracker)
            resized_image = self.show_frame(processed_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return [processed_image], [id_emotions]
        except Exception as e:
            print(f"处理图片时出错: {e}")
            return [], []

    def detect_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png;*.jpeg")])
        if file_path:
            processed_image, id_emotions = self.process_image(file_path)
            # 可以在这里对 id_emotions 进行进一步处理
            print(id_emotions)

    def detect_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi")])
        if file_path:
            cap = cv2.VideoCapture(file_path)
            all_results, all_id_emotions = self.process_video(cap)
            # 可以在这里对 all_id_emotions 进行进一步处理
            print(all_id_emotions)

    def detect_camera(self):
        cap = cv2.VideoCapture(0)
        all_results, all_id_emotions = self.process_video(cap)
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