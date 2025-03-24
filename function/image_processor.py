import cv2
from .color import get_colors
from .tracking import track_objects
from .frame_processor import process_frame
from .image import resize_image

def process_image(file_path, seg_model, emotion_model):
    try:
        image = cv2.imread(file_path)
        if image is None:
            raise ValueError("无法读取图片文件")

        from deep_sort_realtime.deepsort_tracker import DeepSort
        tracker = DeepSort(max_age=5, n_init=2)
        colors = get_colors(10)
        detections = process_frame(image, seg_model, emotion_model)
        tracks = tracker.update_tracks(detections, frame=image)
        id_emotions = []  # 存储图片中目标的 ID 和情绪信息
        for track in tracks:
            if track.is_confirmed():
                track_id = track.track_id
                emotion_label = track.det_class
                id_emotions.append((track_id, emotion_label))

        processed_image = track_objects(image, detections, colors, tracker)
        resized_image = resize_image(processed_image)
        cv2.imshow('YOLO Object Detection', resized_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return [processed_image], [id_emotions]
    except Exception as e:
        print(f"处理图片时出错: {e}")
        return [], []