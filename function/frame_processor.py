from ultralytics import YOLO
from .color import get_colors
from .image import resize_image

def process_frame(frame, seg_model, emotion_model):
    # 实例分割
    seg_results = seg_model(frame)
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

                    person_image = resize_image(person_image, 640)
                    emotion_results = emotion_model(person_image)

                    if len(emotion_results[0].boxes) > 0:
                        emotion_class_id = int(emotion_results[0].boxes.cls[0].item())
                        emotion_names = emotion_model.names
                        emotion_label = emotion_names[emotion_class_id]
                    else:
                        emotion_label = 'Unknown'

                    # 检测结果添加到 detections 列表中
                    detections.append(([x1, y1, x2 - x1, y2 - y1], 1.0, emotion_label))

    return detections