import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort

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