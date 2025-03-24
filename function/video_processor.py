import cv2
from .color import get_colors
from .tracking import track_objects
from .frame_processor import process_frame
from .image import resize_image
def process_video(cap, seg_model, emotion_model):
    all_results = []
    all_id_emotions = []  # 新增列表用于存储每帧的 ID 和情绪信息
    try:
        from deep_sort_realtime.deepsort_tracker import DeepSort
        tracker = DeepSort(max_age=5, n_init=2)
        colors = get_colors(10)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            detections = process_frame(frame, seg_model, emotion_model)
            tracks = tracker.update_tracks(detections, frame=frame)
            frame_id_emotions = []  # 存储当前帧的 ID 和情绪信息
            for track in tracks:
                if track.is_confirmed():
                    track_id = track.track_id
                    emotion_label = track.det_class
                    frame_id_emotions.append((track_id, emotion_label))

            all_id_emotions.append(frame_id_emotions)

            processed_frame = track_objects(frame, detections, colors, tracker)

            resized_frame = resize_image(processed_frame)
            cv2.imshow('YOLO Object Detection', resized_frame)
            all_results.append(resized_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        print(f"处理视频时出错: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return all_results, all_id_emotions