import os
import sys
from ultralytics import YOLO


def load_models():
    # 检查是否是打包后的环境
    if getattr(sys, 'frozen', False):
        # 如果是打包后的环境，使用 sys._MEIPASS 获取临时目录
        base_path = sys._MEIPASS
    else:
        # 如果是开发环境，获取 function 文件夹的上级目录
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 构建实例分割模型路径
    seg_model_path = os.path.join(base_path, 'models', 'yolo11n-seg.pt')
    try:
        seg_model = YOLO(seg_model_path)
    except Exception as e:
        print(f"实例分割模型加载失败: {e}")
        raise

    # 构建情绪识别模型路径
    emotion_model_path = os.path.join(base_path, 'models', 'best.pt')
    try:
        emotion_model = YOLO(emotion_model_path)
    except Exception as e:
        print(f"情绪识别模型加载失败: {e}")
        raise

    return seg_model, emotion_model
