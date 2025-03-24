import os
from ultralytics import YOLO

# 获取 utils 文件夹的上级目录，即项目根目录
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 构建 trained_model.pt 的完整路径
trained_model_path = os.path.join(root_dir, 'models', 'best.pt')
# 构建 yolo11n-seg.pt 的完整路径
yolo_seg_model_path = os.path.join(root_dir, 'models', 'yolo11n-seg.pt')


# 定义一个函数来进行模型转换
def convert_model_to_onnx(model_path):
    try:
        # 加载模型
        model = YOLO(model_path)
        # 转换模型为 ONNX 格式
        success = model.export(format='onnx')
        if success:
            print(f"{os.path.basename(model_path)} 模型成功转换为 ONNX 格式！")
        else:
            print(f"{os.path.basename(model_path)} 模型转换失败。")
    except Exception as e:
        print(f"转换 {os.path.basename(model_path)} 模型时出现错误: {e}")


# 转换 trained_model.pt
convert_model_to_onnx(trained_model_path)
# 转换 yolo11n-seg.pt
convert_model_to_onnx(yolo_seg_model_path)
