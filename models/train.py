
from ultralytics import YOLO


model = YOLO('best.pt')

results = model.train(
        data='data.yaml',
        epochs=10,
        imgsz=640,
        batch=12,  # 减小批量大小
        device=0
)

    # 保存训练好的模型
model.save('trained_model_1.pt')