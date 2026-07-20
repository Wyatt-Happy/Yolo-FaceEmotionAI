<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue" alt="Python" />
  <img src="https://img.shields.io/badge/YOLO-11n-seg-brightgreen" alt="YOLO" />
  <img src="https://img.shields.io/badge/DeepSORT-1.3.2-orange" alt="DeepSORT" />
  <img src="https://img.shields.io/badge/dataset-fer2013-purple" alt="fer2013" />
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License" />
</p>

<h1 align="center">Yolo-FaceEmotionAI</h1>
<p align="center"><strong>YOLO 人脸检测 + DeepSORT 多目标追踪 + 8 类情绪识别</strong></p>
<p align="center">图像 / 视频 / 摄像头实时画面，一键检测人脸情绪——Anger、Contempt、Disgust、Fear、Happy、Neutral、Sad、Surprise</p>

---

## ✨ 能力

| 能力 | 说明 |
|------|------|
| 🔍 **实例分割** | YOLO11n-seg 精准定位人脸区域，提取单人 ROI |
| 😊 **情绪识别** | fer2013 训练微调，8 种情绪分类（Anger / Contempt / Disgust / Fear / Happy / Neutral / Sad / Surprise） |
| 🏷️ **多目标追踪** | DeepSORT 为每张人脸分配唯一 ID，跨帧持续追踪 |
| 🖼️ **图像检测** | 选择单张图片 → 标注人脸框 + 情绪标签 + 追踪 ID |
| 🎬 **视频检测** | 选择视频文件 → 逐帧处理，实时预览，按 Q 退出 |
| 📷 **摄像头** | 调用摄像头实时检测，按 Q 退出 |
| 🖥️ **GUI 界面** | Tkinter 三按钮界面，无需命令行操作 |

---

## 🚀 快速开始

```bash
# 1. 创建环境
conda create -n face-emotion-ai python=3.10 -y
conda activate face-emotion-ai

# 2. 安装依赖
pip install -r requirements.txt

# 3. 启动 GUI
python gui/main_gui.py
```

点击 **Select Image** / **Select Video** / **Start Camera** 即可开始检测。

---

## 🔄 检测流程

```
┌──────────────────────────────────────────────────────────────────┐
│  1. 输入                                                         │
│     图像文件 / 视频文件 / 摄像头 (cv2.VideoCapture)                │
└──────────────────────────┬───────────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│  2. 实例分割 (YOLO11n-seg)                                       │
│                                                                  │
│  检测所有 class_id=0（人）的目标                                  │
│  提取每个人脸区域的 ROI  (x1, y1, x2, y2)                         │
└──────────────────────────┬───────────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│  3. 情绪识别 (best.pt)                                           │
│                                                                  │
│  ROI → resize 640×640 → YOLO 分类                                │
│  输出: Happy / Sad / Anger / Neutral / Surprise / ...            │
└──────────────────────────┬───────────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│  4. DeepSORT 追踪                                                │
│                                                                  │
│  detection → tracker.update_tracks()                             │
│  分配 track_id → 绑定 emotion_label                              │
└──────────────────────────┬───────────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│  5. 可视化输出                                                    │
│                                                                  │
│  每人绘制: 边界框 + ID:3 - Happy                                 │
│  cv2.imshow 实时显示 / 按 Q 退出                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 🏗️ 架构

```
输入（图片/视频/摄像头）
         │
    ┌────┴────┐
    │  GUI    │  ← tkinter 三按钮界面 (gui/main_gui.py)
    └────┬────┘
         ▼
    ┌────────┐
    │ model  │  ← 加载 YOLO11n-seg + best.pt (function/model_loader.py)
    │ loader │
    └───┬────┘
        ▼
    ┌──────────┐
    │  frame   │  ← 单帧：实例分割 → ROI → 情绪分类 (function/frame_processor.py)
    │ processor│
    └───┬──────┘
        ▼
    ┌──────────┐
    │ tracking │  ← DeepSORT 多目标追踪 + ID 绑定 (function/tracking.py)
    └───┬──────┘
        ▼
    ┌──────────┐
    │  output  │  ← cv2.imshow 标注帧输出
    └──────────┘
```

### 数据流关键节点

| 阶段 | 输入 | 输出 | 核心模块 |
|------|------|------|---------|
| 模型加载 | `models/yolo11n-seg.pt` `models/best.pt` | seg_model, emotion_model | `model_loader.py` |
| 实例分割 | 原始帧 | 人脸 ROI 列表 | `frame_processor.py` |
| 情绪分类 | 人脸 ROI | emotion_label | `frame_processor.py` |
| 目标追踪 | detections + frame | track_id + 标注帧 | `tracking.py` |
| 流水线 | 视频帧序列 | 全部标注帧 + ID-情绪映射 | `video_processor.py` / `image_processor.py` |

---

## 📂 项目结构

```
Yolo-FaceEmotionAI/
├── function/               # 核心逻辑层
│   ├── model_loader.py     # 模型加载器
│   ├── frame_processor.py  # 单帧检测（分割 + 情绪）
│   ├── video_processor.py  # 视频流水线
│   ├── image_processor.py  # 图片流水线
│   ├── tracking.py         # DeepSORT 多目标追踪
│   ├── color.py            # 颜色生成工具
│   └── image.py            # 图像缩放工具
├── gui/                    # GUI 入口
│   └── main_gui.py         # Tkinter 界面（三按钮）
├── models/                 # 模型与训练
│   ├── yolo11n-seg.pt      # YOLO11 实例分割预训练权重
│   ├── best.pt             # 情绪识别训练权重
│   ├── data.yaml           # fer2013 数据集配置（8 类）
│   └── train.py            # 训练脚本
├── utils/                  # 辅助工具
│   ├── setup.py            # setuptools 打包配置
│   └── model_onnx.py       # YOLO → ONNX 格式转换
├── requirements.txt        # 依赖清单
└── LICENSE                 # MIT
```

---

## 🎯 情绪类别（fer2013）

| # | 情绪 | 英文 |
|---|------|------|
| 0 | 愤怒 | Anger |
| 1 | 蔑视 | Contempt |
| 2 | 厌恶 | Disgust |
| 3 | 恐惧 | Fear |
| 4 | 开心 | Happy |
| 5 | 中性 | Neutral |
| 6 | 悲伤 | Sad |
| 7 | 惊讶 | Surprise |

---

## 🔧 训练自己的模型

```bash
# 1. 准备 fer2013 数据集，按 data.yaml 配置路径
# 2. 运行训练
python models/train.py
```

配置文件 `models/data.yaml`：

```yaml
train: "data/train/images"
val:   "data/valid/images"
test:  "data/test/images"
nc: 8
names: ["Anger","Contempt","Disgust","Fear","Happy","Neutral","Sad","Surprise"]
network: "yolo11n"
```

---

## 🔄 模型导出 ONNX

```bash
python utils/model_onnx.py
```

将 `best.pt` 和 `yolo11n-seg.pt` 导出为 ONNX 格式，提升跨平台部署兼容性。

---

<p align="center">张允泽 · MIT License</p>
# 友情链接
[linux.do](https://linux.do)
