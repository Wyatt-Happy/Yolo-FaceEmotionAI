from setuptools import setup, find_packages
import os

# 获取项目根目录
root_dir = os.path.dirname(os.path.abspath(__file__))

setup(
    name='FaceEmotionAI',
    version='1.0.0',
    author='Wyatt',
    author_email='1300843619@qq.com',
    description='A face emotion recognition project using YOLO algorithm and fer2013 dataset',
    long_description=open(os.path.join(root_dir, 'README.md')).read() if os.path.exists(
        os.path.join(root_dir, 'README.md')) else '',
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/Yolo-FaceEmotionAI',

    # 自动发现所有的包
    packages=find_packages(),

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],

    python_requires='>=3.6',

    install_requires=[
        'numpy==1.24.3',
        'opencv-python==4.11.0.86',
        'ultralytics==8.3.86',
        'deep_sort_realtime==1.3.2',
    ],

    # 包含非 Python 文件
    package_data={
        '': ['*.pt', '*.yaml'],
        'models': ['best.pt', 'yolo11n-seg.pt', 'train.py', 'data.yaml'],
        'gui': ['main_gui.py'],
        'utils': ['model_onnx.py', 'setup.py']
    },
    include_package_data=True,

    # 命令行入口点
    entry_points={
        'console_scripts': [
            'yolo-faceemotionai = gui.main_gui:main'
        ]
    }
)
