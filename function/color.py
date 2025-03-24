import numpy as np
import cv2

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