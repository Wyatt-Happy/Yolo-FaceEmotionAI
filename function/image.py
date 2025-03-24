import cv2

def resize_image(image, target_width=640):
    height, width = image.shape[:2]
    aspect_ratio = height / width
    target_height = int(target_width * aspect_ratio)
    dim = (target_width, target_height)
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)