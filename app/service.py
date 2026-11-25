import cv2 as cv
import numpy as np


def get_dimensions(img_bytes):
    img = cv.imdecode(np.frombuffer(img_bytes, np.uint8), cv.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid img")
    
    height, width, channels = img.shape[:]

    return {"height":height,
            "width":width,
            "channels":channels}
