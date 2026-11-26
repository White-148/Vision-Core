import cv2 as cv
import numpy as np
from ultralytics import YOLO

model = YOLO("yolo11n.pt")


def check_img(img_bytes):
    img = cv.imdecode(np.frombuffer(img_bytes, np.uint8), cv.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid img")

    return img


def detection(img):
    results = model(img)
    detections = []

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        obj = {"object name":model.names[cls_id],
                     "confidence":round(conf, 3)}

        detections.append(obj)

    return detections

