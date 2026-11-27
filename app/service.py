import cv2 as cv
import numpy as np
from ultralytics import YOLO
from sqlalchemy.orm import Session
import models

model = YOLO("yolo11n.pt")


def check_img(img_bytes:bytes) -> np.ndarray:
    img = cv.imdecode(np.frombuffer(img_bytes, np.uint8), cv.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid img")

    return img


def detect_object(img:np.ndarray) -> list[dict]:
    results = model(img)
    if not results or results[0].boxes is None:
        return []
    
    detections = []

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        obj = {"object_name":model.names[cls_id],
                     "confidence":round(conf, 3)}

        detections.append(obj)

    return detections



def count(detections:list[dict]) -> dict:
    objs = {}

    for detect in detections:
        objs[detect["object_name"]] = objs.get(detect["object_name"], 0) + 1

    summary = {"total_objects":len(detections),
               "object_count":objs,
               "detections":detections}

    return summary



def save_detection(db_session:Session, summary:dict):
    try:
        data = models.Detection(**summary)
        db_session.add(data)
        db_session.commit()
        db_session.refresh(data)
        return data.id
    except Exception:
        db_session.rollback()
        raise 


def get_detection(detection_id:int, db_session:Session):
    data = db_session.query(models.Detection).filter(models.Detection.id == detection_id).first()
    if data:
        return data
    else:
        raise ValueError("detection not found")