import db
from sqlalchemy import Integer, Column, JSON
from pydantic import BaseModel

class Detection(db.Base):
    __tablename__= "detection"
    id = Column(Integer, primary_key=True)
    total_objects = Column(Integer)
    object_count = Column(JSON)
    detections = Column(JSON)

class DetectionItems(BaseModel):
    object_name: str
    confidence: float

class DetectionResponse(BaseModel):
    total_objects: int
    object_count: dict[str, int]
    detections: list[DetectionItems]
