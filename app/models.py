import db
from sqlalchemy import Integer, Column, JSON

class Detection(db.Base):
    __tablename__= "detection"
    id = Column(Integer, primary_key=True)
    total_objects = Column(Integer)
    object_count = Column(JSON)
    detections = Column(JSON)
