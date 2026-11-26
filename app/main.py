from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from sqlalchemy.orm import Session
import db
import service
db.Base.metadata.create_all(db.engine)

app = FastAPI()

@app.get("/")
def root():
    return {"Message":"It is working"}


@app.post("/detect")
async def create_detection(file:UploadFile = File(...), db:Session = Depends(db.get_db)):
    img_bytes = await file.read()

    try:
        img = service.check_img(img_bytes)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid image")

    detections = service.detect_object(img)
    summary = service.count(detections)

    try:
        detection_id = service.save_detection(db, summary)
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to save detection")
    
    return {"detection_id":detection_id}
