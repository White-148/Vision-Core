from fastapi import FastAPI, UploadFile, File, HTTPException
import service

app = FastAPI()

@app.get("/")
def root():
    return {"Message":"It is working"}


@app.post("/detect")
async def detect(file:UploadFile = File(...)):
    img_bytes = await file.read()

    try:
        dimensions = service.get_dimensions(img_bytes)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid image")

    return dimensions
