from typing import Annotated
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from model.model import predict_pipeline
from model.model import __version__ as model_version
import io

app = FastAPI()

@app.get("/")
def home():
    return {"helth_check": "OK", "model_version": model_version}


# it will work well for large files like images, videos, large binaries, etc. without consuming all the memory - doc 
@app.post("/predict")
async def predict(
    file: Annotated[UploadFile, File(description="A file read as UploadFile")],
):
    contents = await file.read()
    image_pil = predict_pipeline(contents)
    
    buffer = io.BytesIO()
    image_pil.save(buffer, format="JPEG")
    buffer.seek(0)

    return StreamingResponse(buffer, media_type="image/jpeg")
