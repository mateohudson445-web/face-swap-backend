from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import io

app = FastAPI()

# Allow your video call app frontend to talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"status": "Face Swap Backend is Online"}

@app.post("/process-frame")
async def process_frame(file: UploadFile = File(...)):
    # Read incoming image frame from the video call stream
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if frame is None:
        return {"error": "Invalid frame data"}

    # Placeholder for Face-Swap / DeepFace / Voice engine processing
    # This keeps the framework ultra-lightweight for your first deployment
    height, width, _ = frame.shape
    
    return {
        "status": "Frame received successfully",
        "resolution": f"{width}x{height}",
        "faces_detected": 0 # We will hook up ONNX models here next
    }
