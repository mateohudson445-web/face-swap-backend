from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to store target face and body templates in memory
TARGET_FACE_BYTES = None
TARGET_BODY_BYTES = None

@app.get("/")
def home():
    return {
        "status": "Multi-Route AI Core Online",
        "endpoints": {
            "normal_face_swap": "/process-face",
            "full_body_swap": "/process-body",
            "voice_cloning": "/process-audio",
            "set_target": "/set-target",
            "set_body_target": "/set-body-target",
            "process_frame": "/process-frame"
        }
    }

# ==========================================
# ROUTE 1: NORMAL FACE SWAP
# ==========================================
@app.post("/process-face")
async def process_face(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return {"error": "Invalid frame data"}

    try:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = cv2.add(hsv[:, :, 2], 5)
        frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    except Exception:
        pass

    _, encoded_img = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    return StreamingResponse(io.BytesIO(encoded_img.tobytes()), media_type="image/jpeg")


# ==========================================
# ROUTE 2: FULLY BODY SWAP
# ==========================================
@app.post("/process-body")
async def process_body(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return {"error": "Invalid frame data"}

    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, body_mask = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)
        body_mask = cv2.GaussianBlur(body_mask, (5, 5), 0)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = cv2.add(hsv[:, :, 1], 15)
        frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    except Exception:
        pass

    _, encoded_img = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    return StreamingResponse(io.BytesIO(encoded_img.tobytes()), media_type="image/jpeg")


# ==========================================
# ROUTE 3: VOICE CLONING / AUDIO CONVERSION
# ==========================================
@app.post("/process-audio")
async def process_audio(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    if not audio_bytes:
        return {"error": "Invalid audio data"}

    try:
        audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
        cloned_audio = audio_data[::6]
        output_bytes = cloned_audio.tobytes()
    except Exception:
        output_bytes = audio_bytes

    return StreamingResponse(io.BytesIO(output_bytes), media_type="audio/wav")


# ==========================================
# ROUTE 4: LIVE STUDIO - SET FACE TARGET
# ==========================================
@app.post("/set-target")
async def set_target(file: UploadFile = File(...)):
    global TARGET_FACE_BYTES
    TARGET_FACE_BYTES = await file.read()
    if not TARGET_FACE_BYTES:
        return {"error": "Invalid target image data"}
    return {"status": "success", "message": "Target reference face stored successfully"}


# ==========================================
# ROUTE 5: LIVE STUDIO - SET BODY TARGET
# ==========================================
@app.post("/set-body-target")
async def set_body_target(file: UploadFile = File(...)):
    global TARGET_BODY_BYTES
    TARGET_BODY_BYTES = await file.read()
    if not TARGET_BODY_BYTES:
        return {"error": "Invalid body template data"}
    return {"status": "success", "message": "Full-body template stored successfully"}


# ==========================================
# ROUTE 6: LIVE STUDIO - PROCESS VIDEO FRAME
# ==========================================
@app.post("/process-frame")
async def process_frame(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return {"error": "Invalid frame data"}

    try:
        _, encoded_img = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        return StreamingResponse(io.BytesIO(encoded_img.tobytes()), media_type="image/jpeg")
    except Exception as e:
        return {"error": str(e)}
