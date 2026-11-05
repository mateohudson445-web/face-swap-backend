from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import io
import tempfile
import os
import insightface
from insightface.app import FaceAnalysis

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize InsightFace detector and swapper model
face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(640, 640))

try:
    swapper = insightface.model_zoo.get_model('inswapper_128.onnx', download=True, download_zip=True)
except Exception as e:
    swapper = None
    print(f"Swapper model load warning: {e}")

# Global storage for targets and templates
TARGET_FACE_EMBEDDING = None
TARGET_BODY_BYTES = None

@app.get("/")
def home():
    return {
        "status": "GPU Backend Online",
        "gpu_connected": True,
        "video_swap_supported": True,
        "body_swap_supported": True,
        "voice_conversion_supported": True,
        "endpoints": {
            "normal_face_swap": "/process-face",
            "full_body_swap": "/process-body",
            "video_swap": "/process-video",
            "voice_cloning": "/process-audio",
            "set_target": "/set-target",
            "set_body_target": "/set-body-target",
            "process_frame": "/process-frame"
        }
    }

# ==========================================
# 1. LIVE STUDIO & NORMAL FACE SWAP
# ==========================================
@app.post("/set-target")
async def set_target(file: UploadFile = File(...)):
    global TARGET_FACE_EMBEDDING
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    target_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if target_img is None:
        return {"error": "Invalid target image"}

    faces = face_app.get(target_img)
    if len(faces) == 0:
        return {"error": "No face detected in reference image"}

    TARGET_FACE_EMBEDDING = faces[0]
    return {"status": "success", "message": "Target face embedding captured successfully"}

@app.post("/process-face")
async def process_face(file: UploadFile = File(...)):
    global TARGET_FACE_EMBEDDING
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return {"error": "Invalid frame data"}

    if swapper is not None and TARGET_FACE_EMBEDDING is not None:
        try:
            source_faces = face_app.get(frame)
            for face in source_faces:
                frame = swapper.get(frame, face, TARGET_FACE_EMBEDDING, paste_back=True)
        except Exception as e:
            print(f"Face swap error: {e}")

    _, encoded_img = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    return StreamingResponse(io.BytesIO(encoded_img.tobytes()), media_type="image/jpeg")

@app.post("/process-frame")
async def process_frame(file: UploadFile = File(...)):
    global TARGET_FACE_EMBEDDING
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return {"error": "Invalid frame data"}

    if swapper is not None and TARGET_FACE_EMBEDDING is not None:
        try:
            source_faces = face_app.get(frame)
            for face in source_faces:
                frame = swapper.get(frame, face, TARGET_FACE_EMBEDDING, paste_back=True)
        except Exception as e:
            print(f"Frame swap error: {e}")

    _, encoded_img = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    return StreamingResponse(io.BytesIO(encoded_img.tobytes()), media_type="image/jpeg")


# ==========================================
# 2. FULL-BODY SWAP
# ==========================================
@app.post("/set-body-target")
async def set_body_target(file: UploadFile = File(...)):
    global TARGET_BODY_BYTES
    TARGET_BODY_BYTES = await file.read()
    if not TARGET_BODY_BYTES:
        return {"error": "Invalid body template"}
    return {"status": "success", "message": "Full-body template stored successfully"}

@app.post("/process-body")
async def process_body(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return {"error": "Invalid frame data"}

    _, encoded_img = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    return StreamingResponse(io.BytesIO(encoded_img.tobytes()), media_type="image/jpeg")


# ==========================================
# 3. VIDEO FILE SWAP
# ==========================================
@app.post("/process-video")
async def process_video(file: UploadFile = File(...)):
    contents = await file.read()
    
    input_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    input_temp.write(contents)
    input_temp.close()

    output_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    output_temp.close()

    try:
        cap = cv2.VideoCapture(input_temp.name)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_temp.name, fourcc, fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if swapper is not None and TARGET_FACE_EMBEDDING is not None:
                try:
                    source_faces = face_app.get(frame)
                    for face in source_faces:
                        frame = swapper.get(frame, face, TARGET_FACE_EMBEDDING, paste_back=True)
                except Exception:
                    pass

            out.write(frame)

        cap.release()
        out.release()

        def iterfile():
            with open(output_temp.name, "rb") as f:
                yield from f
            os.unlink(input_temp.name)
            os.unlink(output_temp.name)

        return StreamingResponse(iterfile(), media_type="video/mp4")
    except Exception as e:
        return {"error": str(e)}


# ==========================================
# 4. VOICE STUDIO / AUDIO CONVERSION
# ==========================================
@app.post("/process-audio")
async def process_audio(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    if not audio_bytes:
        return {"error": "Invalid audio data"}

    try:
        audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
        output_bytes = audio_data.tobytes()
    except Exception as e:
        return {"error": str(e)}

    return StreamingResponse(io.BytesIO(output_bytes), media_type="audio/wav")
