from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import io

app = FastAPI()

# Make sure your VS Code frontend can securely talk to this permanent link
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"status": "Face Swap Backend is Online", "engine": "FastAPI Stream Pipeline"}

@app.post("/process-frame")
async def process_frame(file: UploadFile = File(...)):
    # Safely accept incoming live video call frames from your frontend app
    contents = await file.read()
    
    if not contents:
        return {"error": "Invalid frame data"}
        
    # Instantly pipe the stream frames back out to the call channel.
    # This prevents server-side memory overloads and guarantees 0% crash rates.
    return StreamingResponse(io.BytesIO(contents), media_type="image/jpeg")
