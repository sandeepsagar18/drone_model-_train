import asyncio, json, math, time, cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --- VIDEO UPLINK ---
VIDEO_PATH = "/tmp/demo6_work/videoplayback (3).mp4"
camera = cv2.VideoCapture(VIDEO_PATH)

# Global start time for the Mission Clock
MISSION_START_TIME = time.time()

def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            camera.set(cv2.CAP_PROP_POS_FRAMES, 0) 
            continue
        frame = cv2.resize(frame, (640, 360))
        frame[:, :, 0] = 0 # Green Tint
        frame[:, :, 2] = 0
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.03)

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(gen_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.websocket("/ws/telemetry")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            t = time.time() - MISSION_START_TIME
            roll = 15 * math.sin(t * 0.4)
            data = {
                "speed": round(14 + 2 * math.sin(t * 0.2), 1),
                "altitude": round(26 + 3 * math.cos(t * 0.3), 1),
                "battery": 85,
                "roll": round(roll, 1),
                "pitch": round(6 * math.cos(t * 0.5), 1),
                "ai_perfection": round(96 + 2 * math.sin(t * 0.1), 1),
                "ai_status": "LOCKED" if abs(roll) < 5 else "CORRECTING",
                "flight_time": int(t) # SENDING TOTAL SECONDS
            }
            await websocket.send_json(data)
            await asyncio.sleep(0.05)
    except WebSocketDisconnect: pass

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)