import asyncio
import logging
import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from api.routers import pacientes
from api.database import Base, engine
from posture_monitor import PostureMonitor

# ——— Desactivar logs no críticos ———
logging.getLogger("uvicorn.error").setLevel(logging.CRITICAL)
logging.getLogger("uvicorn.access").disabled = True

app = FastAPI()
posture_monitor = PostureMonitor()
Base.metadata.create_all(bind=engine)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(pacientes.router)

processed_frames_queue = asyncio.Queue(maxsize=1)  # Solo el último frame

@app.websocket("/video/input")
async def video_input(websocket: WebSocket):
    await websocket.accept()
    loop = asyncio.get_running_loop()
    try:
        while True:
            data = await websocket.receive_bytes()
            frame = await loop.run_in_executor(None, _decode_jpeg, data)
            if frame is None:
                continue
            processed = await loop.run_in_executor(None, posture_monitor.process_frame, frame)
            jpeg = await loop.run_in_executor(None, _encode_jpeg, processed)
            if processed_frames_queue.full():
                processed_frames_queue.get_nowait()
            await processed_frames_queue.put(jpeg)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass

@app.websocket("/video/output")
async def video_output(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            jpeg = processed_frames_queue.get_nowait()
            asyncio.create_task(websocket.send_bytes(jpeg))
    except WebSocketDisconnect:
        pass
    except Exception:
        pass

def _decode_jpeg(data: bytes):
    arr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def _encode_jpeg(frame):
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
    return buf.tobytes()

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8765,
        log_level="error",
        access_log=False
    )
