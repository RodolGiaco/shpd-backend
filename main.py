import asyncio
import logging
import cv2
import numpy as np
import time
import redis
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from api.models import Sesion, Paciente, MetricaPostural
from api.database import Base, engine, SessionLocal
from api.models import Sesion
from posture_monitor import PostureMonitor
from api.routers import sesiones, pacientes, metricas

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
r = redis.Redis(host="redis", port=6379, decode_responses=True)
logging.getLogger("posture_monitor").setLevel(logging.DEBUG)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("uvicorn.error").setLevel(logging.WARNING)
Base.metadata.create_all(bind=engine)
db = SessionLocal()
try:
    nueva_sesion = Sesion(intervalo_segundos=600, modo="monitor_activo")
    db.add(nueva_sesion)
    db.commit()
    db.refresh(nueva_sesion)
    MI_SESION_ID = str(nueva_sesion.id)
    
    start_ts = int(time.time())
    r.hset(f"shpd-session:{MI_SESION_ID}", mapping={
        "start_ts": start_ts,
        "intervalo_segundos": nueva_sesion.intervalo_segundos,
    })
finally:
    db.close()


app = FastAPI()
posture_monitor = PostureMonitor(MI_SESION_ID)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(sesiones.router)
app.include_router(pacientes.router)
app.include_router(metricas.router)

processed_frames_queue = asyncio.Queue(maxsize=10)

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
            jpeg = await processed_frames_queue.get()
            await websocket.send_bytes(jpeg)
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
        app,
        host="0.0.0.0",
        port=8765,
        log_level="warning",
        access_log=True
    )
