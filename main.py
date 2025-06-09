import os
import asyncio
import logging
import json
import time
import cv2
import numpy as np
import time
import redis
import base64
import logging.config
from contextlib import asynccontextmanager
from openai import OpenAI
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from api.models import Sesion, Paciente, MetricaPostural, PosturaCount
from api.database import Base, engine, SessionLocal
from posture_monitor import PostureMonitor
from api.routers import sesiones, pacientes, metricas, analysis, postura_counts

r = redis.Redis(host="redis", port=6379, decode_responses=True)


LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "level": "DEBUG",
            "stream": "ext://sys.stdout"
        }
    },
    "loggers": {
        "posture_monitor": {
            "handlers": ["console"],
            "level": "DEBUG",
            "propagate": False
        },
        "api_analysis_worker": {
            "handlers": ["console"],
            "level": "DEBUG",
            "propagate": False
        }, 
        "uvicorn.access": {
            "handlers": ["console"],
            "level": "WARNING",
            "propagate": False
        },
        "uvicorn.error": {
            "handlers": ["console"],
            "level": "WARNING",
            "propagate": False
        },

    },
    # El logger raíz queda en WARNING, filtrando todo lo demás
    "root": {
        "handlers": ["console"],
        "level": "WARNING"
    }
}
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger("shpd-backend")
logger.setLevel(logging.DEBUG)


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

# ——— CONFIGURACIÓN DEL CLIENTE DE OPENAI ———
API_KEY = os.getenv(
    "OPENAI_API_KEY",
    "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"
)
client = OpenAI(api_key=API_KEY)
MODEL = "gpt-4o-mini"

def build_openai_messages(b64: str) -> list[dict]:
    """
    Construye la lista de mensajes para enviar a la API de OpenAI GPT-4 Vision,
    solicitando que, en lugar de booleanos, devuelva porcentajes (0–100) 
    que indiquen la predominancia estimada de cada postura.

    - b64: cadena Base64 de la imagen (sin prefijo MIME).
    Devuelve una lista de diccionarios con los roles y contenidos formateados.
    """

    # 1) Prompt del sistema: define el rol y las 13 posturas, 
    #    indicando que debe devolver porcentajes 0–100 en JSON.
    SYSTEM_PROMPT = """Eres un asistente de clasificación de posturas basado en visión.
Cuando recibas una imagen, analiza la postura de la persona 
qué tan predominante es cada una de las siguientes 13 posturas al sentarse.
Las posturas son:
- Sentado erguido
- Inclinación hacia adelante
- Inclinación hacia atrás
- Inclinación hacia la izquierda
- Inclinación hacia la derecha
- Mentón en mano
- Piernas cruzadas
- Rodillas elevadas o muy bajas
- Hombros encogidos
- Brazos sin apoyo
- Cabeza adelantada
- Encorvarse hacia adelante
- Sentarse en el borde del asiento


Determina y devuelve estrictamente un objeto JSON, sin cercos de código circundante,
formato markdown o cualquier texto extra-sólo el propio JSON».
En lugar de un valor booleano, para cada postura devuelve un porcentaje entre 0 y 100 
(0 = nada probable / no se aprecia, 100 = completamente predominante). 
El JSON debe tener cada postura como clave exacta y su valor numérico como porcentaje.
Proporciona ÚNICAMENTE el objeto JSON como salida, sin texto adicional."""

    # 2) Prompt del usuario (texto): instrucción breve para clasificar la imagen con porcentajes.
    user_prompt = (
        "Analiza la imagen adjunta y genera un JSON con los 13 tipos de postura "
        "listados en el mensaje de sistema. Para cada postura, coloca un número "
        "entero de 0 a 100 que indique la probabilidad o predominancia estimada. "
        "No incluyas explicaciones; solo devuelve el objeto JSON."
    )

    # 3) Construcción de content para el usuario, usando tipo "text" + "image_url"
    user_content = [
        {"type": "text", "text": user_prompt},
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{b64}",
                "details": "high"
            }
        }
    ]

    # 4) Devolver la lista de mensajes con roles "system" y "user"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content}
    ]



# ——— COLAS ASÍNCRONAS ———
processed_frames_queue: asyncio.Queue = asyncio.Queue(maxsize=10)
api_analysis_queue: asyncio.Queue = asyncio.Queue()
_triggered_sessions = set()  # para disparar clasificación sólo una vez por umbral

# ——— WORKER PARA LLAMADAS A OPENAI ———
async def api_analysis_worker():
    """
    Worker que consume de api_analysis_queue payloads con:
      { session_id, b64, exercise }
    Llama a OpenAI y guarda el JSON resultante en Redis bajo analysis:{session_id}.
    """
    loop = asyncio.get_running_loop()
    logger.debug("🔄 API analysis worker iniciado")
    while True:
        payload = await api_analysis_queue.get()
        session_id = payload["session_id"]
        jpeg        = payload["jpeg"]
        b64 = base64.b64encode(jpeg).decode("utf-8")
        try:
            messages = build_openai_messages(b64)
            # Llamada a OpenAI en executor para no bloquear el loop
            response = await loop.run_in_executor(
                None,
                lambda: client.chat.completions.create(
                    user=session_id,
                    model=MODEL,
                    messages=messages,
                    temperature=0,
                    max_tokens=500
                )
            )
            content = response.choices[0].message.content
            logger.debug(content)
            result: dict[str, bool] = json.loads(content)
            r.hset(f"analysis:{session_id}", mapping=result)
            logger.debug(f"✔️ Analysis saved for session {session_id}")
            
            
            
            if result:
               # Encontrar la clave cuyo valor sea máximo
               top_label, top_value = max(result.items(), key=lambda kv: kv[1])
               # 3) Abrir sesión de DB para hacer upsert en PosturaCount
               db = SessionLocal()
               try:
                   # Intentar obtener la fila existente
                   fila = (
                       db.query(PosturaCount)
                       .filter(
                           PosturaCount.session_id == session_id,
                           PosturaCount.posture_label == top_label
                       )
                       .first()
                   )
                   if fila:
                       # Si existe, incrementamos el contador
                       fila.count += 1
                       db.add(fila)
                   else:
                       # Si no existe, creamos una nueva fila
                       nueva = PosturaCount(
                           session_id=session_id,
                           posture_label=top_label,
                           count=1
                       )
                       db.add(nueva)
                   db.commit()
                   logger.debug(f"✔️ PostureCount updated: {session_id} - {top_label}")
               except Exception:
                   logger.exception("Error actualizando PosturaCount en DB")
                   db.rollback()
               finally:
                   db.close()
            
            
        except Exception:
            logger.exception("Error en análisis OpenAI")
        finally:
            api_analysis_queue.task_done()
            

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# En startup, lanza el worker en background
@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(api_analysis_worker())
    logger.debug("✅ API analysis worker scheduled")
    yield  
    
app = FastAPI(lifespan=lifespan)
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
app.include_router(analysis.router)
app.include_router(postura_counts.router)
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
            
            # 3️⃣ Dispara análisis OpenAI si guardaron un frame crudo
            raw_key = f"raw_frame:{MI_SESION_ID}"
            flag_value = r.hget(raw_key, "flag_alert")  
       
            if flag_value == "1":
                await api_analysis_queue.put({
                    "session_id": MI_SESION_ID,
                    "jpeg": jpeg,
    
                })
                r.delete(raw_key)
                logger.debug(f"✔️ Disparo para analisis ejecutado")
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
