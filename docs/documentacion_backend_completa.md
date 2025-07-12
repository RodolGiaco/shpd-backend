# Documentación Técnica del Backend - Sistema de Detección de Posturas

## Índice de Contenidos

1. [Introducción](#introducción)
2. [Arquitectura General del Sistema](#arquitectura-general-del-sistema)
3. [Tecnologías y Dependencias](#tecnologías-y-dependencias)
4. [Estructura del Proyecto](#estructura-del-proyecto)
5. [Módulo Principal (main.py)](#módulo-principal-mainpy)
6. [Monitor de Posturas (posture_monitor.py)](#monitor-de-posturas-posture_monitorpy)
7. [Capa de Datos](#capa-de-datos)
8. [API REST y Endpoints](#api-rest-y-endpoints)
9. [Sistema de Comunicación en Tiempo Real](#sistema-de-comunicación-en-tiempo-real)
10. [Integración con OpenAI](#integración-con-openai)
11. [Sistema de Caché con Redis](#sistema-de-caché-con-redis)
12. [Despliegue y Containerización](#despliegue-y-containerización)
13. [Flujo de Datos y Casos de Uso](#flujo-de-datos-y-casos-de-uso)
14. [Análisis de Fortalezas y Oportunidades de Mejora](#análisis-de-fortalezas-y-oportunidades-de-mejora)
15. [Conclusiones](#conclusiones)

---

## Introducción

El presente documento describe la arquitectura e implementación del backend de un sistema de detección de posturas corporales en tiempo real. Este sistema forma parte de un proyecto de ingeniería orientado a mejorar la salud postural mediante el análisis automático de imágenes y la generación de alertas preventivas.

El backend está diseñado siguiendo principios de arquitectura de microservicios, implementando una API REST con FastAPI, procesamiento de video en tiempo real mediante WebSockets, análisis de posturas con MediaPipe, y clasificación avanzada mediante la API de OpenAI. La arquitectura permite la escalabilidad horizontal y el procesamiento concurrente de múltiples sesiones de usuarios.

### Objetivos del Sistema

1. **Procesamiento en tiempo real**: Analizar streams de video para detectar posturas corporales inadecuadas.
2. **Alertas inteligentes**: Generar notificaciones cuando se detectan posturas incorrectas sostenidas.
3. **Análisis mediante IA**: Utilizar modelos de visión por computadora para clasificación detallada de posturas.
4. **Persistencia de datos**: Almacenar métricas y estadísticas para análisis posterior.
5. **Integración multiplataforma**: Soportar clientes web y bot de Telegram.

---

## Arquitectura General del Sistema

El backend implementa una arquitectura orientada a eventos con los siguientes componentes principales:

### Componentes Principales

1. **Servidor FastAPI**: Núcleo del sistema que expone la API REST y maneja conexiones WebSocket.
2. **Motor de Procesamiento de Video**: Módulo especializado en análisis de posturas usando MediaPipe.
3. **Sistema de Colas Asíncronas**: Gestión de tareas de procesamiento mediante colas en memoria.
4. **Base de Datos PostgreSQL**: Almacenamiento persistente de sesiones, pacientes y métricas.
5. **Cache Redis**: Almacenamiento temporal de datos de sesión y métricas en tiempo real.
6. **Worker de Análisis IA**: Proceso asíncrono para clasificación de posturas con OpenAI.

### Diagrama de Arquitectura

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Cliente Web   │     │  Bot Telegram    │     │ Otros Clientes  │
└────────┬────────┘     └────────┬─────────┘     └────────┬────────┘
         │                       │                          │
         └───────────────────────┴──────────────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │    FastAPI Backend      │
                    │  ┌─────────────────┐    │
                    │  │  WebSocket API  │    │
                    │  └─────────────────┘    │
                    │  ┌─────────────────┐    │
                    │  │    REST API     │    │
                    │  └─────────────────┘    │
                    └────────────┬────────────┘
                                 │
        ┌────────────────────────┼────────────────────────┐
        │                        │                        │
┌───────▼────────┐      ┌────────▼────────┐     ┌────────▼────────┐
│ PostureMonitor │      │  Async Workers  │     │   API Routers   │
│   (MediaPipe)  │      │  (OpenAI API)   │     │   (Endpoints)   │
└────────────────┘      └─────────────────┘     └─────────────────┘
        │                        │                        │
        └────────────────────────┼────────────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │    Data Layer          │
                    │  ┌─────────────────┐   │
                    │  │   PostgreSQL    │   │
                    │  └─────────────────┘   │
                    │  ┌─────────────────┐   │
                    │  │     Redis       │   │
                    │  └─────────────────┘   │
                    └─────────────────────────┘
```

---

## Tecnologías y Dependencias

El proyecto utiliza un stack tecnológico moderno optimizado para procesamiento en tiempo real:

### Dependencias Principales

```python
# requirements.txt
opencv-python-headless    # Procesamiento de imágenes sin GUI
websockets               # Comunicación bidireccional en tiempo real
mediapipe               # Detección de puntos clave corporales
numpy                   # Operaciones numéricas y matrices
fastapi                 # Framework web asíncrono de alto rendimiento
uvicorn                 # Servidor ASGI para FastAPI
sqlalchemy              # ORM para base de datos
psycopg2-binary        # Driver PostgreSQL
redis                   # Sistema de caché en memoria
openai>=0.27.0         # API de OpenAI para análisis con IA
requests               # Cliente HTTP para integraciones
```

### Justificación Tecnológica

1. **FastAPI**: Elegido por su rendimiento superior, soporte nativo de async/await y generación automática de documentación OpenAPI.

2. **MediaPipe**: Solución de Google para detección de poses humanas con alta precisión y rendimiento optimizado.

3. **PostgreSQL + SQLAlchemy**: Combinación robusta para persistencia de datos con soporte transaccional completo.

4. **Redis**: Cache de alta velocidad para datos temporales y comunicación entre procesos.

5. **OpenAI GPT-4 Vision**: Modelo de IA avanzado para clasificación detallada de posturas a partir de imágenes.

---

## Estructura del Proyecto

La organización del código sigue una estructura modular clara:

```
/workspace/
├── main.py                    # Punto de entrada principal
├── posture_monitor.py         # Motor de análisis de posturas
├── requirements.txt           # Dependencias del proyecto
├── Dockerfile                 # Configuración de contenedor
├── api/                       # Módulo de API
│   ├── __init__.py
│   ├── database.py           # Configuración de base de datos
│   ├── models.py             # Modelos SQLAlchemy
│   ├── schemas.py            # Esquemas Pydantic
│   └── routers/              # Endpoints organizados por dominio
│       ├── analysis.py       # Endpoints de análisis
│       ├── calibracion.py    # Endpoints de calibración
│       ├── metricas.py       # Endpoints de métricas
│       ├── pacientes.py      # Endpoints de pacientes
│       ├── postura_counts.py # Endpoints de conteo de posturas
│       ├── sesiones.py       # Endpoints de sesiones
│       └── timeline.py       # Endpoints de línea temporal
└── deploy/                    # Configuraciones de despliegue
    ├── backend-svc.yaml      # Servicio Kubernetes backend
    ├── database-svc.yaml     # Servicio Kubernetes DB
    ├── redis-deploy.yaml     # Despliegue Redis
    └── shpd-backend.yaml     # Despliegue principal
```

---

## Módulo Principal (main.py)

El archivo `main.py` constituye el punto de entrada del backend y orquesta todos los componentes del sistema. A continuación se detalla su implementación:

### Importaciones y Configuración Inicial

```python
import os
import asyncio
import logging
import json
import time
import cv2
import numpy as np
import redis
import base64
import logging.config
from contextlib import asynccontextmanager
from openai import OpenAI
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
```

Las importaciones se organizan en:
- **Librerías estándar**: `os`, `asyncio`, `logging`, `json`, `time`
- **Procesamiento de imágenes**: `cv2` (OpenCV), `numpy`
- **Servicios externos**: `redis`, `openai`
- **Framework web**: `fastapi`, `uvicorn`
- **Módulos locales**: Modelos, base de datos, monitores y routers

### Configuración de Logging

El sistema implementa una configuración detallada de logging para facilitar el debugging y monitoreo:

```python
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
        "posture_monitor": {"handlers": ["console"], "level": "DEBUG"},
        "api_analysis_worker": {"handlers": ["console"], "level": "DEBUG"},
        "uvicorn.access": {"handlers": ["console"], "level": "WARNING"},
        "uvicorn.error": {"handlers": ["console"], "level": "WARNING"}
    },
    "root": {"handlers": ["console"], "level": "WARNING"}
}
```

Esta configuración:
- Define un formato consistente para todos los logs
- Separa niveles de logging por módulo
- Filtra logs verbosos de uvicorn para reducir ruido

### Cliente de OpenAI

La integración con OpenAI se configura mediante variables de entorno:

```python
API_KEY = os.getenv("OPENAI_API_KEY", "sk-proj-...")
client = OpenAI(api_key=API_KEY)
MODEL = "gpt-4o-mini"
```

### Función de Construcción de Mensajes para OpenAI

Esta función es crítica para el análisis de posturas mediante IA:

```python
def build_openai_messages(b64: str) -> list[dict]:
    """
    Construye la lista de mensajes para enviar a la API de OpenAI GPT-4 Vision,
    solicitando porcentajes (0-100) que indiquen la predominancia de cada postura.
    """
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
    """
```

La función:
1. Define un prompt de sistema detallado con las 13 posturas a analizar
2. Construye el mensaje del usuario incluyendo la imagen en base64
3. Retorna la estructura de mensajes esperada por la API de OpenAI

### Sistema de Colas Asíncronas

El backend utiliza colas asíncronas para gestionar el flujo de procesamiento:

```python
processed_frames_queue: asyncio.Queue = asyncio.Queue(maxsize=10)
api_analysis_queue: asyncio.Queue = asyncio.Queue()
_triggered_sessions = set()  # Control de disparos únicos por sesión
```

- **processed_frames_queue**: Almacena frames procesados para transmisión
- **api_analysis_queue**: Cola de trabajos pendientes de análisis con IA
- **_triggered_sessions**: Evita análisis duplicados por sesión

### Worker de Análisis con IA

El worker asíncrono consume tareas de análisis:

```python
async def api_analysis_worker():
    """
    Worker que consume de api_analysis_queue payloads con:
      { session_id, b64, exercise }
    Llama a OpenAI y guarda el JSON resultante en Redis.
    """
    loop = asyncio.get_running_loop()
    logger.debug("🔄 API analysis worker iniciado")
    
    while True:
        payload = await api_analysis_queue.get()
        session_id = payload["session_id"]
        jpeg = payload["jpeg"]
        bad_time = payload["bad_time"]
        b64 = base64.b64encode(jpeg).decode("utf-8")
        
        try:
            # Llamada a OpenAI en executor para no bloquear
            messages = build_openai_messages(b64)
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
```

Este worker:
1. Opera en un bucle infinito consumiendo tareas
2. Codifica imágenes en base64
3. Ejecuta llamadas a OpenAI en un thread pool para no bloquear
4. Almacena resultados en Redis
5. Actualiza contadores de posturas en la base de datos
6. Registra eventos en la línea temporal

### Gestión del Ciclo de Vida de la Aplicación

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(api_analysis_worker())
    logger.debug("✅ API analysis worker scheduled")
    yield
```

El gestor de contexto asíncrono:
- Inicia el worker de análisis al arrancar la aplicación
- Garantiza limpieza ordenada al cerrar

### Configuración de FastAPI y CORS

```python
app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

La configuración CORS permite:
- Acceso desde cualquier origen (desarrollo)
- Todos los métodos HTTP
- Todas las cabeceras
- Credenciales en peticiones

### Inclusión de Routers

```python
app.include_router(sesiones.router)
app.include_router(pacientes.router)
app.include_router(metricas.router)
app.include_router(analysis.router)
app.include_router(postura_counts.router)
app.include_router(timeline.router)
app.include_router(calibracion.router)
```

Los routers organizan los endpoints por dominio funcional.

### WebSocket de Entrada de Video

El endpoint más complejo del sistema maneja el stream de video:

```python
@app.websocket("/video/input/{device_id}")
async def video_input(websocket: WebSocket, device_id: str):
    await websocket.accept()
    loop = asyncio.get_running_loop()
    
    # Detectar modo calibración
    calibrating_query = websocket.scope.get("query_string", b"").decode().find("calibracion=1") >= 0
    
    # Variables de estado
    posture_monitor = None
    current_session_id = None
    
    try:
        while True:
            # 1. Verificar session_id desde Redis
            redis_shpd_key = f"shpd-data:{device_id}"
            session_id = r.hget(redis_shpd_key, "session_id")
            
            # 2. Determinar modo (calibración o normal)
            mode = r.hget(redis_shpd_key, "mode")
            calibrating = (mode != "normal") if mode else calibrating_query
            
            # 3. Reinicializar PostureMonitor si cambió la sesión
            if session_id != current_session_id:
                posture_monitor = PostureMonitor(session_id, save_metrics=not calibrating)
                current_session_id = session_id
            
            # 4. Procesar frame
            data = await websocket.receive_bytes()
            frame = await loop.run_in_executor(None, _decode_jpeg, data)
            
            if posture_monitor:
                processed = await loop.run_in_executor(None, posture_monitor.process_frame, frame)
                jpeg = await loop.run_in_executor(None, _encode_jpeg, processed)
            
            # 5. Encolar frame procesado
            if processed_frames_queue.full():
                processed_frames_queue.get_nowait()
            await processed_frames_queue.put(jpeg)
            
            # 6. Disparar análisis si hay alerta
            if posture_monitor and not calibrating:
                raw_key = f"raw_frame:{session_id}"
                if r.hget(raw_key, "flag_alert") == "1":
                    await api_analysis_queue.put({
                        "session_id": session_id,
                        "jpeg": jpeg,
                        "bad_time": r.hget(raw_key, "bad_time")
                    })
                    r.delete(raw_key)
```

Este endpoint:
1. Acepta conexiones WebSocket por dispositivo
2. Detecta y gestiona modos de operación (calibración/normal)
3. Mantiene un monitor de posturas por sesión
4. Procesa frames de manera asíncrona
5. Gestiona la cola de frames procesados
6. Dispara análisis con IA cuando se detectan alertas

### WebSocket de Salida de Video

```python
@app.websocket("/video/output")
async def video_output(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            jpeg = await processed_frames_queue.get()
            await websocket.send_bytes(jpeg)
    except WebSocketDisconnect:
        pass
```

Este endpoint simple:
- Consume frames de la cola procesada
- Los transmite a clientes conectados
- Maneja desconexiones gracefully

### Funciones Auxiliares de Codificación

```python
def _decode_jpeg(data: bytes):
    arr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def _encode_jpeg(frame):
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
    return buf.tobytes()
```

Estas funciones:
- Convierten entre bytes JPEG y arrays NumPy
- Aplican compresión JPEG con calidad 50 para optimizar ancho de banda

### Inicialización y Arranque

```python
Base.metadata.create_all(bind=engine)

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8765,
        log_level="warning",
        access_log=True
    )
```

El servidor:
- Crea las tablas de base de datos si no existen
- Escucha en todas las interfaces en el puerto 8765
- Configura logging moderado para producción

---

## Monitor de Posturas (posture_monitor.py)

El módulo `posture_monitor.py` implementa el motor de análisis de posturas utilizando MediaPipe. Este componente es fundamental para la detección en tiempo real de posturas corporales incorrectas.

### Arquitectura del Monitor

```python
import cv2
import time
import math as m
import mediapipe as mp
import argparse
import json
import os
from datetime import datetime
from api.database import SessionLocal
from api.models import MetricaPostural
import logging
import redis
```

### Clase PostureMonitor

La clase principal encapsula toda la lógica de detección:

```python
class PostureMonitor:
    def __init__(self, session_id: str, *, save_metrics: bool = True):
        logger.info(f"[PostureMonitor] Instanciado para session_id={session_id} save_metrics={save_metrics}")
        self.mp_drawing = mp.solutions.drawing_utils
        self.session_id = session_id
        self.save_metrics = save_metrics
```

#### Parámetros de Inicialización

- **session_id**: Identificador único de la sesión de monitoreo
- **save_metrics**: Booleano que indica si se deben persistir las métricas (False en modo calibración)

#### Configuración de MediaPipe

```python
self.mp_pose = mp.solutions.pose
self.pose = self.mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
```

Los parámetros de MediaPipe están optimizados para:
- **static_image_mode=False**: Procesamiento de video continuo
- **min_detection_confidence=0.5**: Balance entre precisión y velocidad
- **min_tracking_confidence=0.5**: Seguimiento estable de puntos clave

### Sistema de Umbrales Configurables

El monitor carga umbrales desde un archivo JSON o usa valores por defecto:

```python
if os.path.exists("calibration.json"):
    with open("calibration.json", "r") as f:
        data = json.load(f)
        self.args.offset_threshold = data.get("offset_threshold", self.args.offset_threshold)
        self.args.neck_angle_threshold = data.get("neck_angle_threshold", self.args.neck_angle_threshold)
        self.args.torso_angle_threshold = data.get("torso_angle_threshold", self.args.torso_angle_threshold)
        self.args.time_threshold = data.get("time_threshold", self.args.time_threshold)
```

Los umbrales controlan:
- **offset_threshold**: Desalineación de hombros (píxeles)
- **neck_angle_threshold**: Ángulo máximo de inclinación del cuello (grados)
- **torso_angle_threshold**: Ángulo máximo de inclinación del torso (grados)
- **time_threshold**: Tiempo antes de generar alerta (segundos)

### Funciones Matemáticas de Cálculo

#### Cálculo de Distancia Euclidiana

```python
def findDistance(self, x1, y1, x2, y2):
    dist = m.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return dist
```

#### Cálculo de Ángulo de Inclinación

```python
def findAngle(self, x1, y1, x2, y2):
    theta = m.acos((y2 - y1) * (-y1) / (m.sqrt((x2 - x1)**2 + (y2 - y1)**2) * y1))
    degree = int(180/m.pi) * theta
    return degree
```

Esta función calcula el ángulo entre:
- Un vector formado por dos puntos corporales
- El eje vertical

La fórmula utiliza el producto escalar normalizado para obtener el coseno del ángulo.

### Procesamiento de Frames

El método principal `process_frame` ejecuta el análisis:

```python
def process_frame(self, image):
    h, w = image.shape[:2]
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    keypoints = self.pose.process(image_rgb)
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
```

#### Extracción de Puntos Clave

```python
lm = keypoints.pose_landmarks
lmPose = self.mp_pose.PoseLandmark

# Detección de ausencia de persona
if lm is None:
    delta = 1.0 / fps
    r.hincrbyfloat(buffer_key, "tiempo_parado", round(delta,1))
    return image

# Puntos clave del lado derecho
r_shldr_x = int(lm.landmark[lmPose.RIGHT_SHOULDER].x * w)
r_shldr_y = int(lm.landmark[lmPose.RIGHT_SHOULDER].y * h)
r_ear_x = int(lm.landmark[lmPose.RIGHT_EAR].x * w)
r_ear_y = int(lm.landmark[lmPose.RIGHT_EAR].y * h)
r_hip_x = int(lm.landmark[lmPose.RIGHT_HIP].x * w)
r_hip_y = int(lm.landmark[lmPose.RIGHT_HIP].y * h)
```

El sistema:
1. Convierte coordenadas normalizadas a píxeles
2. Extrae puntos del hombro, oreja y cadera derechos
3. Estos puntos definen la postura del usuario

#### Cálculo de Ángulos Posturales

```python
neck_inclination = self.findAngle(r_shldr_x, r_shldr_y, r_ear_x, r_ear_y)
torso_inclination = self.findAngle(r_hip_x, r_hip_y, r_shldr_x, r_shldr_y)
```

- **neck_inclination**: Ángulo entre hombro-oreja y vertical (inclinación de cabeza)
- **torso_inclination**: Ángulo entre cadera-hombro y vertical (inclinación de espalda)

### Evaluación de Postura

```python
if neck_inclination < self.args.neck_angle_threshold and torso_inclination < self.args.torso_angle_threshold:
    # Postura correcta
    self.bad_frames = 0
    self.good_frames += 1
    color = (127, 233, 100)  # Verde
    r.hincrby(buffer_key, "good_frames", 1)
    self.flag_transition = True
    self.flag_alert = True
else:
    # Postura incorrecta
    self.good_frames = 0
    self.bad_frames += 1
    color = (50, 50, 255)   # Rojo
    r.hincrby(buffer_key, "bad_frames", 1)
    
    if self.flag_transition:
        r.hincrby(buffer_key, "transiciones_malas", 1)
        self.flag_transition = False
```

El sistema:
1. Compara ángulos con umbrales configurados
2. Mantiene contadores separados para frames buenos/malos
3. Detecta transiciones entre posturas
4. Actualiza estadísticas en Redis en tiempo real

### Visualización y Feedback

```python
# Dibujar texto informativo
cv2.putText(image, angle_text_string_neck, (10, 30), self.font, 0.6, color, 2)
cv2.putText(image, angle_text_string_torso, (10, 60), self.font, 0.6, color, 2)

# Dibujar puntos clave
cv2.circle(image, (r_shldr_x, r_shldr_y), 7, (255, 255, 255), 2)
cv2.circle(image, (r_ear_x, r_ear_y), 7, (255, 255, 255), 2)
cv2.circle(image, (r_hip_x, r_hip_y), 7, (0, 255, 255), -1)

# Dibujar líneas de referencia
cv2.line(image, (r_shldr_x, r_shldr_y), (r_ear_x, r_ear_y), color, 2)
cv2.line(image, (r_hip_x, r_hip_y), (r_shldr_x, r_shldr_y), color, 2)
```

La visualización incluye:
- Valores numéricos de ángulos
- Puntos corporales detectados
- Líneas que muestran la inclinación
- Código de colores según estado postural

### Sistema de Alertas

```python
if self.save_metrics:
    if bad_time > self.args.time_threshold:
        if self.flag_alert:
            self.sendWarning()
            raw_key = f"raw_frame:{self.session_id}"
            r.hincrby(buffer_key, "alert_count", 1)
            r.hset(raw_key, "flag_alert", "1")
            r.hset(raw_key, "bad_time", round(bad_time, 1))
            self.flag_alert = False
            self.bad_frames = 0  # Reiniciar contador
```

El sistema de alertas:
1. Se activa cuando el tiempo en mala postura excede el umbral
2. Marca el frame para análisis con IA
3. Evita alertas repetidas con flag_alert
4. Reinicia el contador para la siguiente detección

### Recolección de Métricas

```python
accum = r.hgetall(buffer_key)
good_acc = int(accum.get("good_frames", 0))
bad_acc = int(accum.get("bad_frames", 0))
alert_count = int(accum.get("alert_count", 0))
transiciones_malas = int(accum.get("transiciones_malas", 0))
tiempo_sentado = float(accum.get("tiempo_sentado", 0))
tiempo_parado = float(accum.get("tiempo_parado", 0))

total = good_acc + bad_acc or 1
percentage_good = round(good_acc / total * 100, 1)
percentage_bad = round(bad_acc / total * 100, 1)

datos = {
    "actual": "menton_en_mano",
    "porcentaje_correcta": percentage_good,
    "porcentaje_incorrecta": percentage_bad,
    "transiciones_malas": transiciones_malas,
    "tiempo_sentado": tiempo_sentado,
    "tiempo_parado": tiempo_parado,
    "alertas_enviadas": alert_count,
}
```

Las métricas incluyen:
- Porcentajes de tiempo en buena/mala postura
- Número de transiciones posturales
- Tiempo total sentado vs parado
- Contador de alertas generadas

### Modo Calibración

```python
if not self.save_metrics:
    calib_key = f"calib:{self.session_id}"
    if good_time > 0:
        r.hincrbyfloat(calib_key, "good_time", round(1.0 / fps, 2))
    if bad_time > 0:
        r.hincrbyfloat(calib_key, "bad_time", round(1.0 / fps, 2))
```

En modo calibración:
- No se guardan métricas permanentes
- Se acumulan tiempos buenos/malos en Redis temporal
- Permite al usuario ajustar umbrales según su ergonomía

### Método de Ejecución Standalone

```python
def run(self):
    cap = cv2.VideoCapture(self.args.video) if self.args.video else cv2.VideoCapture(0)
    
    while True:
        success, image = cap.read()
        if not success:
            break
            
        image = self.process_frame(image)
        cv2.imshow('MediaPipe Pose', image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
```

Este método permite:
- Ejecutar el monitor independientemente para pruebas
- Procesar archivos de video o cámara web
- Visualización directa con OpenCV

---

## Capa de Datos

La capa de datos del sistema implementa un diseño robusto utilizando SQLAlchemy como ORM y PostgreSQL como motor de base de datos. La arquitectura separa claramente las responsabilidades entre configuración de conexión, modelos de datos y esquemas de validación.

### Configuración de Base de Datos (database.py)

El módulo `database.py` establece la configuración fundamental para la conexión y gestión de sesiones:

```python
import os
from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base, Session

# URL de conexión configurable por variable de entorno
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://user:password@postgres-service:5432/shpd_db"
)
```

#### Motor de SQLAlchemy

```python
engine = create_engine(
    DATABASE_URL,
    echo=False,    # True para debugging SQL
    future=True    # API 2.0 de SQLAlchemy
)
```

Configuración del motor:
- **echo=False**: Desactiva el logging SQL en producción
- **future=True**: Habilita la API moderna de SQLAlchemy 2.0

#### Fábrica de Sesiones

```python
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)
```

Parámetros de sesión:
- **autocommit=False**: Control manual de transacciones
- **autoflush=False**: Evita escrituras automáticas a la BD
- **bind=engine**: Asocia las sesiones al motor configurado

#### Dependencia de FastAPI

```python
def get_db() -> Generator[Session, None, None]:
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

Esta función generadora:
1. Crea una nueva sesión para cada request
2. La proporciona mediante inyección de dependencias
3. Garantiza el cierre de la sesión al finalizar

### Modelos de Datos (models.py)

Los modelos definen la estructura de las tablas en la base de datos:

#### Modelo Paciente

```python
class Paciente(Base):
    __tablename__ = "pacientes"

    id = Column(Integer, primary_key=True, index=True)
    telegram_id = Column(String, unique=True, index=True, nullable=False)
    device_id = Column(String, unique=True, index=True, nullable=False)
    nombre = Column(String, nullable=False)
    edad = Column(Integer)
    sexo = Column(String)
    diagnostico = Column(String)
```

Características del modelo:
- **id**: Clave primaria autoincremental
- **telegram_id**: Identificador único para integración con bot
- **device_id**: Identificador del dispositivo de monitoreo
- **Índices únicos**: En telegram_id y device_id para búsquedas rápidas

#### Modelo MetricaPostural

```python
class MetricaPostural(Base):
    __tablename__ = "metricas_posturales"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    sesion_id = Column(UUID(as_uuid=True), ForeignKey("sesiones.id", ondelete="CASCADE"), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    datos = Column(JSONB, nullable=False)
    created_at = Column(DateTime, server_default=func.now())
```

Diseño del modelo:
- **UUID como PK**: Identificadores únicos distribuidos
- **JSONB**: Almacenamiento flexible de métricas complejas
- **CASCADE**: Eliminación automática al borrar sesión padre
- **server_default**: Timestamp automático del servidor

#### Modelo Sesion

```python
class Sesion(Base):
    __tablename__ = "sesiones"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    intervalo_segundos = Column(Integer, nullable=False)
    modo = Column(String, nullable=False)
```

Atributos:
- **intervalo_segundos**: Duración planificada de la sesión
- **modo**: Tipo de sesión (normal, calibración, etc.)

#### Modelo PosturaCount

```python
class PosturaCount(Base):
    __tablename__ = "postura_counts"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True, nullable=False)
    posture_label = Column(String, nullable=False)
    count = Column(Integer, default=0, nullable=False)
```

Este modelo:
- Mantiene contadores de cada tipo de postura por sesión
- Optimizado para consultas agregadas
- Índice en session_id para agrupaciones rápidas

### Esquemas de Validación (schemas.py)

Los esquemas Pydantic validan y serializan datos entre la API y la base de datos:

#### Esquemas de Métricas

```python
class MetricaIn(BaseModel):
    sesion_id: UUID
    timestamp: datetime
    datos: dict

class MetricaOut(MetricaIn):
    id: UUID
    created_at: datetime
    
    class Config:
        orm_mode = True
```

Características:
- **MetricaIn**: Datos de entrada sin campos autogenerados
- **MetricaOut**: Incluye campos generados por la BD
- **orm_mode**: Permite crear instancias desde objetos SQLAlchemy

#### Esquemas de Paciente

```python
class PacienteOut(BaseModel):
    id: int
    telegram_id: str
    device_id: str
    nombre: str
    edad: int
    sexo: str | None = None
    diagnostico: str | None = None

    class Config:
        orm_mode = True
```

Validaciones:
- Campos opcionales con valores por defecto None
- Tipado estricto para cada campo
- Serialización automática de modelos ORM

#### Esquemas de Sesión

```python
class SesionIn(BaseModel):
    intervalo_segundos: int
    modo: str

class SesionOut(SesionIn):
    id: UUID
    
    class Config:
        orm_mode = True
```

Patrón de herencia:
- **SesionIn**: Solo campos editables por el usuario
- **SesionOut**: Hereda de SesionIn y agrega campos generados

#### Esquemas de Conteo de Posturas

```python
class PosturaCountOut(BaseModel):
    session_id: str
    posture_label: str
    count: int
    
    class Config:
        orm_mode = True
```

### Diseño de la Base de Datos

#### Diagrama Entidad-Relación

```
┌─────────────────┐     ┌──────────────────┐
│    Paciente     │     │     Sesion       │
├─────────────────┤     ├──────────────────┤
│ id (PK)         │     │ id (PK, UUID)    │
│ telegram_id     │     │ intervalo_seg    │
│ device_id       │     │ modo             │
│ nombre          │     └──────────────────┘
│ edad            │              │
│ sexo            │              │ 1
│ diagnostico     │              │
└─────────────────┘              │ *
                        ┌────────▼─────────┐
                        │ MetricaPostural  │
                        ├──────────────────┤
                        │ id (PK, UUID)    │
                        │ sesion_id (FK)   │
                        │ timestamp        │
                        │ datos (JSONB)    │
                        │ created_at       │
                        └──────────────────┘
                                 
                        ┌──────────────────┐
                        │  PosturaCount    │
                        ├──────────────────┤
                        │ id (PK)          │
                        │ session_id       │
                        │ posture_label    │
                        │ count            │
                        └──────────────────┘
```

### Ventajas del Diseño

1. **Flexibilidad**: JSONB permite evolución del esquema sin migraciones
2. **Rendimiento**: Índices estratégicos en campos de búsqueda frecuente
3. **Integridad**: Claves foráneas con acciones en cascada
4. **Escalabilidad**: UUIDs permiten sistemas distribuidos
5. **Mantenibilidad**: Separación clara entre modelos y validación

---

## API REST y Endpoints

El backend expone una API REST completa organizada en módulos temáticos. Cada router maneja un dominio específico del sistema, facilitando el mantenimiento y la escalabilidad.

### Router de Sesiones (sesiones.py)

El router de sesiones gestiona el ciclo de vida completo de las sesiones de monitoreo:

#### Crear Sesión

```python
@router.post("/", response_model=SesionOut)
def crear_sesion(s: SesionIn, db: Session = Depends(get_db)) -> SesionOut:
    nueva = Sesion(**s.dict())
    db.add(nueva)
    db.commit()
    db.refresh(nueva)
    # Borrar marca de sesión finalizada si existe
    if hasattr(s, 'device_id'):
        r.delete(f"ended:{s.device_id}")
    return nueva
```

**Endpoint**: `POST /sesiones/`
- **Entrada**: `SesionIn` (intervalo_segundos, modo)
- **Salida**: `SesionOut` (incluye UUID generado)
- **Función**: Crea nueva sesión y limpia marcas previas

#### Listar Sesiones

```python
@router.get("/", response_model=List[SesionOut])
def listar_sesiones(db: Session = Depends(get_db)) -> List[SesionOut]:
    return db.query(Sesion).all()
```

**Endpoint**: `GET /sesiones/`
- **Salida**: Lista de todas las sesiones
- **Uso**: Histórico y administración

#### Progreso de Sesión

```python
@router.get("/progress/{session_id}")
def get_session_progress(session_id: str):
    key = f"shpd-session:{session_id}"
    data = r.hgetall(key)
    start_ts = int(data.get("start_ts", 0))
    intervalo = int(data.get("intervalo_segundos", 0))
    now = int(time.time())
    elapsed = now - start_ts
    if elapsed > intervalo:
        elapsed = intervalo
    return {"intervalo_segundos": intervalo, "elapsed": elapsed}
```

**Endpoint**: `GET /sesiones/progress/{session_id}`
- **Parámetros**: session_id (UUID)
- **Salida**: Tiempo transcurrido y duración total
- **Función**: Monitoreo en tiempo real del progreso

#### Finalizar Sesión

```python
@router.post("/end/{device_id}")
def finalizar_sesion(device_id: str, db: Session = Depends(get_db)):
    shpd_data = r.hgetall(f"shpd-data:{device_id}")
    session_id = shpd_data.get("session_id")
    
    if not session_id:
        return {"ok": False, "message": "No se encontró session_id"}
    
    # Verificar si ya fue finalizada
    ended_key = f"ended:{session_id}"
    if r.get(ended_key):
        return {"ok": False, "message": "Sesión ya finalizada"}
    
    # Enviar reporte y limpiar datos
    enviar_reporte_telegram(session_id, device_id, db)
    r.setex(ended_key, 3600, "1")  # Marca por 1 hora
    r.hdel(f"shpd-data:{device_id}", "session_id")
    
    return {"ok": True, "message": "Sesión finalizada"}
```

**Endpoint**: `POST /sesiones/end/{device_id}`
- **Función**: Finaliza sesión, genera reporte y limpia datos
- **Integración**: Envía resumen por Telegram

#### Reiniciar Sesión

```python
@router.post("/reiniciar/{session_id}")
def reiniciar_sesion(session_id: str, device_id: str | None = Query(None), db: Session = Depends(get_db)):
    # Validar UUID
    try:
        uuid_obj = uuid.UUID(session_id)
    except ValueError:
        return JSONResponse(status_code=400, content={"ok": False, "message": "session_id inválido"})
    
    # Limpiar todos los datos asociados
    r.delete(
        f"shpd-data:{session_id}",
        f"metricas:{session_id}",
        f"analysis:{session_id}",
        f"raw_frame:{session_id}",
        f"ended:{session_id}",
        f"timeline:{session_id}"
    )
    
    # Limpiar base de datos
    db.query(PosturaCount).filter(PosturaCount.session_id == str(uuid_obj)).delete()
    db.query(MetricaPostural).filter(MetricaPostural.sesion_id == uuid_obj).delete()
    db.commit()
    
    return {"ok": True, "message": "Sesión reiniciada"}
```

**Endpoint**: `POST /sesiones/reiniciar/{session_id}`
- **Función**: Limpia todos los datos para comenzar de nuevo
- **Uso**: Recuperación de errores o nueva calibración

### Router de Pacientes (pacientes.py)

Gestiona la información de los pacientes monitoreados:

```python
@router.get("/{device_id}", response_model=PacienteOut)
def obtener_paciente_por_device_id(device_id: str, db: Session = Depends(get_db)):
    paciente = db.query(Paciente).filter(Paciente.device_id == device_id).first()
    if not paciente:
        raise HTTPException(status_code=404, detail="Paciente no encontrado")
    return paciente
```

**Endpoint**: `GET /pacientes/{device_id}`
- **Función**: Obtiene datos del paciente por dispositivo
- **Uso**: Personalización de la experiencia

### Router de Métricas (metricas.py)

Proporciona acceso a las métricas posturales en tiempo real:

```python
@router.get("/metricas/{sesion_id}")
def obtener_metricas(sesion_id: str):
    key = f"metricas:{sesion_id}"
    ultimas = r.lrange(key, -1, -1)  # última métrica
    return json.loads(ultimas[0]) if ultimas else {}
```

**Endpoint**: `GET /metricas/{sesion_id}`
- **Salida**: Última métrica disponible
- **Contenido**:
  - porcentaje_correcta
  - porcentaje_incorrecta
  - transiciones_malas
  - tiempo_sentado/parado
  - alertas_enviadas

### Router de Análisis (analysis.py)

Acceso a los resultados del análisis con IA:

```python
@router.get("/analysis/{sesion_id}")
def obtener_analysis(sesion_id: str):
    key = f"analysis:{sesion_id}"
    raw = r.hgetall(key)
    
    # Convertir valores a enteros
    result: dict[str, int] = {}
    for k, v in raw.items():
        try:
            result[k] = int(v)
        except ValueError:
            result[k] = int(float(v)) if v else 0
    
    return result
```

**Endpoint**: `GET /analysis/{sesion_id}`
- **Salida**: Diccionario con las 13 posturas y sus porcentajes
- **Formato**: `{"Sentado erguido": 85, "Inclinación hacia adelante": 15, ...}`

### Router de Conteo de Posturas (postura_counts.py)

Estadísticas agregadas de posturas detectadas:

```python
@router.get("/{session_id}", response_model=List[PosturaCountOut])
def obtener_postura_counts(session_id: str, db: Session = Depends(get_db)):
    resultados = (
        db.query(PosturaCount)
        .filter(PosturaCount.session_id == session_id)
        .all()
    )
    if not resultados:
        raise HTTPException(status_code=404, detail="No counts found")
    return resultados
```

**Endpoint**: `GET /postura_counts/{session_id}`
- **Salida**: Lista de posturas con sus conteos
- **Uso**: Gráficos y estadísticas de sesión

### Router de Timeline (timeline.py)

Historial temporal de eventos posturales:

```python
@router.get("/{session_id}")
def obtener_timeline(session_id: str):
    key = f"timeline:{session_id}"
    raw = r.lrange(key, 0, -1)
    eventos = []
    
    for item in raw:
        try:
            ev = json.loads(item)
            # Formatear timestamp
            ts = datetime.fromisoformat(ev.get("timestamp"))
            ev["timestamp"] = ts.strftime("%H:%M:%S")
            eventos.append(ev)
        except Exception:
            continue
            
    return eventos
```

**Endpoint**: `GET /timeline/{session_id}`
- **Salida**: Lista cronológica de cambios posturales
- **Formato**: `[{"timestamp": "14:30:15", "postura": "Mentón en mano", "tiempo_mala_postura": 12.5}]`

### Router de Calibración (calibracion.py)

Endpoints específicos para el modo calibración:

#### Progreso de Calibración

```python
@router.get("/progress/{session_id}")
def calib_progress(session_id: str):
    data = r.hgetall(f"calib:{session_id}")
    good = float(data.get("good_time", 0))
    bad = float(data.get("bad_time", 0))
    return {
        "good_time": good,
        "bad_time": bad,
        "correcta": good > bad
    }
```

**Endpoint**: `GET /calib/progress/{session_id}`
- **Función**: Monitorea tiempos en calibración
- **Uso**: Ajuste de umbrales personalizados

#### Gestión de Modos

```python
@router.post("/mode/{device_id}/{mode}")
def set_mode(device_id: str, mode: str):
    if mode not in ("calib", "normal"):
        raise HTTPException(status_code=400, detail="mode must be 'calib' or 'normal'")
    
    key = f"shpd-data:{device_id}"
    r.hset(key, mapping={"mode": mode})
    return {"device_id": device_id, "mode": mode}
```

**Endpoint**: `POST /calib/mode/{device_id}/{mode}`
- **Valores**: "calib" o "normal"
- **Función**: Cambia entre modo calibración y normal

### Documentación Automática

FastAPI genera documentación interactiva automáticamente:

- **Swagger UI**: Disponible en `/docs`
- **ReDoc**: Disponible en `/redoc`
- **OpenAPI Schema**: En `/openapi.json`

### Manejo de Errores

Todos los endpoints implementan manejo consistente de errores:

```python
# 400 Bad Request - Datos inválidos
{"detail": "Validation error", "errors": [...]}

# 404 Not Found - Recurso no existe
{"detail": "Resource not found"}

# 500 Internal Server Error - Error del servidor
{"detail": "Internal server error"}
```

### Autenticación y Seguridad

El sistema actual implementa:
- **CORS habilitado**: Permite acceso desde cualquier origen (desarrollo)
- **Validación de entrada**: Mediante esquemas Pydantic
- **Inyección SQL prevenida**: Uso de ORM con parámetros seguros

### Mejoras de Seguridad Recomendadas

1. **Autenticación JWT**: Para proteger endpoints sensibles
2. **Rate Limiting**: Prevenir abuso de la API
3. **CORS restrictivo**: Limitar orígenes en producción
4. **HTTPS obligatorio**: Encriptación de datos en tránsito
5. **API Keys**: Para integraciones externas

---

## Sistema de Comunicación en Tiempo Real

El backend implementa comunicación bidireccional en tiempo real mediante WebSockets, permitiendo el streaming de video y el procesamiento instantáneo de frames.

### Arquitectura de WebSockets

El sistema utiliza dos endpoints WebSocket principales:

1. **Entrada de Video**: Recibe frames desde el cliente
2. **Salida de Video**: Transmite frames procesados

### WebSocket de Entrada

```python
@app.websocket("/video/input/{device_id}")
async def video_input(websocket: WebSocket, device_id: str):
```

#### Flujo de Procesamiento

1. **Aceptación de Conexión**
```python
await websocket.accept()
loop = asyncio.get_running_loop()
```

2. **Detección de Modo**
```python
calibrating_query = websocket.scope.get("query_string", b"").decode().find("calibracion=1") >= 0
await websocket.send_text(json.dumps({"type": "modo", "calibracion": calibrating_query}))
```

3. **Gestión Dinámica de Sesiones**
```python
while True:
    redis_shpd_key = f"shpd-data:{device_id}"
    session_id = r.hget(redis_shpd_key, "session_id")
    
    if session_id != current_session_id:
        posture_monitor = PostureMonitor(session_id, save_metrics=not calibrating)
        current_session_id = session_id
```

4. **Procesamiento Asíncrono de Frames**
```python
data = await websocket.receive_bytes()
frame = await loop.run_in_executor(None, _decode_jpeg, data)
processed = await loop.run_in_executor(None, posture_monitor.process_frame, frame)
jpeg = await loop.run_in_executor(None, _encode_jpeg, processed)
```

### WebSocket de Salida

```python
@app.websocket("/video/output")
async def video_output(websocket: WebSocket):
    await websocket.accept()
    while True:
        jpeg = await processed_frames_queue.get()
        await websocket.send_bytes(jpeg)
```

### Gestión de Colas

El sistema utiliza colas asíncronas para desacoplar el procesamiento:

```python
processed_frames_queue: asyncio.Queue = asyncio.Queue(maxsize=10)
```

Características:
- **Tamaño limitado**: Previene desbordamiento de memoria
- **Política FIFO**: Frames más antiguos se descartan si la cola está llena
- **Backpressure automático**: El productor se bloquea si la cola está llena

### Optimización de Rendimiento

1. **Procesamiento en Thread Pool**
```python
await loop.run_in_executor(None, posture_monitor.process_frame, frame)
```
- Evita bloquear el event loop
- Permite procesamiento paralelo de múltiples conexiones

2. **Compresión JPEG Adaptativa**
```python
cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
```
- Calidad 50: Balance entre calidad visual y ancho de banda
- Reduce latencia de transmisión

3. **Gestión de Desconexiones**
```python
except WebSocketDisconnect:
    logger.info(f"WebSocket desconectado para device_id: {device_id}")
```

### Protocolo de Comunicación

#### Mensajes de Control
```json
{
    "type": "modo",
    "calibracion": true
}
```

#### Formato de Datos
- **Entrada**: Bytes JPEG raw
- **Salida**: Bytes JPEG procesados con overlays visuales

---

## Integración con OpenAI

El sistema utiliza la API de OpenAI GPT-4 Vision para análisis avanzado de posturas cuando MediaPipe detecta anomalías sostenidas.

### Configuración del Cliente

```python
API_KEY = os.getenv("OPENAI_API_KEY", "sk-proj-...")
client = OpenAI(api_key=API_KEY)
MODEL = "gpt-4o-mini"
```

### Construcción de Prompts

El sistema utiliza un prompt especializado para clasificación de posturas:

```python
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

Determina y devuelve estrictamente un objeto JSON, sin cercos de código circundante.
Para cada postura devuelve un porcentaje entre 0 y 100.
"""
```

### Worker de Análisis Asíncrono

```python
async def api_analysis_worker():
    loop = asyncio.get_running_loop()
    while True:
        payload = await api_analysis_queue.get()
        
        # Codificar imagen
        b64 = base64.b64encode(payload["jpeg"]).decode("utf-8")
        
        # Llamar a OpenAI
        response = await loop.run_in_executor(
            None,
            lambda: client.chat.completions.create(
                user=payload["session_id"],
                model=MODEL,
                messages=build_openai_messages(b64),
                temperature=0,
                max_tokens=500
            )
        )
```

### Procesamiento de Resultados

```python
content = response.choices[0].message.content
result: dict[str, int] = json.loads(content)

# Encontrar postura predominante
top_label, top_value = max(result.items(), key=lambda kv: kv[1])

# Actualizar base de datos
fila = db.query(PosturaCount).filter(
    PosturaCount.session_id == session_id,
    PosturaCount.posture_label == top_label
).first()

if fila:
    fila.count += 1
else:
    nueva = PosturaCount(
        session_id=session_id,
        posture_label=top_label,
        count=1
    )
    db.add(nueva)
```

### Gestión de Eventos

```python
# Guardar en timeline
evt = {
    "timestamp": datetime.utcnow().isoformat(),
    "postura": top_label,
    "tiempo_mala_postura": bad_time
}
r.rpush(f"timeline:{session_id}", json.dumps(evt))
r.ltrim(f"timeline:{session_id}", -200, -1)  # Mantener últimos 200 eventos
```

### Optimizaciones

1. **Procesamiento Asíncrono**: No bloquea el flujo principal
2. **Rate Limiting Implícito**: Solo se analiza cuando hay alertas
3. **Caché de Resultados**: Se almacenan en Redis para consulta rápida
4. **Modelo Optimizado**: GPT-4o-mini para balance costo/rendimiento

---

## Sistema de Caché con Redis

Redis actúa como columna vertebral para el almacenamiento temporal y la comunicación entre componentes.

### Conexión y Configuración

```python
r = redis.Redis(host="redis", port=6379, decode_responses=True)
```

### Estructuras de Datos Utilizadas

#### 1. Datos de Sesión (Hash)
```python
Key: shpd-data:{device_id}
Fields:
  - session_id: UUID de la sesión activa
  - mode: "normal" o "calib"
  - good_frames: Contador de frames correctos
  - bad_frames: Contador de frames incorrectos
  - transiciones_malas: Contador de cambios posturales
  - tiempo_sentado: Tiempo acumulado sentado
  - tiempo_parado: Tiempo acumulado de pie
  - alert_count: Número de alertas generadas
```

#### 2. Información de Sesión (Hash)
```python
Key: shpd-session:{session_id}
Fields:
  - start_ts: Timestamp de inicio (Unix)
  - intervalo_segundos: Duración planificada
```

#### 3. Métricas en Tiempo Real (List)
```python
Key: metricas:{session_id}
Formato: JSON strings con estructura:
{
    "porcentaje_correcta": 75.5,
    "porcentaje_incorrecta": 24.5,
    "transiciones_malas": 5,
    "tiempo_sentado": 300.0,
    "tiempo_parado": 10.0,
    "alertas_enviadas": 2
}
```

#### 4. Resultados de Análisis (Hash)
```python
Key: analysis:{session_id}
Fields: Cada postura con su porcentaje
Example:
  "Sentado erguido": "85"
  "Inclinación hacia adelante": "15"
```

#### 5. Timeline de Eventos (List)
```python
Key: timeline:{session_id}
Formato: JSON strings con estructura:
{
    "timestamp": "2024-01-15T14:30:15",
    "postura": "Mentón en mano",
    "tiempo_mala_postura": 12.5
}
```

#### 6. Datos de Calibración (Hash)
```python
Key: calib:{session_id}
Fields:
  - good_time: Tiempo acumulado en buena postura
  - bad_time: Tiempo acumulado en mala postura
```

### Operaciones Comunes

#### Incremento Atómico
```python
r.hincrby(buffer_key, "good_frames", 1)
r.hincrbyfloat(buffer_key, "tiempo_sentado", round(delta, 1))
```

#### Gestión de Listas con Límite
```python
r.rpush(key, json.dumps(datos))
r.ltrim(key, -50, -1)  # Mantener últimos 50 elementos
```

#### TTL para Datos Temporales
```python
r.setex(ended_key, 3600, "1")  # Expira en 1 hora
```

### Ventajas del Uso de Redis

1. **Velocidad**: Acceso en microsegundos
2. **Atomicidad**: Operaciones atómicas para contadores
3. **Pub/Sub**: Potencial para eventos en tiempo real
4. **Persistencia Opcional**: Snapshots para recuperación
5. **Escalabilidad**: Clustering para alta disponibilidad

---

## Despliegue y Containerización

El sistema está diseñado para desplegarse en entornos containerizados usando Docker y Kubernetes.

### Dockerfile

```dockerfile
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \    # OpenGL para OpenCV
    libglib2.0-0 \       # GLib para MediaPipe
    libsm6 \             # Session Management
    libxext6 \           # X11 extensions
    libxrender1 \        # X Rendering Extension
    ffmpeg \             # Procesamiento multimedia
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instalar dependencias Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código fuente
COPY . .

CMD ["python", "main.py"]
```

### Configuración de Kubernetes

#### Deployment del Backend
```yaml
# shpd-backend.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: shpd-backend
spec:
  replicas: 2
  selector:
    matchLabels:
      app: shpd-backend
  template:
    spec:
      containers:
      - name: backend
        image: shpd-backend:latest
        ports:
        - containerPort: 8765
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-credentials
              key: url
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: openai-credentials
              key: api-key
```

#### Service del Backend
```yaml
# backend-svc.yaml
apiVersion: v1
kind: Service
metadata:
  name: backend-service
spec:
  selector:
    app: shpd-backend
  ports:
  - protocol: TCP
    port: 8765
    targetPort: 8765
  type: LoadBalancer
```

#### Deployment de Redis
```yaml
# redis-deploy.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redis
spec:
  replicas: 1
  selector:
    matchLabels:
      app: redis
  template:
    spec:
      containers:
      - name: redis
        image: redis:7-alpine
        ports:
        - containerPort: 6379
        volumeMounts:
        - name: redis-storage
          mountPath: /data
```

### Variables de Entorno

```bash
# Producción
DATABASE_URL=postgresql://user:pass@postgres-service:5432/shpd_db
REDIS_URL=redis://redis-service:6379/0
OPENAI_API_KEY=sk-proj-...

# Desarrollo
DATABASE_URL=postgresql://localhost:5432/shpd_dev
REDIS_URL=redis://localhost:6379/0
```

### Pipeline de CI/CD

```yaml
# .gitlab-ci.yml ejemplo
stages:
  - build
  - test
  - deploy

build:
  stage: build
  script:
    - docker build -t $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA .
    - docker push $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA

test:
  stage: test
  script:
    - pip install -r requirements.txt
    - pytest tests/

deploy:
  stage: deploy
  script:
    - kubectl set image deployment/shpd-backend backend=$CI_REGISTRY_IMAGE:$CI_COMMIT_SHA
```

### Monitoreo y Logging

1. **Prometheus Metrics**: Exportar métricas de rendimiento
2. **ELK Stack**: Centralización de logs
3. **Health Checks**: Endpoints de salud para K8s
4. **Distributed Tracing**: Jaeger para trazabilidad

### Consideraciones de Producción

1. **Escalado Horizontal**: Múltiples réplicas del backend
2. **Load Balancing**: Distribución de carga WebSocket
3. **SSL/TLS**: Terminación en el ingress controller
4. **Backup**: Snapshots periódicos de PostgreSQL y Redis
5. **Secrets Management**: Kubernetes secrets o HashiCorp Vault

---

## Flujo de Datos y Casos de Uso

### Flujo General del Sistema

El sistema sigue un flujo de datos bien definido desde la captura de video hasta la generación de reportes:

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│   Cliente   │────▶│  WebSocket   │────▶│ PostureMonitor  │
│  (Cámara)   │     │   /input     │     │   (MediaPipe)   │
└─────────────┘     └──────────────┘     └────────┬────────┘
                                                   │
                           ┌───────────────────────┴────────────────┐
                           │                                        │
                    ┌──────▼──────┐                         ┌──────▼──────┐
                    │   Análisis  │                         │    Redis    │
                    │   Básico    │                         │   (Caché)   │
                    └──────┬──────┘                         └──────┬──────┘
                           │                                        │
                           │ Si alerta                              │
                    ┌──────▼──────┐                                │
                    │   OpenAI    │                                │
                    │   Worker    │                                │
                    └──────┬──────┘                                │
                           │                                        │
                           └────────────────┬───────────────────────┘
                                           │
                                    ┌──────▼──────┐
                                    │ PostgreSQL  │
                                    │    (BD)     │
                                    └──────┬──────┘
                                           │
                    ┌──────────────────────┴────────────────────┐
                    │                                           │
             ┌──────▼──────┐                            ┌──────▼──────┐
             │  REST API   │                            │  WebSocket   │
             │  Endpoints  │                            │   /output    │
             └──────┬──────┘                            └──────┬──────┘
                    │                                           │
                    └───────────────┬───────────────────────────┘
                                   │
                            ┌──────▼──────┐
                            │   Cliente   │
                            │ (Frontend)  │
                            └─────────────┘
```

### Casos de Uso Principales

#### 1. Sesión de Monitoreo Normal

**Actor**: Usuario final (paciente)
**Flujo**:
1. El usuario inicia sesión desde la aplicación web
2. Se crea una nueva sesión con duración predefinida
3. La cámara comienza a transmitir video vía WebSocket
4. MediaPipe analiza cada frame detectando puntos clave
5. Se calculan ángulos posturales en tiempo real
6. Si se detecta mala postura sostenida:
   - Se genera una alerta visual
   - Se envía el frame a OpenAI para análisis detallado
   - Se registra el evento en la timeline
7. Al finalizar la sesión:
   - Se genera un reporte completo
   - Se envía resumen por Telegram
   - Se limpian datos temporales

**Código relevante**:
```python
# Creación de sesión
POST /sesiones/
{
    "intervalo_segundos": 1800,
    "modo": "normal"
}

# Monitoreo en tiempo real
WS /video/input/{device_id}

# Consulta de métricas
GET /metricas/{session_id}

# Finalización
POST /sesiones/end/{device_id}
```

#### 2. Modo Calibración

**Actor**: Usuario configurando el sistema
**Flujo**:
1. Usuario activa modo calibración
2. Se realizan posturas de referencia
3. El sistema mide tiempos en buena/mala postura
4. Usuario ajusta umbrales según feedback visual
5. Los nuevos umbrales se guardan para futuras sesiones

**Código relevante**:
```python
# Activar calibración
POST /calib/mode/{device_id}/calib

# Monitorear progreso
GET /calib/progress/{session_id}

# WebSocket con query parameter
WS /video/input/{device_id}?calibracion=1
```

#### 3. Análisis Histórico

**Actor**: Profesional de salud
**Flujo**:
1. Accede al historial de sesiones del paciente
2. Visualiza métricas agregadas por sesión
3. Revisa timeline de eventos posturales
4. Analiza tendencias en tipos de posturas
5. Genera reportes para seguimiento

**Endpoints utilizados**:
```python
# Listar sesiones
GET /sesiones/

# Obtener conteos de posturas
GET /postura_counts/{session_id}

# Ver timeline detallada
GET /timeline/{session_id}

# Análisis por IA
GET /analysis/{session_id}
```

#### 4. Integración con Bot de Telegram

**Actor**: Sistema automatizado
**Flujo**:
1. Al finalizar sesión, se recopilan métricas finales
2. Se consulta información del paciente
3. Se formatea reporte con estadísticas clave
4. Se envía vía API al servicio del bot
5. Usuario recibe notificación en Telegram

**Implementación**:
```python
def enviar_reporte_telegram(session_id, device_id, db: Session):
    # Obtener datos del paciente
    paciente = db.query(Paciente).filter(
        Paciente.device_id == device_id
    ).first()
    
    # Recopilar métricas
    metricas = r.lrange(f"metricas:{session_id}", 0, -1)
    ultima = json.loads(metricas[-1]) if metricas else {}
    
    # Formatear y enviar
    resumen = f"✅ Reporte de sesión\n..."
    payload = {"telegram_id": paciente.telegram_id, "resumen": resumen}
    requests.post(BOT_API_URL, json=payload)
```

### Flujo de Procesamiento de Video

#### 1. Recepción de Frame
```python
data = await websocket.receive_bytes()  # JPEG comprimido
frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
```

#### 2. Detección de Pose
```python
image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
keypoints = self.pose.process(image_rgb)  # MediaPipe
```

#### 3. Cálculo de Métricas
```python
neck_angle = self.findAngle(shoulder_x, shoulder_y, ear_x, ear_y)
torso_angle = self.findAngle(hip_x, hip_y, shoulder_x, shoulder_y)
```

#### 4. Evaluación y Visualización
```python
if neck_angle < threshold and torso_angle < threshold:
    # Postura correcta - visualización verde
else:
    # Postura incorrecta - visualización roja
    # Posible disparo de análisis con IA
```

#### 5. Transmisión de Resultado
```python
processed_jpeg = cv2.imencode('.jpg', processed_frame)[1].tobytes()
await processed_frames_queue.put(processed_jpeg)
```

### Gestión de Estados

El sistema mantiene varios estados concurrentes:

1. **Estado de Sesión**: Activa, pausada, finalizada
2. **Estado de Calibración**: Normal o calibración
3. **Estado de Alerta**: Pendiente o enviada
4. **Estado de Conexión**: Conectado o desconectado

Estos estados se gestionan principalmente en Redis para acceso rápido y consistente entre componentes.

---

## Análisis de Fortalezas y Oportunidades de Mejora

### Fortalezas del Diseño Actual

#### 1. Arquitectura Asíncrona
- **Ventaja**: Alta concurrencia sin bloqueos
- **Implementación**: FastAPI + asyncio permiten manejar múltiples streams simultáneos
- **Beneficio**: Escalabilidad vertical eficiente

#### 2. Procesamiento en Tiempo Real
- **Ventaja**: Feedback inmediato al usuario
- **Implementación**: MediaPipe optimizado + WebSockets
- **Beneficio**: Experiencia de usuario fluida

#### 3. Integración de IA Inteligente
- **Ventaja**: Análisis detallado solo cuando es necesario
- **Implementación**: Disparadores basados en umbrales
- **Beneficio**: Optimización de costos de API

#### 4. Persistencia Híbrida
- **Ventaja**: Balance entre velocidad y durabilidad
- **Implementación**: Redis para tiempo real + PostgreSQL para históricos
- **Beneficio**: Rendimiento óptimo sin perder datos

#### 5. Modularidad del Código
- **Ventaja**: Fácil mantenimiento y extensión
- **Implementación**: Routers separados por dominio
- **Beneficio**: Desarrollo paralelo de features

#### 6. Containerización Completa
- **Ventaja**: Despliegue consistente
- **Implementación**: Docker + Kubernetes
- **Beneficio**: DevOps simplificado

### Oportunidades de Mejora

#### 1. Sistema de Autenticación
**Situación Actual**: Sin autenticación implementada
**Propuesta**:
```python
# Implementar JWT con FastAPI-Users
from fastapi_users import FastAPIUsers
from fastapi_users.authentication import JWTAuthentication

auth_backend = JWTAuthentication(secret=SECRET, lifetime_seconds=3600)
```

#### 2. Optimización de Análisis con IA
**Situación Actual**: Una imagen por alerta
**Propuesta**:
- Implementar batching de imágenes
- Análisis de secuencias temporales
- Cache de resultados similares

#### 3. Escalabilidad de WebSockets
**Situación Actual**: Single-instance queue
**Propuesta**:
```python
# Implementar Redis Pub/Sub para multi-instancia
async def video_input_scalable(websocket: WebSocket, device_id: str):
    channel = f"frames:{device_id}"
    pubsub = r.pubsub()
    await pubsub.subscribe(channel)
```

#### 4. Métricas de Sistema
**Situación Actual**: Logging básico
**Propuesta**:
```python
# Integrar Prometheus
from prometheus_client import Counter, Histogram, Gauge

frames_processed = Counter('frames_processed_total', 'Total frames processed')
processing_time = Histogram('frame_processing_seconds', 'Time to process frame')
active_sessions = Gauge('active_sessions', 'Number of active sessions')
```

#### 5. Validación de Datos Mejorada
**Situación Actual**: Validación básica con Pydantic
**Propuesta**:
```python
# Esquemas más estrictos
class MetricaIn(BaseModel):
    sesion_id: UUID
    timestamp: datetime
    datos: dict
    
    @validator('datos')
    def validate_datos(cls, v):
        required_keys = ['porcentaje_correcta', 'porcentaje_incorrecta']
        if not all(k in v for k in required_keys):
            raise ValueError('Faltan campos requeridos')
        return v
```

#### 6. Tests Automatizados
**Situación Actual**: Sin tests visibles
**Propuesta**:
```python
# tests/test_posture_monitor.py
import pytest
from posture_monitor import PostureMonitor

@pytest.fixture
def monitor():
    return PostureMonitor("test-session", save_metrics=False)

def test_angle_calculation(monitor):
    angle = monitor.findAngle(0, 0, 1, 1)
    assert 40 < angle < 50  # ~45 grados
```

#### 7. Gestión de Configuración
**Situación Actual**: Variables hardcodeadas
**Propuesta**:
```python
# config.py
from pydantic import BaseSettings

class Settings(BaseSettings):
    database_url: str
    redis_url: str
    openai_api_key: str
    jwt_secret: str
    
    class Config:
        env_file = ".env"

settings = Settings()
```

#### 8. Documentación de API Mejorada
**Situación Actual**: Documentación automática básica
**Propuesta**:
```python
@router.post("/sesiones/", 
    response_model=SesionOut,
    summary="Crear nueva sesión",
    description="""
    Crea una nueva sesión de monitoreo postural.
    
    La sesión incluye:
    - Duración en segundos
    - Modo de operación (normal/calibración)
    - UUID único generado automáticamente
    """,
    response_description="Sesión creada exitosamente"
)
```

### Matriz de Priorización

| Mejora | Impacto | Esfuerzo | Prioridad |
|--------|---------|----------|-----------|
| Autenticación | Alto | Medio | 1 |
| Tests | Alto | Bajo | 2 |
| Configuración | Medio | Bajo | 3 |
| Métricas | Medio | Medio | 4 |
| Escalabilidad WS | Bajo | Alto | 5 |

---

## Conclusiones

### Logros del Proyecto

El backend desarrollado para el sistema de detección de posturas representa una solución integral y moderna que cumple exitosamente con los objetivos planteados:

1. **Procesamiento en Tiempo Real**: La implementación con MediaPipe y WebSockets permite análisis instantáneo de posturas con latencia mínima.

2. **Inteligencia Artificial Aplicada**: La integración con OpenAI GPT-4 Vision proporciona análisis detallado de 13 tipos diferentes de posturas, superando las capacidades de sistemas tradicionales.

3. **Arquitectura Escalable**: El diseño basado en microservicios con FastAPI, Redis y PostgreSQL permite crecer horizontalmente según la demanda.

4. **Experiencia de Usuario Optimizada**: El feedback visual inmediato y las alertas inteligentes crean una experiencia fluida y educativa.

5. **Integración Multiplataforma**: El soporte para web y Telegram amplía el alcance y la utilidad del sistema.

### Contribuciones Técnicas

#### 1. Patrón de Procesamiento Híbrido
La combinación de análisis local (MediaPipe) con análisis en la nube (OpenAI) establece un nuevo paradigma de eficiencia:
- Procesamiento rápido para detección básica
- Análisis profundo solo cuando es necesario
- Optimización de costos sin sacrificar funcionalidad

#### 2. Gestión de Estado Distribuido
El uso de Redis como "source of truth" temporal demuestra cómo manejar estado en aplicaciones distribuidas:
- Sincronización entre componentes sin acoplamiento
- Recuperación ante fallos
- Métricas en tiempo real sin impacto en rendimiento

#### 3. Pipeline de Video Asíncrono
La implementación de WebSockets con procesamiento asíncrono establece mejores prácticas para aplicaciones de video:
- Sin bloqueos en el event loop
- Gestión eficiente de backpressure
- Escalabilidad natural

### Impacto en Salud Postural

Este sistema tiene el potencial de:
1. **Prevenir lesiones**: Detección temprana de malas posturas
2. **Educar usuarios**: Feedback visual para crear conciencia
3. **Facilitar seguimiento médico**: Datos objetivos para profesionales
4. **Personalizar tratamientos**: Calibración individual de umbrales

### Direcciones Futuras

#### Corto Plazo (3-6 meses)
1. Implementar sistema de autenticación robusto
2. Agregar suite completa de tests
3. Mejorar documentación para desarrolladores
4. Optimizar algoritmos de detección

#### Mediano Plazo (6-12 meses)
1. Desarrollar modelos de ML propios para reducir dependencia de APIs externas
2. Implementar análisis predictivo de problemas posturales
3. Añadir soporte para múltiples cámaras
4. Crear dashboard analítico avanzado

#### Largo Plazo (12+ meses)
1. Expandir a detección de ejercicios y rehabilitación
2. Integrar con dispositivos IoT y wearables
3. Desarrollar versión móvil nativa
4. Establecer plataforma SaaS completa

### Reflexión Final

El desarrollo de este backend demuestra cómo las tecnologías modernas pueden combinarse para crear soluciones que impacten positivamente en la salud de las personas. La arquitectura implementada no solo resuelve el problema inmediato de detección de posturas, sino que establece una base sólida para futuras innovaciones en el campo de la salud digital.

La combinación de procesamiento en tiempo real, inteligencia artificial, y diseño centrado en el usuario crea un sistema que es tanto técnicamente robusto como prácticamente útil. Este proyecto sirve como ejemplo de cómo la ingeniería de software puede contribuir directamente al bienestar humano, estableciendo un puente entre la tecnología avanzada y las necesidades cotidianas de salud.

El código fuente completo, junto con esta documentación, proporciona no solo una solución funcional, sino también un recurso educativo para futuros desarrolladores interesados en aplicaciones de salud digital, procesamiento de video en tiempo real, y arquitecturas de microservicios modernas.

---

*Fin del documento*