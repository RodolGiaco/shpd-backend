# Documentación Técnica del Archivo `main.py`
## Sistema de Monitoreo Postural Inteligente

### 1. Propósito Principal del Archivo

El archivo `main.py` constituye el núcleo central del sistema de monitoreo postural inteligente, implementando un servidor web asíncrono que coordina la captura, procesamiento y análisis de video en tiempo real. Este componente actúa como punto de convergencia entre el hardware de captura (Raspberry Pi con cámara), el procesamiento de inteligencia artificial (OpenAI GPT-4 Vision) y la interfaz de usuario frontend.

### 2. Estructura General del Archivo

El archivo se organiza en las siguientes secciones principales:

#### 2.1 Importaciones y Dependencias
```python
import os, asyncio, logging, json, time, cv2, numpy as np
import redis, base64, logging.config
from contextlib import asynccontextmanager
from openai import OpenAI
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
```

#### 2.2 Configuración de Logging
#### 2.3 Configuración de OpenAI
#### 2.4 Colas Asíncronas
#### 2.5 Worker de Análisis
#### 2.6 Configuración de FastAPI
#### 2.7 Endpoints WebSocket
#### 2.8 Funciones de Utilidad

### 3. Tecnologías Utilizadas

#### 3.1 FastAPI
Framework web moderno para Python que proporciona:
- **Rendimiento asíncrono**: Permite manejar múltiples conexiones simultáneas sin bloqueo
- **WebSockets nativos**: Soporte integrado para comunicación bidireccional en tiempo real
- **Validación automática**: Generación automática de documentación OpenAPI
- **Tipado estático**: Mejora la robustez del código mediante anotaciones de tipo

#### 3.2 OpenCV (cv2)
Biblioteca de visión por computadora que facilita:
- **Procesamiento de imágenes**: Decodificación y codificación de frames JPEG
- **Manipulación de arrays**: Conversión entre formatos de imagen
- **Optimización de rendimiento**: Operaciones vectorizadas para procesamiento eficiente

#### 3.3 Redis
Sistema de almacenamiento en memoria que proporciona:
- **Almacenamiento temporal**: Buffer para frames y métricas en tiempo real
- **Comunicación interproceso**: Coordinación entre diferentes componentes del sistema
- **Estructuras de datos especializadas**: Colas, hashes y listas para diferentes tipos de datos

#### 3.4 OpenAI GPT-4 Vision
Modelo de inteligencia artificial que realiza:
- **Análisis postural avanzado**: Clasificación de 13 tipos diferentes de posturas
- **Procesamiento de imágenes**: Análisis visual de frames capturados
- **Respuestas estructuradas**: Generación de JSON con porcentajes de confianza

#### 3.5 asyncio
Biblioteca de programación asíncrona que permite:
- **Concurrencia no bloqueante**: Manejo eficiente de múltiples operaciones simultáneas
- **Colas asíncronas**: Comunicación entre diferentes workers del sistema
- **Event loops**: Gestión de eventos y tareas programadas

### 4. Explicación Detallada por Secciones

#### 4.1 Configuración de Logging

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
    # ... configuración de handlers y loggers
}
```

**Propósito**: Establece un sistema de logging jerárquico que permite:
- **Monitoreo en tiempo real**: Seguimiento del comportamiento del sistema
- **Depuración eficiente**: Identificación rápida de errores y problemas
- **Análisis de rendimiento**: Medición de tiempos de respuesta y latencias

**Importancia**: En un sistema de monitoreo postural, el logging es crítico para:
- Verificar el correcto funcionamiento del procesamiento de video
- Detectar fallos en la comunicación con OpenAI
- Monitorear el rendimiento del sistema en hardware limitado

#### 4.2 Configuración de OpenAI

```python
API_KEY = os.getenv("OPENAI_API_KEY", "sk-proj-...")
client = OpenAI(api_key=API_KEY)
MODEL = "gpt-4o-mini"
```

**Propósito**: Configura el cliente de OpenAI para análisis postural avanzado.

**Funcionalidad**: 
- **Autenticación segura**: Uso de variables de entorno para credenciales
- **Modelo optimizado**: GPT-4o-mini para balance entre precisión y velocidad
- **Configuración centralizada**: Fácil modificación de parámetros del modelo

#### 4.3 Función `build_openai_messages()`

```python
def build_openai_messages(b64: str) -> list[dict]:
    SYSTEM_PROMPT = """Eres un asistente de clasificación de posturas basado en visión.
    Cuando recibas una imagen, analiza la postura de la persona 
    qué tan predominante es cada una de las siguientes 13 posturas al sentarse.
    Las posturas son:
    - Sentado erguido
    - Inclinación hacia adelante
    # ... más posturas
    """
```

**Propósito**: Construye los mensajes estructurados para la API de OpenAI.

**Características técnicas**:
- **Prompt de sistema**: Define el contexto y las 13 posturas a clasificar
- **Formato JSON**: Solicita respuestas estructuradas con porcentajes (0-100)
- **Codificación Base64**: Convierte imágenes JPEG a formato compatible con la API

**Integración**: Se conecta con el worker asíncrono para procesar frames cuando se detectan alertas posturales.

#### 4.4 Colas Asíncronas

```python
processed_frames_queue: asyncio.Queue = asyncio.Queue(maxsize=10)
api_analysis_queue: asyncio.Queue = asyncio.Queue()
```

**Propósito**: Implementa un sistema de comunicación asíncrona entre componentes.

**Funcionalidad**:
- **`processed_frames_queue`**: Buffer circular para frames procesados (máximo 10)
- **`api_analysis_queue`**: Cola para solicitudes de análisis a OpenAI
- **Gestión de memoria**: Previene desbordamientos en hardware limitado

#### 4.5 Worker de Análisis (`api_analysis_worker()`)

```python
async def api_analysis_worker():
    """
    Worker que consume de api_analysis_queue payloads con:
      { session_id, b64, exercise }
    Llama a OpenAI y guarda el JSON resultante en Redis bajo analysis:{session_id}.
    """
```

**Propósito**: Procesa asíncronamente las solicitudes de análisis postural.

**Flujo de trabajo**:
1. **Consumo de cola**: Espera payloads de análisis desde la cola asíncrona
2. **Codificación**: Convierte frames JPEG a Base64
3. **Llamada a OpenAI**: Ejecuta análisis en executor para no bloquear el loop
4. **Procesamiento de respuesta**: Extrae y valida el JSON de clasificación
5. **Almacenamiento**: Guarda resultados en Redis y base de datos
6. **Actualización de métricas**: Incrementa contadores de posturas detectadas

**Optimizaciones para hardware limitado**:
- **Ejecutor separado**: Evita bloqueo del event loop principal
- **Manejo de errores robusto**: Continúa funcionando ante fallos de API
- **Limpieza automática**: Libera recursos después de cada análisis

#### 4.6 Configuración de FastAPI

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(api_analysis_worker())
    logger.debug("✅ API analysis worker scheduled")
    yield  

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**Propósito**: Configura la aplicación web con middleware y workers de inicio.

**Características**:
- **Lifespan management**: Inicia workers automáticamente al arrancar la aplicación
- **CORS habilitado**: Permite comunicación desde frontend en diferentes dominios
- **Routers incluidos**: Integra endpoints para gestión de sesiones, pacientes y métricas

#### 4.7 Endpoint WebSocket `/video/input/{device_id}`

```python
@app.websocket("/video/input/{device_id}")
async def video_input(websocket: WebSocket, device_id: str):
    await websocket.accept()
    loop = asyncio.get_running_loop()
    
    # Detectar modo calibración: query ?calibracion=1
    calibrating_query = websocket.scope.get("query_string", b"").decode().find("calibracion=1") >= 0
```

**Propósito**: Recibe streams de video desde dispositivos Raspberry Pi.

**Funcionalidades principales**:

**a) Detección de modo de operación**:
- **Modo normal**: Procesamiento completo con análisis postural
- **Modo calibración**: Solo procesamiento básico sin guardado de métricas

**b) Gestión dinámica de sesiones**:
```python
redis_shpd_key = f"shpd-data:{device_id}"
session_id = r.hget(redis_shpd_key, "session_id")
```

**c) Procesamiento de frames**:
```python
data = await websocket.receive_bytes()
frame = await loop.run_in_executor(None, _decode_jpeg, data)
if frame is None:
    continue
```

**d) Integración con PostureMonitor**:
```python
if session_id != current_session_id:
    posture_monitor = PostureMonitor(session_id, save_metrics=not calibrating)
    current_session_id = session_id
```

**e) Disparo de análisis OpenAI**:
```python
if posture_monitor is not None and not calibrating:
    raw_key = f"raw_frame:{session_id}"
    flag_value = r.hget(raw_key, "flag_alert")  
    if flag_value == "1":
        await api_analysis_queue.put({
            "session_id": session_id,
            "jpeg": jpeg,
            "bad_time": bad_time
        })
```

#### 4.8 Endpoint WebSocket `/video/output`

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

**Propósito**: Distribuye frames procesados a clientes frontend.

**Características**:
- **Streaming en tiempo real**: Envía frames procesados inmediatamente
- **Manejo de desconexiones**: Limpia recursos automáticamente
- **Buffer circular**: Previene acumulación excesiva de frames

#### 4.9 Funciones de Utilidad

**`_decode_jpeg(data: bytes)`**:
```python
def _decode_jpeg(data: bytes):
    arr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)
```

**`_encode_jpeg(frame)`**:
```python
def _encode_jpeg(frame):
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
    return buf.tobytes()
```

**Propósito**: Optimizan la conversión entre formatos de imagen.

**Optimizaciones para Raspberry Pi**:
- **Calidad JPEG reducida**: 50% para minimizar ancho de banda
- **Procesamiento vectorizado**: Uso de NumPy para eficiencia
- **Gestión de memoria**: Conversión directa sin copias innecesarias

### 5. Integración del Archivo `main.py` en el Proyecto

#### 5.1 Arquitectura del Sistema

El archivo `main.py` actúa como **orquestador central** que coordina:

**Entrada de datos**:
- **Raspberry Pi**: Envía frames JPEG vía WebSocket
- **Redis**: Proporciona configuración de sesiones y almacenamiento temporal
- **Base de datos**: Almacena métricas y resultados de análisis

**Procesamiento**:
- **PostureMonitor**: Análisis básico de postura con MediaPipe
- **OpenAI GPT-4 Vision**: Clasificación avanzada de 13 tipos de postura
- **Workers asíncronos**: Procesamiento paralelo sin bloqueo

**Salida de datos**:
- **Frontend**: Recibe frames procesados y métricas en tiempo real
- **Base de datos**: Almacena resultados de análisis y contadores
- **Redis**: Proporciona timeline de eventos y alertas

#### 5.2 Flujo de Datos Completo

1. **Captura**: Raspberry Pi captura frame y lo envía vía WebSocket
2. **Recepción**: `main.py` recibe el frame en `/video/input/{device_id}`
3. **Procesamiento básico**: PostureMonitor analiza ángulos de cuello y torso
4. **Detección de alerta**: Si se supera el umbral de tiempo, se activa flag
5. **Análisis avanzado**: Worker envía frame a OpenAI para clasificación detallada
6. **Almacenamiento**: Resultados se guardan en Redis y base de datos
7. **Distribución**: Frame procesado se envía al frontend vía `/video/output`

#### 5.3 Integración con Módulos Externos

**`posture_monitor.py`**:
- Proporciona análisis básico de postura usando MediaPipe
- Calcula ángulos de inclinación de cuello y torso
- Detecta transiciones entre posturas correctas e incorrectas

**`api/models.py`**:
- Define esquemas de base de datos para sesiones, pacientes y métricas
- Proporciona modelos SQLAlchemy para persistencia de datos

**`api/routers/`**:
- Endpoints REST para gestión de sesiones, pacientes y análisis
- Proporciona API para consulta de métricas y configuración

### 6. Consideraciones para Hardware Limitado (Raspberry Pi)

#### 6.1 Optimizaciones de Rendimiento

**Procesamiento asíncrono**:
- **Event loop único**: Evita overhead de múltiples threads
- **Workers especializados**: Separación de responsabilidades para mejor rendimiento
- **Colas con límites**: Previene desbordamiento de memoria

**Gestión de memoria**:
```python
if processed_frames_queue.full():
    processed_frames_queue.get_nowait()
await processed_frames_queue.put(jpeg)
```

**Compresión de imágenes**:
```python
_, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
```

#### 6.2 Adaptaciones para Redes Limitadas

**WebSockets persistentes**: Evita overhead de reconexiones HTTP
**Codificación eficiente**: Base64 optimizado para transmisión
**Buffer circular**: Limita uso de memoria en dispositivos con recursos limitados

#### 6.3 Manejo de Errores Robusto

**Reconexión automática**: WebSockets se reconectan automáticamente
**Fallback graceful**: Sistema continúa funcionando ante fallos de componentes
**Logging detallado**: Facilita diagnóstico en dispositivos remotos

### 7. Configuración de Inicio

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

**Propósito**: Inicializa la aplicación y crea tablas de base de datos.

**Configuración de producción**:
- **Host 0.0.0.0**: Acepta conexiones desde cualquier interfaz
- **Puerto 8765**: Puerto estándar para el sistema de monitoreo postural
- **Logging configurado**: Balance entre información y rendimiento

### 8. Conclusiones

El archivo `main.py` representa una implementación sofisticada de un sistema de monitoreo postural en tiempo real, caracterizado por:

**Arquitectura escalable**: Diseño modular que permite fácil extensión y mantenimiento
**Rendimiento optimizado**: Uso eficiente de recursos para hardware limitado
**Robustez operacional**: Manejo robusto de errores y recuperación automática
**Integración avanzada**: Combinación de visión por computadora tradicional e inteligencia artificial

Este componente central demuestra las capacidades de Python moderno para aplicaciones de IoT y sistemas de monitoreo en tiempo real, proporcionando una base sólida para el desarrollo de soluciones de salud digital innovadoras.