# Documentación de Adaptación SHPD para Raspberry Pi

## 1. Introducción

### 1.1 Objetivo del Proyecto

El **Smart Healthy Posture Detector (SHPD)** es un sistema de monitoreo postural en tiempo real que ha sido adaptado para ejecutarse íntegramente en una **Raspberry Pi 3 B+**. Esta adaptación elimina la dependencia de servicios externos y microservicios distribuidos, consolidando toda la funcionalidad en un único dispositivo embebido.

### 1.2 Características Principales de la Adaptación

- **Detección Postural Local**: Reemplazo del modelo OpenAI por un modelo TensorFlow Lite personalizado
- **Procesamiento en Tiempo Real**: Análisis de video mediante MediaPipe y OpenCV
- **Almacenamiento Local**: Base de datos SQLite integrada para persistencia de datos
- **Interfaz Web Local**: Servidor FastAPI embebido para visualización y control
- **Sistema Autónomo**: Funcionamiento independiente sin requerimientos de red externa

### 1.3 Arquitectura del Sistema en Raspberry Pi

```
┌─────────────────────────────────────────────────────────────┐
│                    RASPBERRY PI 3 B+                        │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   Cámara    │  │  OpenCV     │  │ MediaPipe   │         │
│  │   USB/CSI   │  │  Captura    │  │  Pose       │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│         │                │                │                │
│         └────────────────┼────────────────┘                │
│                          │                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ TensorFlow  │  │  Análisis   │  │  Clasif.    │         │
│  │    Lite     │  │  Postural   │  │  Posturas   │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│         │                │                │                │
│         └────────────────┼────────────────┘                │
│                          │                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   SQLite    │  │   FastAPI   │  │   Redis     │         │
│  │  Database   │  │   Server    │  │   Cache     │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│         │                │                │                │
│         └────────────────┼────────────────┘                │
│                          │                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   Logging   │  │  Métricas   │  │  Timeline   │         │
│  │   Local     │  │  Posturales │  │  Events     │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
```

## 2. Componentes del Sistema

### 2.1 Entrada de Video
- **Cámara USB** o **Cámara CSI** de Raspberry Pi
- **OpenCV** para captura y procesamiento de frames
- **Resolución**: 640x480 (optimizada para rendimiento)

### 2.2 Detección Postural
- **MediaPipe Pose**: Detección de landmarks corporales
- **TensorFlow Lite**: Modelo personalizado para clasificación de 13 posturas
- **Análisis Geométrico**: Cálculo de ángulos de cuello y torso

### 2.3 Almacenamiento y Persistencia
- **SQLite**: Base de datos local para sesiones y métricas
- **Redis**: Cache en memoria para datos temporales
- **Sistema de Logging**: Archivos de registro locales

### 2.4 Interfaz y Control
- **FastAPI**: Servidor web local (puerto 8000)
- **WebSocket**: Comunicación en tiempo real
- **Endpoints REST**: API para gestión de sesiones

## 3. Instalación y Configuración

### 3.1 Requisitos del Sistema

#### Hardware
- **Raspberry Pi 3 B+** (recomendado) o superior
- **MicroSD** de 16GB o superior (clase 10)
- **Cámara USB** o **Cámara CSI** de Raspberry Pi
- **Fuente de alimentación** de 5V/2.5A mínimo

#### Software
- **Raspberry Pi OS** (Bullseye o superior)
- **Python 3.9+**
- **Git**

### 3.2 Configuración Inicial de Raspberry Pi

```bash
# Actualizar el sistema
sudo apt update && sudo apt upgrade -y

# Instalar dependencias del sistema
sudo apt install -y python3-pip python3-venv git cmake build-essential

# Habilitar cámara (si usa CSI)
sudo raspi-config
# Navigate to: Interface Options > Camera > Enable

# Configurar memoria de GPU (recomendado: 128MB)
sudo raspi-config
# Navigate to: Performance Options > GPU Memory > 128
```

### 3.3 Clonación y Configuración del Proyecto

```bash
# Crear directorio del proyecto
mkdir ~/shpd-rpi && cd ~/shpd-rpi

# Clonar el repositorio principal
git clone https://github.com/RodolGiaco/shpd.git

# Clonar el modelo personalizado
git clone https://github.com/RodolGiaco/shpd-model.git

# Crear entorno virtual
python3 -m venv venv
source venv/bin/activate

# Instalar dependencias
pip install --upgrade pip
pip install -r requirements.txt
```

### 3.4 Dependencias Específicas para Raspberry Pi

```bash
# Instalar dependencias adicionales para RPi
sudo apt install -y libhdf5-dev libhdf5-serial-dev libatlas-base-dev libjasper-dev libqtcore4 libqtgui4 libqt4-test

# Instalar OpenCV optimizado para RPi
pip install opencv-python-headless==4.8.1.78

# Instalar TensorFlow Lite
pip install tensorflow-lite==2.13.0

# Instalar MediaPipe
pip install mediapipe==0.10.3

# Otras dependencias
pip install fastapi uvicorn sqlalchemy redis websockets numpy
```

### 3.5 Configuración del Modelo TensorFlow Lite

```bash
# Copiar el modelo personalizado
cp shpd-model/posture_model.tflite ./models/
cp shpd-model/labels.txt ./models/

# Verificar la estructura de archivos
ls -la models/
# Debe mostrar:
# - posture_model.tflite
# - labels.txt
```

## 4. Estructura del Proyecto Adaptado

### 4.1 Organización de Archivos

```
shpd-rpi/
├── main_rpi.py              # Script principal de ejecución
├── posture_monitor.py       # Monitor postural con MediaPipe
├── tensorflow_classifier.py # Clasificador TensorFlow Lite
├── models/
│   ├── posture_model.tflite # Modelo personalizado
│   └── labels.txt          # Etiquetas de posturas
├── api/
│   ├── __init__.py
│   ├── database.py         # Configuración SQLite
│   ├── models.py           # Modelos de datos
│   └── routers/            # Endpoints de API
├── config/
│   ├── calibration.json    # Configuración de calibración
│   └── settings.py         # Configuración general
├── logs/                   # Archivos de registro
├── data/                   # Base de datos SQLite
└── requirements.txt        # Dependencias
```

### 4.2 Configuración de Base de Datos

```python
# api/database.py
import sqlite3
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Configuración para SQLite local
SQLALCHEMY_DATABASE_URL = "sqlite:///./data/shpd.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, 
    connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
```

### 4.3 Configuración de Redis Local

```python
# config/redis_config.py
import redis

# Configuración de Redis local
REDIS_CONFIG = {
    'host': 'localhost',
    'port': 6379,
    'db': 0,
    'decode_responses': True
}

# Inicializar conexión Redis
redis_client = redis.Redis(**REDIS_CONFIG)
```

## 5. Implementación del Clasificador TensorFlow Lite

### 5.1 Clase TensorFlowClassifier

```python
# tensorflow_classifier.py
import tensorflow as tf
import numpy as np
import cv2
from typing import Dict, List

class TensorFlowClassifier:
    def __init__(self, model_path: str, labels_path: str):
        """
        Inicializa el clasificador TensorFlow Lite para Raspberry Pi
        
        Args:
            model_path: Ruta al archivo .tflite
            labels_path: Ruta al archivo de etiquetas
        """
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        # Obtener detalles del modelo
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Cargar etiquetas
        with open(labels_path, 'r') as f:
            self.labels = [line.strip() for line in f.readlines()]
        
        # Configuración de entrada
        self.input_shape = self.input_details[0]['shape']
        self.input_height = self.input_shape[1]
        self.input_width = self.input_shape[2]
        
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocesa la imagen para el modelo TensorFlow Lite
        
        Args:
            image: Imagen de entrada (BGR)
            
        Returns:
            Imagen preprocesada
        """
        # Convertir BGR a RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Redimensionar
        resized_image = cv2.resize(rgb_image, (self.input_width, self.input_height))
        
        # Normalizar (0-1)
        normalized_image = resized_image.astype(np.float32) / 255.0
        
        # Agregar dimensión de batch
        input_image = np.expand_dims(normalized_image, axis=0)
        
        return input_image
    
    def classify_posture(self, image: np.ndarray) -> Dict[str, float]:
        """
        Clasifica la postura en la imagen
        
        Args:
            image: Imagen de entrada
            
        Returns:
            Diccionario con posturas y sus probabilidades (0-100)
        """
        # Preprocesar imagen
        input_image = self.preprocess_image(image)
        
        # Establecer tensor de entrada
        self.interpreter.set_tensor(self.input_details[0]['index'], input_image)
        
        # Ejecutar inferencia
        self.interpreter.invoke()
        
        # Obtener resultados
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        # Convertir a probabilidades
        probabilities = output_data[0]
        
        # Crear diccionario de resultados
        results = {}
        for i, label in enumerate(self.labels):
            # Convertir a porcentaje (0-100)
            percentage = float(probabilities[i] * 100)
            results[label] = percentage
        
        return results
```

### 5.2 Integración con PostureMonitor

```python
# Modificación en posture_monitor.py
from tensorflow_classifier import TensorFlowClassifier

class PostureMonitor:
    def __init__(self, session_id: str, *, save_metrics: bool = True):
        # ... código existente ...
        
        # Inicializar clasificador TensorFlow Lite
        model_path = "models/posture_model.tflite"
        labels_path = "models/labels.txt"
        self.classifier = TensorFlowClassifier(model_path, labels_path)
        
    def process_frame(self, image):
        # ... código existente de MediaPipe ...
        
        # Clasificación de postura con TensorFlow Lite
        if self.save_metrics and bad_time > self.args.time_threshold:
            if self.flag_alert:
                # Clasificar postura actual
                posture_results = self.classifier.classify_posture(image)
                
                # Guardar resultados en Redis
                self.save_posture_classification(posture_results)
                
                # ... resto del código de alerta ...
    
    def save_posture_classification(self, results: Dict[str, float]):
        """
        Guarda la clasificación de postura en Redis
        """
        key = f"posture_classification:{self.session_id}"
        self.redis_client.hset(key, mapping=results)
        self.redis_client.expire(key, 3600)  # Expirar en 1 hora
```

## 6. Script Principal de Ejecución

### 6.1 main_rpi.py

```python
#!/usr/bin/env python3
"""
Script principal para ejecutar SHPD en Raspberry Pi
"""

import os
import sys
import logging
import asyncio
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import redis
from contextlib import asynccontextmanager

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/shpd_rpi.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Configuración de Redis local
redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gestión del ciclo de vida de la aplicación
    """
    logger.info("🚀 Iniciando SHPD en Raspberry Pi...")
    
    # Verificar servicios
    try:
        redis_client.ping()
        logger.info("✅ Redis conectado")
    except Exception as e:
        logger.error(f"❌ Error conectando a Redis: {e}")
        sys.exit(1)
    
    # Verificar modelo TensorFlow Lite
    model_path = "models/posture_model.tflite"
    if not os.path.exists(model_path):
        logger.error(f"❌ Modelo no encontrado: {model_path}")
        sys.exit(1)
    logger.info("✅ Modelo TensorFlow Lite cargado")
    
    yield
    
    logger.info("🛑 Cerrando SHPD...")

# Crear aplicación FastAPI
app = FastAPI(
    title="SHPD Raspberry Pi",
    description="Smart Healthy Posture Detector para Raspberry Pi",
    version="1.0.0",
    lifespan=lifespan
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Importar routers
from api.routers import sesiones, pacientes, metricas, analysis

app.include_router(sesiones.router, prefix="/api/v1")
app.include_router(pacientes.router, prefix="/api/v1")
app.include_router(metricas.router, prefix="/api/v1")
app.include_router(analysis.router, prefix="/api/v1")

@app.get("/")
async def root():
    """
    Endpoint raíz
    """
    return {
        "message": "SHPD Raspberry Pi",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """
    Verificación de salud del sistema
    """
    try:
        # Verificar Redis
        redis_client.ping()
        redis_status = "healthy"
    except:
        redis_status = "unhealthy"
    
    # Verificar modelo
    model_path = "models/posture_model.tflite"
    model_status = "healthy" if os.path.exists(model_path) else "unhealthy"
    
    return {
        "status": "healthy" if redis_status == "healthy" and model_status == "healthy" else "unhealthy",
        "services": {
            "redis": redis_status,
            "model": model_status
        }
    }

def main():
    """
    Función principal
    """
    # Crear directorios necesarios
    os.makedirs("logs", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Configuración del servidor
    config = uvicorn.Config(
        app=app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True
    )
    
    # Iniciar servidor
    server = uvicorn.Server(config)
    logger.info("🌐 Servidor iniciado en http://0.0.0.0:8000")
    server.run()

if __name__ == "__main__":
    main()
```

## 7. Configuración de Calibración

### 7.1 Archivo de Calibración

```json
{
    "offset_threshold": 100,
    "neck_angle_threshold": 25,
    "torso_angle_threshold": 10,
    "time_threshold": 10,
    "camera_settings": {
        "width": 640,
        "height": 480,
        "fps": 15
    },
    "model_settings": {
        "confidence_threshold": 0.7,
        "max_detections": 1
    }
}
```

### 7.2 Script de Calibración

```python
# calibration_tool.py
import cv2
import json
import argparse
from posture_monitor import PostureMonitor

def run_calibration():
    """
    Herramienta de calibración para ajustar umbrales
    """
    print("🔧 Iniciando calibración de SHPD...")
    
    # Crear monitor en modo calibración
    monitor = PostureMonitor("calibration", save_metrics=False)
    
    # Ejecutar calibración
    monitor.run()
    
    # Guardar configuración
    config = {
        "offset_threshold": monitor.args.offset_threshold,
        "neck_angle_threshold": monitor.args.neck_angle_threshold,
        "torso_angle_threshold": monitor.args.torso_angle_threshold,
        "time_threshold": monitor.args.time_threshold
    }
    
    with open("config/calibration.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("✅ Calibración completada")

if __name__ == "__main__":
    run_calibration()
```

## 8. Ejecución del Sistema

### 8.1 Inicio del Sistema

```bash
# Activar entorno virtual
source venv/bin/activate

# Verificar servicios
sudo systemctl start redis-server
sudo systemctl enable redis-server

# Ejecutar sistema principal
python main_rpi.py
```

### 8.2 Verificación del Sistema

```bash
# Verificar estado del sistema
curl http://localhost:8000/health

# Verificar logs
tail -f logs/shpd_rpi.log

# Verificar base de datos
sqlite3 data/shpd.db ".tables"
```

### 8.3 Acceso a la Interfaz Web

- **URL**: `http://[IP_RASPBERRY_PI]:8000`
- **Documentación API**: `http://[IP_RASPBERRY_PI]:8000/docs`
- **Interfaz de monitoreo**: `http://[IP_RASPBERRY_PI]:8000/monitor`

## 9. Optimizaciones para Raspberry Pi

### 9.1 Configuración de Rendimiento

```bash
# Configurar overclock (opcional)
sudo raspi-config
# Navigate to: Performance Options > Overclock > Medium

# Configurar memoria de GPU
sudo raspi-config
# Navigate to: Performance Options > GPU Memory > 128

# Deshabilitar servicios innecesarios
sudo systemctl disable bluetooth
sudo systemctl disable hciuart
```

### 9.2 Optimizaciones de Python

```python
# config/performance.py
import os

# Configuraciones de rendimiento
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reducir logs de TensorFlow
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'  # Optimizar OpenCV

# Configuración de threads
import threading
threading.stack_size(8*1024*1024)  # 8MB stack size
```

### 9.3 Configuración de Cámara

```python
# config/camera.py
import cv2

def get_optimal_camera_settings():
    """
    Obtiene configuraciones óptimas de cámara para RPi
    """
    return {
        'width': 640,
        'height': 480,
        'fps': 15,
        'buffer_size': 1,
        'fourcc': cv2.VideoWriter_fourcc(*'MJPG')
    }

def configure_camera(cap):
    """
    Configura la cámara con parámetros optimizados
    """
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    return cap
```

## 10. Monitoreo y Mantenimiento

### 10.1 Logs del Sistema

```bash
# Ver logs en tiempo real
tail -f logs/shpd_rpi.log

# Ver logs de errores
grep "ERROR" logs/shpd_rpi.log

# Ver logs de rendimiento
grep "FPS" logs/shpd_rpi.log
```

### 10.2 Métricas de Rendimiento

```python
# monitoring/performance.py
import psutil
import time
import logging

class PerformanceMonitor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def get_system_metrics(self):
        """
        Obtiene métricas del sistema
        """
        return {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'temperature': self.get_cpu_temperature(),
            'disk_usage': psutil.disk_usage('/').percent
        }
    
    def get_cpu_temperature(self):
        """
        Obtiene temperatura de la CPU
        """
        try:
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                temp = float(f.read()) / 1000.0
            return temp
        except:
            return None
    
    def log_performance(self):
        """
        Registra métricas de rendimiento
        """
        metrics = self.get_system_metrics()
        self.logger.info(f"Performance: CPU={metrics['cpu_percent']}% "
                        f"RAM={metrics['memory_percent']}% "
                        f"TEMP={metrics['temperature']}°C")
```

### 10.3 Limpieza de Datos

```bash
# Script de limpieza automática
#!/bin/bash
# cleanup.sh

# Limpiar logs antiguos (más de 7 días)
find logs/ -name "*.log" -mtime +7 -delete

# Limpiar datos temporales de Redis
redis-cli FLUSHDB

# Comprimir base de datos
sqlite3 data/shpd.db "VACUUM;"
```

## 11. Solución de Problemas

### 11.1 Problemas Comunes

#### Error de Cámara
```bash
# Verificar permisos de cámara
sudo usermod -a -G video $USER

# Verificar dispositivos de video
ls -la /dev/video*

# Reiniciar servicios de cámara
sudo systemctl restart camera.service
```

#### Error de Memoria
```bash
# Verificar uso de memoria
free -h

# Limpiar caché
sudo sh -c "echo 3 > /proc/sys/vm/drop_caches"

# Aumentar swap
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile
# Cambiar CONF_SWAPSIZE=100 a CONF_SWAPSIZE=512
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

#### Error de TensorFlow Lite
```bash
# Verificar instalación
python3 -c "import tensorflow.lite as tflite; print('OK')"

# Reinstalar TensorFlow Lite
pip uninstall tensorflow-lite
pip install tensorflow-lite==2.13.0
```

### 11.2 Diagnóstico del Sistema

```bash
# Verificar estado general
./diagnostic.sh

# Verificar conectividad de red
ping -c 4 8.8.8.8

# Verificar servicios
sudo systemctl status redis-server
sudo systemctl status camera.service
```

## 12. Conclusiones

### 12.1 Ventajas de la Adaptación

1. **Autonomía Total**: Sistema independiente sin dependencias externas
2. **Bajo Costo**: Utilización de hardware económico y accesible
3. **Privacidad**: Procesamiento local sin envío de datos externos
4. **Escalabilidad**: Fácil replicación en múltiples dispositivos
5. **Personalización**: Modelo entrenado específicamente para el dominio

### 12.2 Limitaciones y Consideraciones

1. **Rendimiento**: Limitado por las capacidades del hardware
2. **Precisión**: Modelo local vs. modelo cloud de OpenAI
3. **Mantenimiento**: Requiere gestión local de actualizaciones
4. **Almacenamiento**: Limitado por la capacidad de la microSD

### 12.3 Futuras Mejoras

1. **Optimización de Modelo**: Cuantización y optimización específica para RPi
2. **Interfaz Móvil**: Aplicación móvil para control remoto
3. **Análisis Avanzado**: Integración de análisis temporal y tendencias
4. **Notificaciones**: Sistema de alertas por email/SMS
5. **Backup Automático**: Sincronización con servicios cloud

---

**Nota**: Esta documentación está diseñada para ser parte de una tesis de ingeniería electrónica, proporcionando una guía completa para la implementación y operación del sistema SHPD en Raspberry Pi. El sistema resultante es completamente autónomo y puede funcionar sin conexión a internet, procesando todas las tareas de detección postural localmente en el dispositivo embebido.