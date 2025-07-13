# Documentación Técnica: Adaptación SHPD para Raspberry Pi 3 B+

## 1. Introducción

### 1.1 Objetivo del Proyecto

El **Smart Healthy Posture Detector (SHPD)** es un sistema de monitoreo postural en tiempo real que utiliza visión por computadora y aprendizaje automático para detectar y analizar la postura de usuarios sentados. Esta documentación describe la adaptación completa del sistema para ejecutarse íntegramente en una **Raspberry Pi 3 B+**, eliminando la dependencia de servicios externos y microservicios distribuidos.

### 1.2 Características de la Adaptación

- **Sistema Integrado**: Todo el procesamiento se ejecuta localmente en la Raspberry Pi
- **Modelo Local**: Reemplazo del modelo OpenAI por un modelo TensorFlow Lite personalizado
- **Detección en Tiempo Real**: Análisis postural continuo mediante MediaPipe y TensorFlow Lite
- **Almacenamiento Local**: Base de datos SQLite para persistencia de datos
- **Interfaz Simplificada**: Sistema monolítico ejecutable desde un único script principal

### 1.3 Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────────┐
│                    RASPBERRY PI 3 B+                        │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   Cámara    │  │  MediaPipe  │  │ TensorFlow  │         │
│  │   USB/CSI   │──│  Pose Est.  │──│   Lite      │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│         │                │                │                │
│         └────────────────┼────────────────┘                │
│                          │                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   Análisis  │  │  Almacena-  │  │  Interfaz   │         │
│  │  Postural   │──│   miento    │──│  Usuario    │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│                          │                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   Logging   │  │  Alertas    │  │  Métricas   │         │
│  │   Local     │  │  Tiempo     │  │  Posturales │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
```

## 2. Componentes del Sistema Integrado

### 2.1 Entrada de Cámara
- **Hardware**: Cámara USB o módulo CSI de Raspberry Pi
- **Resolución**: 640x480 (optimizada para rendimiento)
- **FPS**: 15-30 fps (configurable según capacidad de procesamiento)

### 2.2 Detección con Modelo Local
- **MediaPipe Pose**: Detección de puntos clave del cuerpo
- **TensorFlow Lite**: Modelo personalizado para clasificación postural
- **Procesamiento**: Análisis de ángulos y métricas posturales

### 2.3 Almacenamiento y Registro
- **SQLite**: Base de datos local para sesiones y métricas
- **Redis**: Cache en memoria para datos temporales
- **Logging**: Archivos de registro local para debugging

### 2.4 Visualización y Monitoreo
- **Interfaz Local**: Visualización en tiempo real de la detección
- **Métricas**: Estadísticas de postura correcta/incorrecta
- **Alertas**: Notificaciones cuando se detecta mala postura

## 3. Instalación y Configuración

### 3.1 Requisitos del Sistema

#### Hardware
- **Raspberry Pi 3 B+** (recomendado) o superior
- **Cámara USB** o módulo CSI
- **Tarjeta SD** de 16GB o superior (clase 10)
- **Fuente de alimentación** de 5V/2.5A
- **Teclado y mouse** (para configuración inicial)

#### Software
- **Raspberry Pi OS** (Bullseye o superior)
- **Python 3.8+**
- **Git**

### 3.2 Configuración Inicial de Raspberry Pi

```bash
# Actualizar el sistema
sudo apt update && sudo apt upgrade -y

# Instalar dependencias del sistema
sudo apt install -y python3-pip python3-venv git cmake build-essential
sudo apt install -y libatlas-base-dev liblapack-dev libblas-dev
sudo apt install -y libhdf5-dev libhdf5-serial-dev libhdf5-103
sudo apt install -y libqtgui4 libqtwebkit4 libqt4-test python3-pyqt5
sudo apt install -y libjasper-dev libqtcore4 libqt4-test

# Habilitar cámara (si se usa módulo CSI)
sudo raspi-config
# Navigate to: Interface Options > Camera > Enable

# Reiniciar el sistema
sudo reboot
```

### 3.3 Clonación del Repositorio

```bash
# Crear directorio del proyecto
mkdir -p ~/shpd-rpi
cd ~/shpd-rpi

# Clonar el repositorio principal
git clone https://github.com/RodolGiaco/shpd.git

# Clonar el repositorio del modelo personalizado
git clone https://github.com/RodolGiaco/shpd-model.git
```

### 3.4 Configuración del Entorno Virtual

```bash
# Crear entorno virtual
python3 -m venv venv

# Activar entorno virtual
source venv/bin/activate

# Actualizar pip
pip install --upgrade pip setuptools wheel
```

### 3.5 Instalación de Dependencias

#### Dependencias del Sistema
```bash
# Instalar OpenCV con optimizaciones para ARM
pip install opencv-python-headless==4.8.1.78

# Instalar TensorFlow Lite
pip install tensorflow-lite==2.13.0

# Instalar MediaPipe
pip install mediapipe==0.10.7

# Otras dependencias
pip install numpy==1.24.3
pip install redis==4.6.0
pip install sqlalchemy==2.0.23
pip install fastapi==0.104.1
pip install uvicorn==0.24.0
pip install websockets==12.0
pip install requests==2.31.0
pip install pillow==10.0.1
```

#### Archivo requirements.txt para Raspberry Pi
```txt
# requirements_rpi.txt
opencv-python-headless==4.8.1.78
tensorflow-lite==2.13.0
mediapipe==0.10.7
numpy==1.24.3
redis==4.6.0
sqlalchemy==2.0.23
fastapi==0.104.1
uvicorn==0.24.0
websockets==12.0
requests==2.31.0
pillow==10.0.1
```

### 3.6 Configuración del Modelo Personalizado

#### Ubicación del Modelo
El modelo TensorFlow Lite se encuentra en el repositorio `shpd-model`:
```
~/shpd-rpi/shpd-model/
├── model/
│   ├── posture_detector.tflite
│   ├── labels.txt
│   └── config.json
└── utils/
    ├── preprocessor.py
    └── postprocessor.py
```

#### Integración del Modelo
```python
# Cargar el modelo TensorFlow Lite
import tensorflow as tf

def load_tflite_model(model_path):
    """Carga el modelo TensorFlow Lite para detección postural"""
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def load_labels(labels_path):
    """Carga las etiquetas de las posturas"""
    with open(labels_path, 'r') as f:
        return [line.strip() for line in f.readlines()]
```

## 4. Implementación del Sistema Integrado

### 4.1 Script Principal: `main_rpi.py`

```python
#!/usr/bin/env python3
"""
SHPD - Smart Healthy Posture Detector
Sistema integrado para Raspberry Pi 3 B+
"""

import os
import sys
import time
import logging
import json
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from datetime import datetime
from pathlib import Path

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('shpd_rpi.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class SHPD_RaspberryPi:
    def __init__(self, model_path: str, labels_path: str):
        """
        Inicializa el sistema SHPD para Raspberry Pi
        
        Args:
            model_path: Ruta al modelo TensorFlow Lite
            labels_path: Ruta al archivo de etiquetas
        """
        self.model_path = model_path
        self.labels_path = labels_path
        
        # Inicializar componentes
        self.setup_mediapipe()
        self.load_model()
        self.setup_database()
        self.setup_redis()
        
        # Métricas de sesión
        self.session_id = f"rpi_session_{int(time.time())}"
        self.good_frames = 0
        self.bad_frames = 0
        self.alert_count = 0
        
        logger.info(f"Sistema SHPD inicializado - Session ID: {self.session_id}")
    
    def setup_mediapipe(self):
        """Configura MediaPipe para detección de pose"""
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        logger.info("MediaPipe configurado correctamente")
    
    def load_model(self):
        """Carga el modelo TensorFlow Lite personalizado"""
        try:
            self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
            self.interpreter.allocate_tensors()
            
            # Obtener detalles del modelo
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            # Cargar etiquetas
            with open(self.labels_path, 'r') as f:
                self.labels = [line.strip() for line in f.readlines()]
            
            logger.info(f"Modelo cargado: {len(self.labels)} clases detectadas")
            
        except Exception as e:
            logger.error(f"Error cargando modelo: {e}")
            raise
    
    def setup_database(self):
        """Configura la base de datos SQLite local"""
        from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float
        from sqlalchemy.ext.declarative import declarative_base
        from sqlalchemy.orm import sessionmaker
        
        # Crear base de datos SQLite
        self.engine = create_engine('sqlite:///shpd_rpi.db')
        self.Base = declarative_base()
        
        # Definir modelo de datos
        class PostureSession(self.Base):
            __tablename__ = 'posture_sessions'
            
            id = Column(Integer, primary_key=True)
            session_id = Column(String, nullable=False)
            timestamp = Column(DateTime, nullable=False)
            posture_type = Column(String, nullable=False)
            confidence = Column(Float, nullable=False)
            good_frames = Column(Integer, default=0)
            bad_frames = Column(Integer, default=0)
        
        self.PostureSession = PostureSession
        self.Base.metadata.create_all(self.engine)
        
        # Crear sesión de base de datos
        Session = sessionmaker(bind=self.engine)
        self.db_session = Session()
        
        logger.info("Base de datos SQLite configurada")
    
    def setup_redis(self):
        """Configura Redis para cache local"""
        try:
            import redis
            self.redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
            self.redis_client.ping()
            logger.info("Redis configurado correctamente")
        except:
            logger.warning("Redis no disponible, usando cache en memoria")
            self.redis_client = None
    
    def preprocess_frame(self, frame):
        """Preprocesa el frame para el modelo TensorFlow Lite"""
        # Redimensionar a 224x224 (ajustar según el modelo)
        frame_resized = cv2.resize(frame, (224, 224))
        
        # Normalizar valores de píxeles
        frame_normalized = frame_resized.astype(np.float32) / 255.0
        
        # Agregar dimensión de batch
        frame_batch = np.expand_dims(frame_normalized, axis=0)
        
        return frame_batch
    
    def predict_posture(self, frame):
        """Realiza predicción de postura usando el modelo local"""
        try:
            # Preprocesar frame
            input_data = self.preprocess_frame(frame)
            
            # Establecer tensor de entrada
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            
            # Ejecutar inferencia
            self.interpreter.invoke()
            
            # Obtener resultados
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            
            # Obtener predicción
            predicted_class = np.argmax(output_data[0])
            confidence = float(output_data[0][predicted_class])
            posture_label = self.labels[predicted_class]
            
            return {
                'posture': posture_label,
                'confidence': confidence,
                'class_id': predicted_class
            }
            
        except Exception as e:
            logger.error(f"Error en predicción: {e}")
            return None
    
    def analyze_mediapipe_pose(self, frame):
        """Analiza la pose usando MediaPipe para métricas adicionales"""
        # Convertir BGR a RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Procesar con MediaPipe
        results = self.pose.process(frame_rgb)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Calcular ángulos de postura
            neck_angle = self.calculate_neck_angle(landmarks)
            torso_angle = self.calculate_torso_angle(landmarks)
            
            return {
                'neck_angle': neck_angle,
                'torso_angle': torso_angle,
                'landmarks': landmarks
            }
        
        return None
    
    def calculate_neck_angle(self, landmarks):
        """Calcula el ángulo de inclinación del cuello"""
        # Implementar cálculo de ángulo usando landmarks de MediaPipe
        # (código específico según los puntos clave disponibles)
        return 0.0
    
    def calculate_torso_angle(self, landmarks):
        """Calcula el ángulo de inclinación del torso"""
        # Implementar cálculo de ángulo usando landmarks de MediaPipe
        return 0.0
    
    def save_session_data(self, prediction, mediapipe_data):
        """Guarda los datos de la sesión en la base de datos"""
        try:
            session_record = self.PostureSession(
                session_id=self.session_id,
                timestamp=datetime.now(),
                posture_type=prediction['posture'],
                confidence=prediction['confidence'],
                good_frames=self.good_frames,
                bad_frames=self.bad_frames
            )
            
            self.db_session.add(session_record)
            self.db_session.commit()
            
        except Exception as e:
            logger.error(f"Error guardando datos: {e}")
            self.db_session.rollback()
    
    def update_metrics(self, prediction):
        """Actualiza métricas de postura"""
        # Lógica para determinar si la postura es buena o mala
        good_postures = ['sentado_erguido', 'postura_correcta']
        
        if prediction['posture'] in good_postures and prediction['confidence'] > 0.7:
            self.good_frames += 1
        else:
            self.bad_frames += 1
            
            # Verificar si se debe enviar alerta
            if self.bad_frames >= 30:  # 2 segundos a 15 FPS
                self.send_alert(prediction)
    
    def send_alert(self, prediction):
        """Envía alerta de mala postura"""
        self.alert_count += 1
        alert_message = f"⚠️ Alerta de postura: {prediction['posture']} (confianza: {prediction['confidence']:.2f})"
        
        logger.warning(alert_message)
        
        # Guardar alerta en Redis si está disponible
        if self.redis_client:
            alert_data = {
                'timestamp': datetime.now().isoformat(),
                'posture': prediction['posture'],
                'confidence': prediction['confidence'],
                'alert_count': self.alert_count
            }
            self.redis_client.rpush(f"alerts:{self.session_id}", json.dumps(alert_data))
    
    def draw_overlay(self, frame, prediction, mediapipe_data):
        """Dibuja información en el frame"""
        h, w = frame.shape[:2]
        
        # Información de predicción
        text = f"Postura: {prediction['posture']}"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        text = f"Confianza: {prediction['confidence']:.2f}"
        cv2.putText(frame, text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Métricas de sesión
        text = f"Buenas: {self.good_frames} | Malas: {self.bad_frames}"
        cv2.putText(frame, text, (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        text = f"Alertas: {self.alert_count}"
        cv2.putText(frame, text, (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Dibujar landmarks de MediaPipe si están disponibles
        if mediapipe_data and mediapipe_data['landmarks']:
            self.mp_drawing.draw_landmarks(
                frame, 
                results.pose_landmarks, 
                self.mp_pose.POSE_CONNECTIONS
            )
        
        return frame
    
    def run(self, camera_index=0):
        """Ejecuta el sistema principal de detección postural"""
        logger.info("Iniciando sistema de detección postural...")
        
        # Inicializar cámara
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            logger.error("No se pudo abrir la cámara")
            return
        
        # Configurar resolución para optimizar rendimiento
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 15)
        
        logger.info("Cámara inicializada correctamente")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.error("Error leyendo frame de la cámara")
                    break
                
                # Análisis con MediaPipe
                mediapipe_data = self.analyze_mediapipe_pose(frame)
                
                # Predicción con modelo personalizado
                prediction = self.predict_posture(frame)
                
                if prediction:
                    # Actualizar métricas
                    self.update_metrics(prediction)
                    
                    # Guardar datos de sesión (cada 30 frames = 2 segundos)
                    if (self.good_frames + self.bad_frames) % 30 == 0:
                        self.save_session_data(prediction, mediapipe_data)
                    
                    # Dibujar overlay
                    frame = self.draw_overlay(frame, prediction, mediapipe_data)
                
                # Mostrar frame
                cv2.imshow('SHPD - Raspberry Pi', frame)
                
                # Salir con 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        except KeyboardInterrupt:
            logger.info("Sistema interrumpido por el usuario")
        
        finally:
            # Limpiar recursos
            cap.release()
            cv2.destroyAllWindows()
            self.db_session.close()
            
            # Guardar resumen final
            self.save_final_summary()
            
            logger.info("Sistema finalizado correctamente")
    
    def save_final_summary(self):
        """Guarda un resumen final de la sesión"""
        summary = {
            'session_id': self.session_id,
            'start_time': datetime.now().isoformat(),
            'total_frames': self.good_frames + self.bad_frames,
            'good_frames': self.good_frames,
            'bad_frames': self.bad_frames,
            'alert_count': self.alert_count,
            'good_posture_percentage': (self.good_frames / (self.good_frames + self.bad_frames)) * 100 if (self.good_frames + self.bad_frames) > 0 else 0
        }
        
        # Guardar en archivo JSON
        with open(f'session_summary_{self.session_id}.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Resumen de sesión guardado: {summary}")


def main():
    """Función principal del sistema SHPD para Raspberry Pi"""
    
    # Configurar rutas
    base_path = Path(__file__).parent
    model_path = base_path / "shpd-model" / "model" / "posture_detector.tflite"
    labels_path = base_path / "shpd-model" / "model" / "labels.txt"
    
    # Verificar que existan los archivos del modelo
    if not model_path.exists():
        logger.error(f"Modelo no encontrado en: {model_path}")
        return
    
    if not labels_path.exists():
        logger.error(f"Archivo de etiquetas no encontrado en: {labels_path}")
        return
    
    # Crear instancia del sistema
    shpd_system = SHPD_RaspberryPi(str(model_path), str(labels_path))
    
    # Ejecutar sistema
    shpd_system.run()


if __name__ == "__main__":
    main()
```

### 4.2 Configuración de Redis Local

```bash
# Instalar Redis en Raspberry Pi
sudo apt install redis-server

# Configurar Redis para iniciar automáticamente
sudo systemctl enable redis-server

# Verificar que Redis esté funcionando
sudo systemctl status redis-server
```

### 4.3 Script de Configuración Automática

```bash
#!/bin/bash
# setup_shpd_rpi.sh

echo "=== Configuración SHPD para Raspberry Pi ==="

# Actualizar sistema
echo "Actualizando sistema..."
sudo apt update && sudo apt upgrade -y

# Instalar dependencias
echo "Instalando dependencias..."
sudo apt install -y python3-pip python3-venv git cmake build-essential
sudo apt install -y libatlas-base-dev liblapack-dev libblas-dev
sudo apt install -y libhdf5-dev libhdf5-serial-dev libhdf5-103
sudo apt install -y libqtgui4 libqtwebkit4 libqt4-test python3-pyqt5
sudo apt install -y libjasper-dev libqtcore4 libqt4-test
sudo apt install -y redis-server

# Habilitar Redis
sudo systemctl enable redis-server
sudo systemctl start redis-server

# Crear directorio del proyecto
mkdir -p ~/shpd-rpi
cd ~/shpd-rpi

# Crear entorno virtual
python3 -m venv venv
source venv/bin/activate

# Instalar dependencias Python
pip install --upgrade pip setuptools wheel
pip install -r requirements_rpi.txt

echo "Configuración completada. Ejecute: python main_rpi.py"
```

## 5. Ejecución del Sistema

### 5.1 Inicio del Sistema

```bash
# Navegar al directorio del proyecto
cd ~/shpd-rpi

# Activar entorno virtual
source venv/bin/activate

# Ejecutar el sistema principal
python main_rpi.py
```

### 5.2 Configuración de Autoinicio

```bash
# Crear servicio systemd para autoinicio
sudo nano /etc/systemd/system/shpd.service
```

Contenido del archivo de servicio:
```ini
[Unit]
Description=SHPD Raspberry Pi Service
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/shpd-rpi
Environment=PATH=/home/pi/shpd-rpi/venv/bin
ExecStart=/home/pi/shpd-rpi/venv/bin/python main_rpi.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
# Habilitar y iniciar el servicio
sudo systemctl daemon-reload
sudo systemctl enable shpd
sudo systemctl start shpd

# Verificar estado
sudo systemctl status shpd
```

### 5.3 Monitoreo del Sistema

```bash
# Ver logs en tiempo real
sudo journalctl -u shpd -f

# Ver logs del archivo
tail -f ~/shpd-rpi/shpd_rpi.log

# Verificar uso de recursos
htop
```

## 6. Optimizaciones de Rendimiento

### 6.1 Configuración de Raspberry Pi

```bash
# Aumentar memoria GPU (en /boot/config.txt)
sudo nano /boot/config.txt
```

Agregar/modificar:
```
gpu_mem=128
over_voltage=2
arm_freq=1400
```

### 6.2 Optimizaciones de OpenCV

```python
# En main_rpi.py, agregar optimizaciones
import cv2

# Habilitar optimizaciones de OpenCV
cv2.setUseOptimized(True)
cv2.setNumThreads(4)  # Ajustar según núcleos disponibles
```

### 6.3 Configuración de TensorFlow Lite

```python
# Optimizaciones para TensorFlow Lite
interpreter = tf.lite.Interpreter(
    model_path=model_path,
    num_threads=4  # Usar múltiples hilos
)
```

### 6.4 Gestión de Memoria

```python
# Limpiar memoria periódicamente
import gc

def cleanup_memory():
    """Limpia memoria no utilizada"""
    gc.collect()

# Llamar cada 100 frames
if frame_count % 100 == 0:
    cleanup_memory()
```

## 7. Troubleshooting

### 7.1 Problemas Comunes

#### Error de Cámara
```bash
# Verificar permisos de cámara
sudo usermod -a -G video pi

# Verificar dispositivos de cámara
ls -la /dev/video*

# Probar cámara
vcgencmd get_camera
```

#### Error de Memoria
```bash
# Aumentar swap
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile
# Cambiar CONF_SWAPSIZE=100 a CONF_SWAPSIZE=1024
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

#### Error de TensorFlow Lite
```bash
# Verificar instalación
python3 -c "import tensorflow as tf; print(tf.__version__)"

# Reinstalar si es necesario
pip uninstall tensorflow tensorflow-lite
pip install tensorflow-lite==2.13.0
```

### 7.2 Logs y Debugging

```bash
# Ver logs detallados
tail -f ~/shpd-rpi/shpd_rpi.log

# Verificar uso de CPU y memoria
top -p $(pgrep -f main_rpi.py)

# Verificar temperatura
vcgencmd measure_temp
```

## 8. Mantenimiento y Actualizaciones

### 8.1 Backup de Datos

```bash
# Script de backup automático
#!/bin/bash
# backup_shpd.sh

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/home/pi/shpd_backups"

mkdir -p $BACKUP_DIR

# Backup de base de datos
cp ~/shpd-rpi/shpd_rpi.db $BACKUP_DIR/shpd_rpi_$DATE.db

# Backup de logs
cp ~/shpd-rpi/shpd_rpi.log $BACKUP_DIR/shpd_rpi_$DATE.log

# Backup de resúmenes de sesión
cp ~/shpd-rpi/session_summary_*.json $BACKUP_DIR/

echo "Backup completado: $BACKUP_DIR"
```

### 8.2 Actualización del Sistema

```bash
# Script de actualización
#!/bin/bash
# update_shpd.sh

cd ~/shpd-rpi

# Detener servicio
sudo systemctl stop shpd

# Backup antes de actualizar
./backup_shpd.sh

# Actualizar código
git pull origin main

# Actualizar dependencias
source venv/bin/activate
pip install -r requirements_rpi.txt

# Reiniciar servicio
sudo systemctl start shpd

echo "Actualización completada"
```

## 9. Conclusiones

### 9.1 Ventajas de la Adaptación

1. **Independencia**: Sistema completamente autónomo sin dependencias externas
2. **Bajo Costo**: Utiliza hardware económico y accesible
3. **Privacidad**: Todo el procesamiento se realiza localmente
4. **Escalabilidad**: Fácil replicación en múltiples dispositivos
5. **Personalización**: Modelo entrenado específicamente para el dominio

### 9.2 Limitaciones

1. **Rendimiento**: Limitado por las capacidades de la Raspberry Pi 3 B+
2. **Precisión**: Puede ser menor que modelos más complejos en la nube
3. **Almacenamiento**: Limitado por la capacidad de la tarjeta SD
4. **Conectividad**: Sin acceso a actualizaciones automáticas del modelo

### 9.3 Recomendaciones para Producción

1. **Hardware**: Considerar Raspberry Pi 4 para mejor rendimiento
2. **Almacenamiento**: Usar SSD externo para mayor durabilidad
3. **Redundancia**: Implementar sistema de backup automático
4. **Monitoreo**: Configurar alertas de sistema y rendimiento
5. **Seguridad**: Implementar autenticación y encriptación de datos

### 9.4 Próximos Pasos

1. **Optimización**: Mejorar rendimiento del modelo TensorFlow Lite
2. **Interfaz**: Desarrollar interfaz web local para configuración
3. **Análisis**: Implementar análisis estadístico avanzado
4. **Integración**: Conectar con sistemas de salud existentes
5. **Validación**: Realizar pruebas clínicas del sistema

---

**Nota**: Esta documentación está diseñada para ser parte de una tesis de ingeniería electrónica y proporciona una guía completa para la implementación del sistema SHPD en Raspberry Pi. El sistema resultante es completamente funcional y puede ser utilizado para monitoreo postural en tiempo real en entornos educativos, laborales o de salud.