# SHPD Backend - Smart Healthy Posture Detector

<div align="center">

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-009688.svg)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?logo=docker&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-316192?logo=postgresql&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

**Sistema inteligente de análisis postural en tiempo real para la prevención de lesiones ergonómicas**

</div>

## 📋 Tabla de Contenidos

- [Resumen del Sistema](#resumen-del-sistema)
- [Arquitectura](#arquitectura)
- [Tecnologías Utilizadas](#tecnologías-utilizadas)
- [Requisitos del Sistema](#requisitos-del-sistema)
- [Instalación](#instalación)
- [Configuración](#configuración)
- [Uso de la API](#uso-de-la-api)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Contribución](#contribución)
- [Licencia](#licencia)
- [Autor y Créditos](#autor-y-créditos)

## 🎯 Resumen del Sistema

El backend de SHPD es el núcleo de procesamiento de un sistema integral de detección y análisis de posturas corporales en tiempo real. Desarrollado como parte de una tesis de grado en Ingeniería Electrónica, este componente gestiona:

### Funcionalidades Principales

- **📹 Procesamiento de Video en Tiempo Real**: Recepción y análisis de streams de video mediante WebSocket
- **🏃 Detección de Posturas**: Identificación de puntos clave corporales usando MediaPipe
- **🚨 Sistema de Alertas**: Generación inteligente de notificaciones ante posturas incorrectas sostenidas
- **🤖 Análisis con IA**: Clasificación avanzada de posturas mediante integración con OpenAI Vision API
- **💾 Persistencia de Datos**: Almacenamiento de sesiones, métricas y estadísticas posturales
- **📱 Integración Multi-plataforma**: API REST para frontend web y bot de Telegram
- **⚡ Comunicación Bidireccional**: WebSocket para retroalimentación en tiempo real

## 🏗️ Arquitectura

### Diagrama del Sistema

```mermaid
graph TB
    subgraph "Clientes"
        A[Cliente Web]
        B[Bot Telegram]
        C[Dispositivo IoT]
    end
    
    subgraph "Backend SHPD"
        D[FastAPI Server]
        E[WebSocket Handler]
        F[REST API]
        G[PostureMonitor<br/>MediaPipe]
        H[Async Workers<br/>OpenAI API]
        I[Redis Cache]
        J[PostgreSQL DB]
    end
    
    A -->|WebSocket| E
    B -->|HTTP/REST| F
    C -->|WebSocket| E
    
    E --> G
    F --> D
    D --> G
    D --> H
    G --> I
    H --> I
    I --> J
    
    style D fill:#2E7D32
    style G fill:#1976D2
    style H fill:#7B1FA2
```

### Componentes Principales

1. **Servidor FastAPI**: Core asíncrono que orquesta todos los servicios
2. **PostureMonitor**: Motor de análisis postural basado en MediaPipe
3. **Workers Asíncronos**: Procesamiento paralelo de análisis con IA
4. **Capa de Datos**: PostgreSQL para persistencia y Redis para caché
5. **API Gateway**: Endpoints REST y WebSocket para comunicación externa

## 🛠️ Tecnologías Utilizadas

### Stack Principal

| Tecnología | Versión | Uso |
|------------|---------|-----|
| **Python** | 3.8+ | Lenguaje principal |
| **FastAPI** | 0.104.1 | Framework web asíncrono |
| **PostgreSQL** | 13+ | Base de datos relacional |
| **Redis** | 6+ | Cache y almacenamiento temporal |
| **Docker** | 20.10+ | Containerización |

### Librerías de Procesamiento

- **OpenCV**: Procesamiento de imágenes y video
- **MediaPipe**: Detección de landmarks corporales
- **NumPy**: Cálculos numéricos y matrices
- **WebSockets**: Comunicación en tiempo real
- **SQLAlchemy**: ORM para gestión de base de datos
- **OpenAI API**: Análisis avanzado con modelos de visión

## 💻 Requisitos del Sistema

### Hardware Mínimo
- **CPU**: 2 cores (4 recomendado)
- **RAM**: 4GB (8GB recomendado)
- **Almacenamiento**: 10GB disponibles
- **GPU**: Opcional (mejora el rendimiento de MediaPipe)

### Software
- Docker 20.10+
- Docker Compose 2.0+
- Git 2.25+

### Compatibilidad
- ✅ Linux (Ubuntu 20.04+, Debian 10+)
- ✅ macOS (10.15+)
- ✅ Windows (WSL2)
- ✅ Raspberry Pi 4 (ARM64)

## 🚀 Instalación

### 1. Clonar el Repositorio

```bash
git clone https://github.com/tu-usuario/shpd-backend.git
cd shpd-backend
```

### 2. Configuración con Docker (Recomendado)

```bash
# Copiar archivo de configuración de ejemplo
cp .env.example .env

# Editar variables de entorno
nano .env

# Construir e iniciar los servicios
docker-compose up -d
```

### 3. Instalación Manual

```bash
# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/macOS
# o
venv\Scripts\activate  # Windows

# Instalar dependencias
pip install -r requirements.txt

# Inicializar base de datos
python -m api.database init

# Ejecutar servidor
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## ⚙️ Configuración

### Variables de Entorno

Crear un archivo `.env` en la raíz del proyecto:

```env
# Base de Datos
DATABASE_URL=postgresql://user:password@localhost/shpd_db

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379

# OpenAI
OPENAI_API_KEY=tu-api-key

# Servidor
HOST=0.0.0.0
PORT=8000
ENVIRONMENT=development

# Seguridad
SECRET_KEY=tu-clave-secreta
CORS_ORIGINS=["http://localhost:3000"]

# Configuración de Análisis
POSTURE_THRESHOLD=0.7
ALERT_DURATION_SECONDS=5
MAX_FRAME_QUEUE_SIZE=30
```

### Archivo de Configuración Adicional

Para configuraciones avanzadas, editar `config/settings.py`:

```python
# Configuración de MediaPipe
MEDIAPIPE_CONFIG = {
    "min_detection_confidence": 0.5,
    "min_tracking_confidence": 0.5,
    "model_complexity": 1
}

# Configuración de alertas
ALERT_CONFIG = {
    "bad_posture_threshold_seconds": 5,
    "notification_cooldown_seconds": 60
}
```

## 📡 Uso de la API

### Endpoints Principales

#### WebSocket Endpoints

```
ws://localhost:8000/video/input/{device_id}
ws://localhost:8000/video/output
```

#### REST API Endpoints

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| `GET` | `/api/v1/health` | Estado del servidor |
| `POST` | `/api/v1/sesiones` | Crear nueva sesión |
| `GET` | `/api/v1/sesiones/{id}` | Obtener sesión |
| `POST` | `/api/v1/pacientes` | Registrar paciente |
| `GET` | `/api/v1/metricas/{sesion_id}` | Métricas de sesión |
| `POST` | `/api/v1/analysis/frame` | Analizar frame individual |
| `GET` | `/api/v1/timeline/{sesion_id}` | Timeline de eventos |

### Ejemplo de Uso

```python
import requests
import websocket

# Crear sesión
response = requests.post("http://localhost:8000/api/v1/sesiones", 
    json={"paciente_id": 1, "device_id": "cam01"})
sesion_id = response.json()["id"]

# Conectar WebSocket para enviar video
ws = websocket.WebSocket()
ws.connect(f"ws://localhost:8000/video/input/cam01")

# Enviar frames
with open("frame.jpg", "rb") as f:
    ws.send(f.read(), opcode=websocket.ABNF.OPCODE_BINARY)
```

Para documentación completa de la API, consultar [API_DOCUMENTATION.md](docs/API_DOCUMENTATION.md)

## 📁 Estructura del Proyecto

```
shpd-backend/
├── api/                    # Módulo principal de la API
│   ├── routers/           # Endpoints organizados por dominio
│   │   ├── sesiones.py    # Gestión de sesiones
│   │   ├── pacientes.py   # CRUD de pacientes
│   │   ├── metricas.py    # Consulta de métricas
│   │   ├── analysis.py    # Análisis de frames
│   │   └── timeline.py    # Eventos temporales
│   ├── models.py          # Modelos SQLAlchemy
│   ├── schemas.py         # Esquemas Pydantic
│   └── database.py        # Configuración DB
├── deploy/                # Scripts de despliegue
├── docs/                  # Documentación adicional
├── tests/                 # Suite de pruebas
├── main.py               # Punto de entrada
├── posture_monitor.py    # Motor de análisis
├── requirements.txt      # Dependencias Python
├── Dockerfile           # Imagen Docker
└── docker-compose.yml   # Orquestación de servicios
```

## 🤝 Contribución

Las contribuciones son bienvenidas. Por favor, seguir estos pasos:

1. Fork el proyecto
2. Crear una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add: nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abrir un Pull Request

### Guía de Estilo

- Seguir [PEP 8](https://www.python.org/dev/peps/pep-0008/) para código Python
- Documentar funciones con docstrings
- Incluir tests para nuevas funcionalidades
- Actualizar documentación si es necesario

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo [LICENSE](LICENSE) para más detalles.

## 👨‍🎓 Autor y Créditos

### Autor
**Rodolfo Giacomodonatto**  
Estudiante de Ingeniería Electrónica  
Universidad Tecnológica Nacional (UTN)

### Proyecto de Tesis
**SHPD - Smart Healthy Posture Detector**  
*Sistema inteligente para la detección y corrección de posturas inadecuadas*

Desarrollado como proyecto final de carrera para la obtención del título de Ingeniero Electrónico en la Universidad Tecnológica Nacional.

### Contacto Académico
- **Universidad**: UTN - Facultad Regional [Especificar]
- **Carrera**: Ingeniería Electrónica
- **Año**: 2024

---

<div align="center">
  <p>Desarrollado con ❤️ para mejorar la salud postural</p>
  <p>© 2024 Rodolfo Giacomodonatto - UTN</p>
</div>
