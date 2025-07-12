# Documentación de API - SHPD Backend

## Tabla de Contenidos

- [Información General](#información-general)
- [Autenticación](#autenticación)
- [WebSocket API](#websocket-api)
- [REST API](#rest-api)
  - [Sesiones](#sesiones)
  - [Pacientes](#pacientes)
  - [Métricas](#métricas)
  - [Análisis](#análisis)
  - [Timeline](#timeline)
  - [Calibración](#calibración)
- [Códigos de Estado](#códigos-de-estado)
- [Ejemplos de Integración](#ejemplos-de-integración)

## Información General

### URL Base
```
http://localhost:8000/api/v1
```

### Headers Requeridos
```http
Content-Type: application/json
Accept: application/json
```

### Formato de Respuesta
Todas las respuestas siguen el formato:
```json
{
  "success": true,
  "data": {},
  "message": "Operación exitosa",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

## Autenticación

Actualmente el sistema no requiere autenticación para fines de desarrollo. En producción se implementará JWT.

## WebSocket API

### 1. Input de Video

**Endpoint**: `ws://localhost:8000/video/input/{device_id}`

**Descripción**: Recibe stream de video para análisis postural en tiempo real.

**Parámetros**:
- `device_id` (string): Identificador único del dispositivo

**Formato de Mensajes**:
- **Entrada**: Frames en formato JPEG (binary)
- **Salida**: JSON con métricas y alertas

**Ejemplo de Respuesta**:
```json
{
  "type": "posture_analysis",
  "device_id": "cam01",
  "timestamp": "2024-01-15T10:30:00Z",
  "posture": {
    "status": "incorrect",
    "confidence": 0.89,
    "alerts": ["Espalda encorvada detectada"],
    "duration_seconds": 5.2
  },
  "landmarks": {
    "nose": {"x": 0.5, "y": 0.3, "z": -0.1},
    "left_shoulder": {"x": 0.4, "y": 0.5, "z": -0.05}
  }
}
```

### 2. Output de Video

**Endpoint**: `ws://localhost:8000/video/output`

**Descripción**: Transmite video procesado con overlays de análisis.

**Formato**: Frames JPEG con anotaciones visuales

## REST API

### Sesiones

#### Crear Sesión

**POST** `/sesiones`

**Body**:
```json
{
  "paciente_id": 1,
  "device_id": "cam01",
  "descripcion": "Sesión de trabajo en escritorio"
}
```

**Respuesta**:
```json
{
  "id": 123,
  "paciente_id": 1,
  "device_id": "cam01",
  "fecha_inicio": "2024-01-15T10:00:00Z",
  "estado": "activa",
  "descripcion": "Sesión de trabajo en escritorio"
}
```

#### Obtener Sesión

**GET** `/sesiones/{sesion_id}`

**Respuesta**:
```json
{
  "id": 123,
  "paciente": {
    "id": 1,
    "nombre": "Juan Pérez",
    "email": "juan@example.com"
  },
  "fecha_inicio": "2024-01-15T10:00:00Z",
  "fecha_fin": "2024-01-15T11:00:00Z",
  "duracion_minutos": 60,
  "metricas_resumen": {
    "tiempo_buena_postura": 45,
    "tiempo_mala_postura": 15,
    "alertas_generadas": 3
  }
}
```

#### Finalizar Sesión

**PUT** `/sesiones/{sesion_id}/finalizar`

**Respuesta**:
```json
{
  "id": 123,
  "estado": "finalizada",
  "fecha_fin": "2024-01-15T11:00:00Z",
  "resumen": {
    "duracion_total": "01:00:00",
    "porcentaje_buena_postura": 75.0
  }
}
```

### Pacientes

#### Crear Paciente

**POST** `/pacientes`

**Body**:
```json
{
  "nombre": "María García",
  "email": "maria@example.com",
  "edad": 28,
  "condiciones_medicas": "Escoliosis leve"
}
```

#### Obtener Paciente

**GET** `/pacientes/{paciente_id}`

#### Listar Pacientes

**GET** `/pacientes`

**Query Parameters**:
- `limit` (int): Número de resultados (default: 20)
- `offset` (int): Desplazamiento (default: 0)

### Métricas

#### Obtener Métricas de Sesión

**GET** `/metricas/{sesion_id}`

**Respuesta**:
```json
{
  "sesion_id": 123,
  "metricas": [
    {
      "timestamp": "2024-01-15T10:05:00Z",
      "tipo": "postura",
      "valor": "correcta",
      "confianza": 0.92
    },
    {
      "timestamp": "2024-01-15T10:10:00Z",
      "tipo": "alerta",
      "valor": "espalda_encorvada",
      "duracion": 5.5
    }
  ],
  "estadisticas": {
    "total_alertas": 3,
    "tiempo_analizado": 3600,
    "promedio_confianza": 0.87
  }
}
```

### Análisis

#### Analizar Frame Individual

**POST** `/analysis/frame`

**Body**:
```json
{
  "image": "base64_encoded_jpeg",
  "sesion_id": 123,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

**Respuesta**:
```json
{
  "postura_detectada": "incorrecta",
  "confianza": 0.85,
  "detalles": {
    "angulo_espalda": 145,
    "angulo_cuello": 35,
    "hombros_alineados": false
  },
  "recomendaciones": [
    "Enderezar la espalda",
    "Alinear hombros con caderas"
  ]
}
```

#### Obtener Análisis con IA

**POST** `/analysis/ai`

**Body**:
```json
{
  "image": "base64_encoded_jpeg",
  "contexto": "Usuario trabajando en computadora"
}
```

### Timeline

#### Obtener Timeline de Eventos

**GET** `/timeline/{sesion_id}`

**Query Parameters**:
- `start_time` (ISO 8601): Tiempo inicial
- `end_time` (ISO 8601): Tiempo final
- `event_type` (string): Filtrar por tipo de evento

**Respuesta**:
```json
{
  "sesion_id": 123,
  "eventos": [
    {
      "timestamp": "2024-01-15T10:15:00Z",
      "tipo": "cambio_postura",
      "de": "correcta",
      "a": "incorrecta"
    },
    {
      "timestamp": "2024-01-15T10:20:00Z",
      "tipo": "alerta",
      "mensaje": "Postura incorrecta por más de 5 minutos"
    }
  ]
}
```

### Calibración

#### Iniciar Calibración

**POST** `/calibracion/iniciar`

**Body**:
```json
{
  "device_id": "cam01",
  "tipo_calibracion": "escritorio"
}
```

#### Obtener Estado de Calibración

**GET** `/calibracion/estado/{device_id}`

## Códigos de Estado

| Código | Descripción |
|--------|-------------|
| 200 | Operación exitosa |
| 201 | Recurso creado |
| 400 | Solicitud inválida |
| 404 | Recurso no encontrado |
| 422 | Entidad no procesable |
| 500 | Error interno del servidor |

## Ejemplos de Integración

### Python

```python
import requests
import websocket
import base64
import json

# Configuración
API_BASE = "http://localhost:8000/api/v1"
WS_BASE = "ws://localhost:8000"

# Crear sesión
session = requests.post(f"{API_BASE}/sesiones", 
    json={"paciente_id": 1, "device_id": "cam01"})
sesion_id = session.json()["id"]

# Conectar WebSocket
ws = websocket.WebSocketApp(f"{WS_BASE}/video/input/cam01",
    on_message=lambda ws, msg: print(f"Análisis: {msg}"),
    on_error=lambda ws, err: print(f"Error: {err}"))

# Enviar frame
with open("frame.jpg", "rb") as f:
    ws.send(f.read(), opcode=websocket.ABNF.OPCODE_BINARY)

# Obtener métricas
metricas = requests.get(f"{API_BASE}/metricas/{sesion_id}")
print(metricas.json())
```

### JavaScript

```javascript
// Crear sesión
const response = await fetch('http://localhost:8000/api/v1/sesiones', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
        paciente_id: 1,
        device_id: 'cam01'
    })
});
const {id: sesionId} = await response.json();

// Conectar WebSocket
const ws = new WebSocket(`ws://localhost:8000/video/input/cam01`);

ws.onmessage = (event) => {
    const analysis = JSON.parse(event.data);
    console.log('Análisis postural:', analysis);
};

// Enviar frame
const frameBlob = await fetch('frame.jpg').then(r => r.blob());
ws.send(frameBlob);
```

---

Para más información o soporte técnico, contactar al equipo de desarrollo SHPD.