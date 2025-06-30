# api/routers/sesiones.py

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from typing import List
from api.database import get_db
from api.models import Sesion, Paciente
from api.schemas import SesionIn, SesionOut
import time
import redis
import requests
import json
import logging

r = redis.Redis(host="redis", port=6379, decode_responses=True)
router = APIRouter(prefix="/sesiones", tags=["sesiones"])
logger = logging.getLogger(__name__)

BOT_API_URL = "http://bot-api-service:8000/send_report"  # Cambia si tu Service tiene otro nombre

@router.post("/", response_model=SesionOut)
def crear_sesion(
    s: SesionIn,
    db: Session = Depends(get_db)
) -> SesionOut:
    nueva = Sesion(**s.dict())
    db.add(nueva)
    db.commit()
    db.refresh(nueva)
    return nueva

@router.get("/", response_model=List[SesionOut])
def listar_sesiones(
    db: Session = Depends(get_db)
) -> List[SesionOut]:
    return db.query(Sesion).all()

@router.get("/progress/{session_id}")
def get_session_progress(session_id: str):
    """
    Devuelve el progreso de la sesión calculado desde Redis:
    - intervalo_segundos: duración total de la sesión
    - elapsed: segundos transcurridos (no reinicia al recargar)
    """
    key = f"shpd-session:{session_id}"
    data = r.hgetall(key)
    start_ts = int(data.get("start_ts", 0))
    intervalo = int(data.get("intervalo_segundos", 0))
    now = int(time.time())
    elapsed = now - start_ts
    if elapsed > intervalo:
        elapsed = intervalo
    return {"intervalo_segundos": intervalo, "elapsed": elapsed}

def enviar_reporte_telegram(session_id, device_id, db: Session):
    paciente = db.query(Paciente).filter(Paciente.device_id == device_id).first()
    if not paciente:
        raise Exception("Paciente no encontrado para el device_id")
    telegram_id = paciente.telegram_id

    # 2. Obtener métricas finales (de Redis)
    metricas = r.lrange(f"metricas:{session_id}", 0, -1)
    resumen = "No hay métricas registradas."
    if metricas:
        ultima = json.loads(metricas[-1])
        resumen = (
            f"✅ <b>Reporte de sesión finalizada</b>\n"
            f"Correcta: {ultima.get('porcentaje_correcta', 0)}%\n"
            f"Incorrecta: {ultima.get('porcentaje_incorrecta', 0)}%\n"
            f"Sentado: {ultima.get('tiempo_sentado', 0)}s\n"
            f"Parado: {ultima.get('tiempo_parado', 0)}s\n"
            f"Alertas: {ultima.get('alertas_enviadas', 0)}"
        )
    # Limpiar datos de Redis
    r.delete(f"shpd-session:{session_id}")
    r.delete(f"metricas:{session_id}")
    r.delete(f"analysis:{session_id}")
    # 3. Llamar al bot por HTTP
    payload = {"telegram_id": telegram_id, "resumen": resumen}
    try:
        resp = requests.post(BOT_API_URL, json=payload, timeout=5)
        resp.raise_for_status()
        logger.info("Reporte enviado correctamente a Telegram.")
    except Exception as e:
        logger.error(f"Error enviando reporte a Telegram: {e}")
        raise

@router.post("/end/{device_id}")
def finalizar_sesion(device_id: str, db: Session = Depends(get_db)):
    """
    Finaliza la sesión usando el device_id: busca el session_id en Redis, limpia los datos temporales y envía el reporte.
    """
    shpd_data = r.hgetall(f"shpd-data:{device_id}")
    session_id = shpd_data.get("session_id")
    if not session_id:
        return {"ok": False, "message": "No se encontró session_id para este device_id"}

    # Enviar reporte a Telegram
    try:
        enviar_reporte_telegram(session_id, device_id, db)
        return {"ok": True, "message": "Sesión finalizada, datos eliminados y reporte enviado"}
    except Exception as e:
        return {"ok": False, "message": f"Sesión finalizada, pero falló el envío a Telegram: {e}"}