# api/routers/sesiones.py

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from typing import List
from api.database import get_db
from api.models import Sesion
from api.schemas import SesionIn, SesionOut
import time
import redis

r = redis.Redis(host="redis", port=6379, decode_responses=True)
router = APIRouter(prefix="/sesiones", tags=["sesiones"])

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