from fastapi import APIRouter
import redis
import logging

r = redis.Redis(host="redis", port=6379, decode_responses=True)
logger = logging.getLogger("calibracion")
logger.setLevel(logging.DEBUG)
router = APIRouter(prefix="/calib", tags=["calibracion"])

# Nuevo endpoint de progreso numérico
@router.get("/progress/{session_id}")
def calib_progress(session_id: str):
    data = r.hgetall(f"calib:{session_id}")
    good = float(data.get("good_time", 0))
    bad  = float(data.get("bad_time", 0))
    return {
        "good_time": good,
        "bad_time": bad,
        "correcta": good > bad
    }
