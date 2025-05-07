import asyncio
import websockets
import cv2

RTSP_URL = "rtsp://192.168.100.41:8554/stream"

async def stream_frames(websocket):
    cap = cv2.VideoCapture(RTSP_URL)
    if not cap.isOpened():
        print("❌ No se pudo abrir el stream RTSP")
        return

    print("🎥 RTSP abierto correctamente. Esperando clientes WebSocket...")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Frame inválido, terminando.")
            break

        frame_count += 1
        if frame_count % 30 == 0:
            print(f"🟢 Enviados {frame_count} frames")

        _, jpeg = cv2.imencode('.jpg', frame)
        await websocket.send(jpeg.tobytes())

    cap.release()

async def handler(websocket):
    print("🔌 Cliente WebSocket conectado")
    await stream_frames(websocket)

async def main():
    async with websockets.serve(handler, "0.0.0.0", 8765):
        print("✅ Backend WebSocket activo en puerto 8765")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
