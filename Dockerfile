FROM python:3.11-slim

# Evita prompts interactivos durante instalación
ENV DEBIAN_FRONTEND=noninteractive

# Instala dependencias del sistema necesarias para opencv y mediapipe
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py posture_monitor.py .
CMD ["python", "main.py"]
