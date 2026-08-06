# Placeholder Dockerfile — populated in a later step once dependencies
# (OpenCV, YOLO/torch, etc.) and the runtime entrypoint are finalized.
#
# Planned shape:
#   FROM python:3.10-slim (or nvidia/cuda base for GPU builds)
#   WORKDIR /app
#   COPY requirements.txt .
#   RUN pip install -r requirements.txt
#   COPY . .
#   ENTRYPOINT ["python", "run.py"]
