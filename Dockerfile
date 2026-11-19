# 선수 검출 서빙 이미지: docker build -t vb-detect . && docker run -p 8000:8000 vb-detect
# GPU: docker run --gpus all -e VB_DEVICE=0 -p 8000:8000 vb-detect  (호스트에 NVIDIA Container Toolkit 필요)
FROM python:3.12-slim
WORKDIR /app
# ultralytics 가 GUI용 opencv-python 을 끌어와 libGL/libxcb 가 필요 (slim 이미지엔 없음)
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt
COPY serve.py yolo26m.pt ./
ENV VB_MODEL=yolo26m.pt VB_IMGSZ=1280
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD python -c "import urllib.request as u;u.urlopen('http://localhost:8000/health')" || exit 1
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
