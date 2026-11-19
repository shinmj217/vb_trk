"""선수 검출 모델 서빙 (프레임 한 장 → 사람 박스). MLOps 1단계

    uvicorn serve:app --port 8000
    curl -F "file=@frame.jpg" "http://localhost:8000/detect?conf=0.3"

환경변수: VB_MODEL (기본 yolo26m.pt), VB_DEVICE (기본 자동), VB_IMGSZ (기본 1280)
GET /health   모델 로드 여부
POST /detect  multipart 이미지 → {"width","height","persons":[{"box":[x1,y1,x2,y2],"score"}],"ms"}
GET /metrics  Prometheus (요청 수, 추론 지연)
"""
import os
import time
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

MAX_BYTES = 20 * 1024 * 1024  # 20MB (4K JPEG도 충분)
REQUESTS = Counter("vb_detect_requests_total", "detect 요청 수", ["status"])
LATENCY = Histogram("vb_detect_inference_seconds", "모델 추론 시간 (초)", buckets=(.02, .05, .1, .2, .5, 1, 2, 5))
PERSONS = Histogram("vb_detect_persons", "프레임당 검출된 사람 수", buckets=(0, 2, 5, 10, 15, 20, 30))
state = {}


@asynccontextmanager
async def lifespan(app):
    from ultralytics import YOLO
    state["model"] = YOLO(os.environ.get("VB_MODEL", "yolo26m.pt"))
    yield
    state.clear()


app = FastAPI(title="vb_trk player detector", lifespan=lifespan)


@app.get("/health")
def health():
    return {"ok": "model" in state}


@app.post("/detect")
async def detect(file: UploadFile = File(...), conf: float = Query(0.3, ge=0.0, le=1.0)):
    data = await file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        REQUESTS.labels("too_large").inc()
        raise HTTPException(413, "이미지가 20MB를 넘습니다")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        REQUESTS.labels("bad_image").inc()
        raise HTTPException(400, "이미지를 읽을 수 없습니다")
    t0 = time.perf_counter()
    r = state["model"].predict(img, imgsz=int(os.environ.get("VB_IMGSZ", 1280)), classes=[0], conf=conf,
                               device=os.environ.get("VB_DEVICE") or None, verbose=False)[0]
    dt = time.perf_counter() - t0
    boxes, scores = r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()
    LATENCY.observe(dt)
    PERSONS.observe(len(boxes))
    REQUESTS.labels("ok").inc()
    return {"width": img.shape[1], "height": img.shape[0], "ms": round(dt * 1000, 1),
            "persons": [{"box": [round(float(v), 1) for v in b], "score": round(float(s), 3)} for b, s in zip(boxes, scores)]}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
