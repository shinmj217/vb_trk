"""분석 결과 뷰어 서버: python app.py [--outputs outputs] [--data data] [--port 8000]

랠리 복기(영상 + 박스·공·이벤트 겹침 + 코트 지도)와 경기 통계를 브라우저로 본다. 분석 산출물은 읽기만 한다.
경기 = outputs/<이름>_rallies.csv 가 있는 이름, 영상 = data/<이름>.mp4 (UI로 올린 것은 data/uploads/).
영상 올리기 → 분석 시작(파이프라인 5단계를 분리 프로세스로, analysis_jobs.py) → 끝나면 경기 목록에 나타난다. (선수·공 검출 모델 서빙은 serve.py)
"""
import argparse
import os
import shutil
import sys
import threading
import webbrowser
from pathlib import Path

import cv2
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import analysis_jobs as aj
import viewer_data as vd

UI = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent)) / "ui"  # exe 로 묶이면 임시 풀린 폴더 안
MAX_SPAN = 6000  # /frames 한 번에 받는 최대 프레임 수 (약 200초)


def create_app(outputs: Path, data: Path, jobs: Path = Path("jobs"), required=aj.REQUIRED, min_free_gb: float = aj.MIN_FREE_GB) -> FastAPI:
    outputs, data, jobs = Path(outputs), Path(data), Path(jobs)
    app = FastAPI(title="vb_trk viewer")

    def matches() -> dict:  # 요청마다 새로 (분석이 끝나면 바로 목록에 뜬다). 이 목록에 없는 이름은 전부 404
        return {m["stem"]: m for m in vd.find_matches(outputs, data)}

    def match(stem: str) -> dict:
        m = matches().get(stem)
        if m is None:
            raise HTTPException(404, "없는 경기입니다")
        return m

    def rallies(stem: str):
        m = match(stem)
        return vd.load_rallies(outputs, stem, m["fps"], m["frames"])

    @app.middleware("http")
    async def no_cache(request, call_next):  # 화면 파일은 항상 최신본 (수정 후 옛 HTML·JS가 섞여 깨지지 않게)
        r = await call_next(request)
        if request.url.path == "/" or request.url.path.startswith("/static/"):
            r.headers["Cache-Control"] = "no-cache"
        return r

    @app.get("/api/matches")
    def list_matches():
        return list(matches().values())

    @app.get("/api/matches/{stem}/rallies")
    def get_rallies(stem: str):
        rs = rallies(stem)
        for r in rs:
            r["events"] = vd.rally_events(outputs, stem, r["start"], r["end"])
        return rs

    @app.get("/api/matches/{stem}/summary")
    def get_summary(stem: str):
        return vd.summary(rallies(stem))

    @app.get("/api/matches/{stem}/frames")
    def get_frames(stem: str, start: int = Query(...), end: int = Query(...)):
        match(stem)
        if start < 0 or start > end or end - start > MAX_SPAN:
            raise HTTPException(400, f"구간이 올바르지 않습니다 (0 ≤ start ≤ end, 길이 {MAX_SPAN} 이하)")
        return vd.frames_slice(outputs, stem, start, end)

    @app.get("/api/matches/{stem}/figure")
    def get_figure(stem: str):
        match(stem)
        f = outputs / f"{stem}_analysis.png"
        if not f.exists():
            raise HTTPException(404, "분석 그림이 없습니다")
        return FileResponse(f, media_type="image/png")

    @app.get("/video/{stem}")
    def get_video(stem: str):
        match(stem)
        f = vd.video_path(data, stem)
        if f is None:
            raise HTTPException(404, "영상이 없습니다")
        return FileResponse(f, media_type="video/mp4")  # Starlette 가 Range(206·416) 처리

    # ---- 영상 올리기 · 분석 작업 ----
    def job_json(st: dict) -> dict:
        return {**st, "overall": aj.overall(st)}

    @app.put("/api/upload")
    async def upload(request: Request, name: str = Query(...)):
        """본문 = mp4 파일 그대로 (브라우저 진행률 표시를 위해 multipart 대신). data/uploads/<이름>.mp4 로 저장"""
        try:
            stem = aj.safe_stem(name)
        except aj.JobError as e:
            raise HTTPException(400, str(e))
        dest = data / "uploads" / f"{stem}.mp4"
        if dest.exists() or vd.video_path(data, stem) or (outputs / f"{stem}_rallies.csv").exists() or aj.get_state(jobs, stem):
            raise HTTPException(409, f"같은 이름({stem})의 영상이 이미 있습니다. 파일 이름을 바꿔서 올려 주세요")
        size = int(request.headers.get("content-length") or 0)
        if size and shutil.disk_usage(data).free < size + min_free_gb * 1e9:
            raise HTTPException(507, "디스크 여유가 부족합니다")
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_suffix(".part")
        try:
            with open(part, "wb") as f:
                async for chunk in request.stream():
                    await run_in_threadpool(f.write, chunk)
            cap = cv2.VideoCapture(str(part))
            ok = cap.isOpened() and cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0
            cap.release()
            if not ok:
                raise HTTPException(400, "영상을 읽을 수 없습니다 (손상됐거나 mp4가 아닙니다)")
            os.replace(part, dest)
        finally:
            part.unlink(missing_ok=True)
        return job_json(aj.create(jobs, stem, dest))

    @app.get("/api/jobs")
    def get_jobs():
        aj.kick(jobs, data, outputs)  # 러너가 비정상 종료돼 대기열이 멈춘 경우에도 다음 작업을 시작한다
        return [job_json(s) for s in aj.list_states(jobs)]

    @app.post("/api/jobs/{stem}/start")
    def start_job(stem: str):
        try:
            return job_json(aj.start(stem, jobs, data, outputs, required, min_free_gb))
        except aj.JobError as e:
            raise HTTPException(409 if "이미" in str(e) else 400, str(e))

    @app.post("/api/jobs/{stem}/cancel")
    def cancel_job(stem: str):
        try:
            return job_json(aj.cancel(stem, jobs, data, outputs))
        except aj.JobError as e:
            raise HTTPException(400, str(e))

    @app.delete("/api/jobs/{stem}")
    def delete_job(stem: str):
        try:
            return {"deleted": aj.delete(stem, jobs, data, outputs)}
        except aj.JobError as e:
            raise HTTPException(400, str(e))

    @app.get("/")
    def index():
        return FileResponse(UI / "index.html")

    if UI.exists():
        app.mount("/static", StaticFiles(directory=UI), name="static")
    return app


def main():
    import uvicorn
    p = argparse.ArgumentParser()
    base = aj.ROOT if aj.FROZEN else Path(".")  # exe 는 exe 가 있는 폴더 기준 (더블클릭하면 작업 폴더가 달라질 수 있어서)
    p.add_argument("--outputs", default=str(base / "outputs"))
    p.add_argument("--data", default=str(base / "data"))
    p.add_argument("--jobs", default=str(base / "jobs"))
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--no-browser", action="store_true", help="브라우저를 자동으로 열지 않는다")
    a = p.parse_args()
    app = create_app(Path(a.outputs), Path(a.data), Path(a.jobs))
    if not a.no_browser:
        threading.Timer(1.0, webbrowser.open, [f"http://localhost:{a.port}"]).start()
    uvicorn.run(app, host="127.0.0.1", port=a.port)


if __name__ == "__main__":
    main()
