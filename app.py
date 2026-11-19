"""분석 결과 뷰어 서버: python app.py [--outputs outputs] [--data data] [--port 8000]

랠리 복기(영상 + 박스·공·이벤트 겹침 + 코트 지도)와 경기 통계를 브라우저로 본다. 분석 산출물은 읽기만 한다.
경기 = outputs/<이름>_rallies.csv 가 있는 이름, 영상 = data/<이름>.mp4. (선수·공 검출 모델 서빙은 serve.py)
"""
import argparse
import threading
import webbrowser
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import viewer_data as vd

UI = Path(__file__).resolve().parent / "ui"
MAX_SPAN = 6000  # /frames 한 번에 받는 최대 프레임 수 (약 200초)


def create_app(outputs: Path, data: Path) -> FastAPI:
    outputs, data = Path(outputs), Path(data)
    app = FastAPI(title="vb_trk viewer")
    matches = {m["stem"]: m for m in vd.find_matches(outputs, data)}  # 시작할 때 한 번 (이 목록에 없는 이름은 전부 404)

    def match(stem: str) -> dict:
        if stem not in matches:
            raise HTTPException(404, "없는 경기입니다")
        return matches[stem]

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
        return list(matches.values())

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
        f = data / f"{stem}.mp4"
        if not f.exists():
            raise HTTPException(404, "영상이 없습니다")
        return FileResponse(f, media_type="video/mp4")  # Starlette 가 Range(206·416) 처리

    @app.get("/")
    def index():
        return FileResponse(UI / "index.html")

    if UI.exists():
        app.mount("/static", StaticFiles(directory=UI), name="static")
    return app


def main():
    import uvicorn
    p = argparse.ArgumentParser()
    p.add_argument("--outputs", default="outputs")
    p.add_argument("--data", default="data")
    p.add_argument("--port", type=int, default=8000)
    a = p.parse_args()
    app = create_app(Path(a.outputs), Path(a.data))
    threading.Timer(1.0, webbrowser.open, [f"http://localhost:{a.port}"]).start()
    uvicorn.run(app, host="127.0.0.1", port=a.port)


if __name__ == "__main__":
    main()
