"""뷰어가 읽는 분석 결과 (CSV → 랠리·구간·통계). 읽기 전용, 서버(app.py)와 테스트가 같이 쓴다.

outputs/<이름>_{rallies,events,tracks,ball}.csv 를 읽고, 영상 메타는 data/<이름>.mp4 에서 읽는다.
"""
import csv
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

DEFAULT_FPS = 29.97
PAD_S = 2          # 랠리 재생 구간 앞뒤 여유 (초)
GAP_MIN = 30       # 선수 행이 이만큼(프레임) 연속으로 없으면 '코트 인식 실패' 구간
PLAYABLE = {"VP90", "avc1", "H264", "AV01"}  # 브라우저가 읽는 코덱 (Chrome·Edge 기준)
TEAM_IDX = {"LEFT": 0, "RIGHT": 1}
TEAMS = ("LEFT", "RIGHT")


@lru_cache(maxsize=16)
def _cached(path: str, mtime: int):
    df = pd.read_csv(path, encoding="utf-8")
    return df.sort_values("frame").reset_index(drop=True) if "frame" in df.columns else df


def _table(path: Path):
    """CSV → DataFrame (없으면 None). 15만 행 추적 CSV 를 매 요청마다 다시 읽지 않도록 mtime 으로 캐시"""
    return _cached(str(path), path.stat().st_mtime_ns) if path.exists() else None


def _video_meta(video: Path):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return None
    meta = dict(fps=cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS, frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                fourcc=int(cap.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, "little").decode("latin-1"))
    cap.release()
    return meta


def video_path(data: Path, stem: str):
    """영상 파일 위치: data/<이름>.mp4, 없으면 UI로 올린 data/uploads/<이름>.mp4, 둘 다 없으면 None"""
    for p in (Path(data) / f"{stem}.mp4", Path(data) / "uploads" / f"{stem}.mp4"):
        if p.exists():
            return p
    return None


def find_matches(outputs: Path, data: Path):
    """랠리가 1개 이상 있는 결과 = 경기. 영상 메타(fps·크기·재생 가능 여부)를 같이 돌려준다"""
    out = []
    for f in sorted(Path(outputs).glob("*_rallies.csv")):
        stem = f.name[: -len("_rallies.csv")]
        with open(f, encoding="utf-8") as fh:
            if sum(1 for _ in csv.DictReader(fh)) == 0:
                continue
        vp = video_path(data, stem)
        m = _video_meta(vp) if vp else None
        out.append(dict(stem=stem, fps=m["fps"] if m else DEFAULT_FPS, frames=m["frames"] if m else None,
                        width=m["width"] if m else None, height=m["height"] if m else None,
                        has_video=m is not None, playable=bool(m and m["fourcc"] in PLAYABLE)))
    return out


def load_rallies(outputs: Path, stem: str, fps: float, frames):
    """랠리 목록. 재생 구간 window = [시작−2초, 끝+2초] 를 0..frames−1 로 자른다. 득점 팀이 빈칸이면 None"""
    pad, out = round(PAD_S * fps), []
    with open(Path(outputs) / f"{stem}_rallies.csv", encoding="utf-8") as fh:
        for n, r in enumerate(csv.DictReader(fh), 1):
            start, sec = int(r["start"]), float(r["seconds"])
            end, contacts = start + round(sec * fps), int(r["contacts"])
            b = end + pad if frames is None else min(frames - 1, end + pad)
            warn = [w for ok, w in ((contacts <= 2, "접촉 2 이하"), (sec < 5, "5초 미만"), (not r["winner"], "득점 팀 없음")) if ok]
            out.append(dict(n=n, start=start, end=end, serve_team=r["serve_team"], winner=r["winner"] or None,
                            contacts=contacts, seconds=sec, window=[max(0, start - pad), b], warn=warn))
    return out


def rally_events(outputs: Path, stem: str, start: int, end: int):
    """랠리 안 이벤트 (start ≤ event.start < end), winpoint 제외"""
    df = _table(Path(outputs) / f"{stem}_events.csv")
    if df is None or df.empty:
        return []
    df = df[(df.action != "winpoint") & (df.start >= start) & (df.start < end)]
    rows = []
    for e in df.to_dict("records"):
        e = {k: (None if pd.isna(v) else v) for k, v in e.items()}
        e["actor"] = None if e["actor"] is None else int(e["actor"])
        rows.append(dict(start=int(e["start"]), end=int(e["end"]), peak=int(e["peak"]), team=e["team"], action=e["action"],
                         actor=e["actor"], court_x=e["court_x"], court_y=e["court_y"]))
    return rows


def no_court_gaps(frames_present, a: int, b: int, min_len: int = GAP_MIN):
    """a..b 에서 선수 행이 없는 연속 구간 중 min_len 이상 → [첫 프레임, 마지막 프레임]"""
    gaps, run = [], None
    for f in range(a, b + 2):
        if f <= b and f not in frames_present:
            run = [f, f] if run is None else [run[0], f]
        elif run is not None:
            if run[1] - run[0] + 1 >= min_len:
                gaps.append(run)
            run = None
    return gaps


def _rows(df, a, b, cols):
    fr = df.frame.to_numpy()
    return df.iloc[np.searchsorted(fr, a, "left"):np.searchsorted(fr, b, "right")][cols]


def frames_slice(outputs: Path, stem: str, a: int, b: int):
    """프레임 a..b(양끝 포함)의 선수 박스·공 위치와 코트 인식 실패 구간. 파일이 없으면 빈 목록"""
    out = Path(outputs)
    trk, ball = _table(out / f"{stem}_tracks.csv"), _table(out / f"{stem}_ball.csv")
    players, gaps = [], [[a, b]]
    if trk is not None:
        cols = ["frame", "track_id", "team", "x1", "y1", "x2", "y2", "court_x", "court_y", "airborne"]
        for r in _rows(trk, a, b, cols).itertuples(index=False):
            players.append([int(r.frame), int(r.track_id), TEAM_IDX.get(r.team, -1), *(round(float(v), 1) for v in (r.x1, r.y1, r.x2, r.y2)),
                            round(float(r.court_x), 2), round(float(r.court_y), 2), int(r.airborne)])
        gaps = no_court_gaps({p[0] for p in players}, a, b)
    balls = []
    if ball is not None:
        for r in _rows(ball.dropna(subset=["x", "y"]), a, b, ["frame", "x", "y"]).itertuples(index=False):
            balls.append([int(r.frame), round(float(r.x), 1), round(float(r.y), 1)])
    return dict(players=players, ball=balls, gaps=gaps)


def summary(rallies):
    """팀별 득점·서브 득점률·사이드아웃률 (analysis.py 와 같은 정의, 득점 팀 없는 랠리는 제외)"""
    done = [r for r in rallies if r["winner"]]
    res = {"rallies": len(rallies)}
    for t in TEAMS:
        sv = [r for r in done if r["serve_team"] == t]
        rc = [r for r in done if r["serve_team"] != t]
        sw, rw = sum(r["winner"] == t for r in sv), sum(r["winner"] == t for r in rc)
        res[t] = dict(points=sum(r["winner"] == t for r in done), serves=len(sv), serve_wins=sw, receives=len(rc), receive_wins=rw,
                      serve_rate=round(sw / len(sv), 3) if sv else None, sideout_rate=round(rw / len(rc), 3) if rc else None)
    return res
