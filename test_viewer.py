"""뷰어 데이터·API 테스트: python -m pytest test_viewer.py -q (합성 데이터, 영상 없는 환경에서도 돈다)"""
import csv
from pathlib import Path

import cv2
import numpy as np
import pytest

import viewer_data as vd

STEM = "match1_set1"


def _write(path: Path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def make_outputs(tmp_path):
    """합성 결과 한 벌: 랠리 3개(마지막은 득점 팀 없음), 이벤트 4개, 선수 100–200·260–400, 공(101은 빈칸), 1000프레임 영상"""
    out, data = tmp_path / "outputs", tmp_path / "data"
    out.mkdir()
    data.mkdir()
    _write(out / f"{STEM}_rallies.csv", ["start", "serve_team", "winner", "contacts", "seconds"],
           [(100, "RIGHT", "RIGHT", 6, 10.0), (400, "RIGHT", "LEFT", 1, 3.0), (700, "LEFT", "", 5, 8.0)])
    _write(out / f"{STEM}_events.csv", ["start", "end", "peak", "group", "team", "action", "conf", "actor", "court_x", "court_y"],
           [(110, 118, 112, "l-pass", "LEFT", "pass", 0.9, 7, 4.0, 5.0), (120, 128, 124, "l-spike", "LEFT", "spike", 0.8, 8, 4.5, 8.0),
            (450, 458, 452, "r-pass", "RIGHT", "pass", 0.7, 9, 3.0, 12.0), (500, 508, 502, "r-winpoint", "RIGHT", "winpoint", 0.9, "", "", "")])
    trk = [(f, 1, "LEFT", 10.0, 20.0, 50.0, 120.0, 4.0, 5.0, 0) for f in [*range(100, 201), *range(260, 401)]]
    _write(out / f"{STEM}_tracks.csv", ["frame", "track_id", "team", "x1", "y1", "x2", "y2", "court_x", "court_y", "airborne"], trk)
    _write(out / f"{STEM}_ball.csv", ["frame", "x", "y", "score"], [(100, 30.5, 40.5, 0.6), (101, "", "", ""), (102, 33.0, 41.0, 0.6)])
    vw = cv2.VideoWriter(str(data / f"{STEM}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 48))
    for _ in range(1000):
        vw.write(np.zeros((48, 64, 3), np.uint8))
    vw.release()
    # 랠리가 없는 결과(데이터셋 클립처럼 헤더만): 경기로 안 잡혀야 한다
    _write(out / "long_20_rallies.csv", ["start", "serve_team", "winner", "contacts", "seconds"], [])
    return out, data


def test_find_matches(tmp_path):
    out, data = make_outputs(tmp_path)
    m = vd.find_matches(out, data)
    assert [x["stem"] for x in m] == [STEM]
    assert m[0]["fps"] == pytest.approx(30) and m[0]["frames"] == 1000 and m[0]["has_video"]
    assert m[0]["playable"] is False  # mp4v 는 브라우저가 못 읽는다


def test_find_matches_without_video(tmp_path):
    out, data = make_outputs(tmp_path)
    (data / f"{STEM}.mp4").unlink()
    m = vd.find_matches(out, data)[0]
    assert m["fps"] == 29.97 and m["frames"] is None and not m["has_video"] and not m["playable"]


def test_load_rallies(tmp_path):
    out, data = make_outputs(tmp_path)
    r = vd.load_rallies(out, STEM, 30, 1000)
    assert [x["n"] for x in r] == [1, 2, 3]
    assert [x["warn"] for x in r] == [[], ["접촉 2 이하", "5초 미만"], ["득점 팀 없음"]]
    assert r[0]["window"] == [40, 100 + 300 + 60] and r[0]["end"] == 400
    assert r[2]["window"][1] == 999 and r[2]["winner"] is None  # 영상 끝으로 자름


def test_rally_events(tmp_path):
    out, _ = make_outputs(tmp_path)
    ev = vd.rally_events(out, STEM, 100, 400)
    assert [e["action"] for e in ev] == ["pass", "spike"]  # winpoint·구간 밖 제외
    assert ev[0]["actor"] == 7 and ev[0]["court_y"] == 5.0


def test_frames_slice(tmp_path):
    out, _ = make_outputs(tmp_path)
    s = vd.frames_slice(out, STEM, 100, 102)
    assert [p[0] for p in s["players"]] == [100, 101, 102] and s["players"][0][2] == 0  # LEFT=0
    assert [b[0] for b in s["ball"]] == [100, 102]  # 빈 공 행은 버린다
    assert vd.frames_slice(out, "nope", 5, 9) == {"players": [], "ball": [], "gaps": [[5, 9]]}


def test_no_court_gaps():
    present = {*range(100, 201), *range(260, 401)}
    assert vd.no_court_gaps(present, 100, 400) == [[201, 259]]
    assert vd.no_court_gaps(present | set(range(201, 250)), 100, 400) == []  # 남은 구멍 10프레임은 무시


def test_summary(tmp_path):
    out, _ = make_outputs(tmp_path)
    s = vd.summary(vd.load_rallies(out, STEM, 30, 1000))
    assert s["rallies"] == 3
    assert s["RIGHT"] == dict(points=1, serves=2, serve_wins=1, receives=0, receive_wins=0, serve_rate=0.5, sideout_rate=None)
    assert s["LEFT"] == dict(points=1, serves=0, serve_wins=0, receives=2, receive_wins=1, serve_rate=None, sideout_rate=0.5)
