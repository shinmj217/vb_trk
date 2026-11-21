"""분석 작업(업로드 → 5단계 백그라운드 실행) 테스트: python -m pytest test_jobs.py -q
가짜 단계(짧은 python -c)로 상태 전이·진행률·실패·재시작·취소를 확인한다. 진짜 파이프라인은 안 돈다."""
import json
import sys
import time
from pathlib import Path

import pytest

import analysis_jobs as aj

PY = sys.executable


def stage(name, code, weight=1, phases=None):
    return dict(name=name, label=name, weight=weight, phases=phases or {"단계": (0, 1)}, argv=[PY, "-u", "-c", code], env={})


def setup(tmp_path):
    jobs, data, out = tmp_path / "jobs", tmp_path / "data", tmp_path / "outputs"
    for d in (jobs, data / "uploads", out):
        d.mkdir(parents=True)
    (data / "uploads" / "x.mp4").write_bytes(b"fake")
    aj.create(jobs, "x", data / "uploads" / "x.mp4")
    return jobs, data, out


def test_parse_progress():
    ph = {"추적": (0, 0.8), "저장": (0.8, 1.0)}
    assert aj.parse_progress("[진행] 추적 150/300", ph) == pytest.approx(0.4)
    assert aj.parse_progress("[진행] 저장 0/300", ph) == pytest.approx(0.8)
    assert aj.parse_progress("SAM3 코트 50/100  0.4s/frame", {"SAM3 코트": (0, 0.95)}) == pytest.approx(0.475)
    assert aj.parse_progress("아무 줄", ph) is None
    assert aj.parse_progress("[진행] 추적 5/0", ph) is None  # 분모 0


def test_overall():
    st = {"stages": [dict(weight=40, status="done", frac=1), dict(weight=60, status="running", frac=0.5)]}
    assert aj.overall(st) == pytest.approx(0.7)


def test_run_all_done(tmp_path):
    jobs, data, out = setup(tmp_path)
    stages = [stage("a", "print('[진행] 단계 1/2'); print('[진행] 단계 2/2')", 1), stage("b", "print('끝')", 3)]
    aj.run("x", jobs, data, out, stages=stages)
    st = aj.get_state(jobs, "x")
    assert st["status"] == "done" and [s["status"] for s in st["stages"]] == ["done", "done"] and aj.overall(st) == 1
    assert "[진행] 단계 2/2" in (out / "x_pipeline.log").read_text(encoding="utf-8")


def test_run_failure_stops_and_reports(tmp_path):
    jobs, data, out = setup(tmp_path)
    stages = [stage("a", "print('ok')"), stage("b", "import sys; print('폭발 직전'); sys.exit(3)"), stage("c", "print('안 돌아야 함')")]
    aj.run("x", jobs, data, out, stages=stages)
    st = aj.get_state(jobs, "x")
    assert st["status"] == "failed" and st["error"].startswith("b 단계") and "폭발 직전" in st["error"]
    assert [s["status"] for s in st["stages"]] == ["done", "failed", "pending"]


def test_resume_skips_done_stages(tmp_path):
    jobs, data, out = setup(tmp_path)
    mark = tmp_path / "mark.txt"
    first = f"open(r'{mark}', 'a').write('a')"
    stages = [stage("a", first), stage("b", "import sys; sys.exit(1)")]
    aj.run("x", jobs, data, out, stages=stages)                    # a 성공, b 실패
    stages[1] = stage("b", "print('이번엔 성공')")
    aj.run("x", jobs, data, out, stages=stages)                    # 재시작: a 는 건너뛴다
    assert mark.read_text() == "a" and aj.get_state(jobs, "x")["status"] == "done"


def test_dead_runner_is_reported(tmp_path):
    jobs, data, out = setup(tmp_path)
    st = aj.get_state(jobs, "x")
    st.update(status="running", pid=99999999)
    aj.write_state(jobs, st)
    got = aj.get_state(jobs, "x")
    assert got["status"] == "failed" and "예기치 않게" in got["error"]


def test_start_and_cancel_detached(tmp_path, monkeypatch):
    jobs, data, out = setup(tmp_path)
    monkeypatch.setenv("VB_JOB_STAGES_JSON", json.dumps([stage("slow", "import time; print('[진행] 단계 1/9', flush=True); time.sleep(60)")]))
    aj.start("x", jobs, data, out, required=[], min_free_gb=0)
    for _ in range(50):                                            # 러너가 첫 진행 줄을 쓸 때까지
        st = aj.get_state(jobs, "x")
        if st["status"] == "running" and st["stages"][0].get("frac", 0) > 0:
            break
        time.sleep(0.2)
    assert st["status"] == "running" and st["stages"][0]["frac"] == pytest.approx(1 / 9)
    with pytest.raises(aj.JobError):                                # 이미 돌고 있으면 두 번째 시작은 거절
        aj.start("x", jobs, data, out, required=[], min_free_gb=0)
    pid = st["pid"]
    aj.cancel("x", jobs)
    for _ in range(50):
        if not aj.alive(pid):
            break
        time.sleep(0.2)
    assert not aj.alive(pid) and aj.get_state(jobs, "x")["status"] == "cancelled"


def test_start_prechecks(tmp_path):
    jobs, data, out = setup(tmp_path)
    with pytest.raises(aj.JobError, match="모델 파일"):
        aj.start("x", jobs, data, out, required=[tmp_path / "없는파일.pt"], min_free_gb=0)
    with pytest.raises(aj.JobError, match="디스크"):
        aj.start("x", jobs, data, out, required=[], min_free_gb=10**9)
    with pytest.raises(aj.JobError, match="없는 영상"):
        aj.start("nope", jobs, data, out, required=[], min_free_gb=0)


def test_safe_stem():
    assert aj.safe_stem("경기 1 (최종).mp4") == "경기_1_최종"
    assert aj.safe_stem("../../evil.mp4") == "evil"
    with pytest.raises(aj.JobError):
        aj.safe_stem("movie.avi")                                   # mp4 만
    with pytest.raises(aj.JobError):
        aj.safe_stem(".mp4")                                        # 이름이 비면 거절


def test_cleanup_after_done(tmp_path):
    jobs, data, out = setup(tmp_path)
    for suf in ("tracked", "events", "ball"):
        (out / f"x_{suf}.mp4").write_bytes(b"big")
    (out / "x_tracks.csv").write_text("frame\n", encoding="utf-8")
    aj.run("x", jobs, data, out, stages=[stage("a", "print('ok')")])
    assert not list(out.glob("x_*.mp4")) and (out / "x_tracks.csv").exists()   # 큰 영상 3개만 지운다


# ---- API (업로드·작업) ----
import cv2  # noqa: E402
import numpy as np  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import app as viewer_app  # noqa: E402


def tiny_mp4(path: Path, n=30):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 48))
    for _ in range(n):
        vw.write(np.zeros((48, 64, 3), np.uint8))
    vw.release()
    return path.read_bytes()


def api(tmp_path, required=()):
    jobs, data, out = tmp_path / "jobs", tmp_path / "data", tmp_path / "outputs"
    for d in (data, out):
        d.mkdir(parents=True, exist_ok=True)
    return TestClient(viewer_app.create_app(out, data, jobs, required=list(required), min_free_gb=0)), jobs, data, out


def test_api_upload_happy(tmp_path):
    c, jobs, data, _ = api(tmp_path)
    body = tiny_mp4(tmp_path / "src.mp4")
    r = c.put("/api/upload", params={"name": "내 경기 (최종).mp4"}, content=body)
    assert r.status_code == 200 and r.json()["stem"] == "내_경기_최종" and r.json()["status"] == "uploaded" and r.json()["overall"] == 0
    assert (data / "uploads" / "내_경기_최종.mp4").read_bytes() == body and not list((data / "uploads").glob("*.part"))
    assert [j["stem"] for j in c.get("/api/jobs").json()] == ["내_경기_최종"]
    # 같은 이름은 409
    assert c.put("/api/upload", params={"name": "내 경기 (최종).mp4"}, content=body).status_code == 409


def test_api_upload_rejects_bad_files(tmp_path):
    c, _, data, _ = api(tmp_path)
    assert c.put("/api/upload", params={"name": "a.avi"}, content=b"x").status_code == 400
    r = c.put("/api/upload", params={"name": "garbage.mp4"}, content=("mp4가 아닌 아무 바이트" * 100).encode())
    assert r.status_code == 400 and "읽을 수 없습니다" in r.json()["detail"]
    assert not list((data / "uploads").glob("*"))  # 찌꺼기(.part 포함)가 안 남는다


def test_api_upload_name_clash_with_existing_match(tmp_path):
    c, _, data, out = api(tmp_path)
    (out / "match1_rallies.csv").write_text("start,serve_team,winner,contacts,seconds\n", encoding="utf-8")
    assert c.put("/api/upload", params={"name": "match1.mp4"}, content=tiny_mp4(tmp_path / "s.mp4")).status_code == 409


def test_api_start_errors_are_readable(tmp_path):
    c, _, _, _ = api(tmp_path, required=[tmp_path / "없는모델.pt"])
    c.put("/api/upload", params={"name": "x.mp4"}, content=tiny_mp4(tmp_path / "s.mp4"))
    r = c.post("/api/jobs/x/start")
    assert r.status_code == 400 and "모델 파일" in r.json()["detail"]
    assert c.post("/api/jobs/nope/start").status_code == 400
    assert c.post("/api/jobs/x/cancel").status_code == 400  # 돌고 있지 않으면 취소 불가


def test_api_uploaded_video_served_once_results_exist(tmp_path):
    c, _, data, out = api(tmp_path)
    c.put("/api/upload", params={"name": "x.mp4"}, content=tiny_mp4(tmp_path / "s.mp4"))
    assert c.get("/api/matches").json() == []                     # 결과가 없으면 경기가 아니다
    (out / "x_rallies.csv").write_text("start,serve_team,winner,contacts,seconds\n1,LEFT,RIGHT,3,5.0\n", encoding="utf-8")
    assert [m["stem"] for m in c.get("/api/matches").json()] == ["x"]  # 서버를 다시 안 띄워도 바로 목록에 뜬다
    assert c.get("/video/x", headers={"Range": "bytes=0-9"}).status_code == 206


# ---- 영상 삭제 · 대기열 ----
def test_delete_removes_only_exact_stem_files(tmp_path):
    jobs, data, out = setup(tmp_path)                                 # 'x' 가 이미 있다
    up = data / "uploads"
    aj.create(jobs, "x_b", up / "x_b.mp4")                            # 이름이 'x_' 로 시작하는 다른 영상
    for stem in ("x", "x_b"):
        (up / f"{stem}.mp4").write_bytes(b"v")
        (up / f"{stem}.courts_s10.json").write_text("[]", encoding="utf-8")
        for suf in ("tracks.csv", "rallies.csv", "analysis.png", "pipeline.log", "tracked.mp4"):
            (out / f"{stem}_{suf}").write_bytes(b"o")
    aj.delete("x", jobs, data, out)
    assert not list(up.glob("x.*")) and not list(out.glob("x_t*")) and not (out / "x_rallies.csv").exists()
    assert (up / "x_b.mp4").exists() and (out / "x_b_rallies.csv").exists() and aj.get_state(jobs, "x_b")  # 'x_b' 는 그대로
    assert aj.get_state(jobs, "x") is None


def test_delete_refuses_running_and_unknown(tmp_path):
    jobs, data, out = setup(tmp_path)
    st = aj.get_state(jobs, "x")
    st.update(status="queued")
    aj.write_state(jobs, st)
    with pytest.raises(aj.JobError, match="분석 중"):
        aj.delete("x", jobs, data, out)
    with pytest.raises(aj.JobError, match="없는 영상"):
        aj.delete("nope", jobs, data, out)


def test_queue_runs_next_after_first_finishes(tmp_path, monkeypatch):
    jobs, data, out = setup(tmp_path)
    (data / "uploads" / "y.mp4").write_bytes(b"fake")
    aj.create(jobs, "y", data / "uploads" / "y.mp4")
    monkeypatch.setenv("VB_JOB_STAGES_JSON", json.dumps([stage("s", "import time; time.sleep(2); print('ok')")]))
    assert aj.start("x", jobs, data, out, required=[], min_free_gb=0)["status"] == "running"
    q = aj.start("y", jobs, data, out, required=[], min_free_gb=0)    # x 가 도는 중 → 대기
    assert q["status"] == "queued" and q["queued_at"]
    for _ in range(100):                                             # x 가 끝나면 러너가 y 를 이어서 시작·완료
        if aj.get_state(jobs, "y")["status"] == "done":
            break
        time.sleep(0.3)
    assert aj.get_state(jobs, "x")["status"] == "done" and aj.get_state(jobs, "y")["status"] == "done"


def test_cancel_queued_returns_to_uploaded(tmp_path):
    jobs, data, out = setup(tmp_path)
    st = aj.get_state(jobs, "x")
    st.update(status="queued", queued_at=time.time())
    aj.write_state(jobs, st)
    assert aj.cancel("x", jobs)["status"] == "uploaded"


def test_kick_starts_oldest_queued_when_idle(tmp_path, monkeypatch):
    jobs, data, out = setup(tmp_path)
    (data / "uploads" / "y.mp4").write_bytes(b"fake")
    aj.create(jobs, "y", data / "uploads" / "y.mp4")
    monkeypatch.setenv("VB_JOB_STAGES_JSON", json.dumps([stage("s", "print('ok')")]))
    for stem, t in (("y", 200.0), ("x", 100.0)):                      # x 가 더 먼저 줄 섰다
        st = aj.get_state(jobs, stem)
        st.update(status="queued", queued_at=t)
        aj.write_state(jobs, st)
    aj.kick(jobs, data, out)
    for _ in range(100):
        if all(aj.get_state(jobs, s)["status"] == "done" for s in ("x", "y")):
            break
        time.sleep(0.3)
    assert [aj.get_state(jobs, s)["status"] for s in ("x", "y")] == ["done", "done"]


def test_api_delete_and_busy_queue(tmp_path):
    c, jobs, data, out = api(tmp_path)
    for name in ("a.mp4", "b.mp4"):
        assert c.put("/api/upload", params={"name": name}, content=tiny_mp4(tmp_path / "s.mp4")).status_code == 200
    st = aj.get_state(jobs, "a")
    st.update(status="running", pid=__import__("os").getpid())          # 이 테스트 프로세스를 '돌고 있는 러너'로 가장
    aj.write_state(jobs, st)
    r = c.post("/api/jobs/b/start")
    assert r.status_code == 200 and r.json()["status"] == "queued"
    assert c.delete("/api/jobs/a").status_code == 400                   # 도는 중이면 삭제 거절
    assert c.delete("/api/jobs/b").status_code == 400                   # 대기 중도 거절 (먼저 대기 취소)
    assert c.post("/api/jobs/b/cancel").json()["status"] == "uploaded"
    assert c.delete("/api/jobs/b").status_code == 200 and not (data / "uploads" / "b.mp4").exists()
    assert c.delete("/api/jobs/nope").status_code == 400
