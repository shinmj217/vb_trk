"""업로드한 영상의 분석 작업: 파이프라인 5단계를 서버와 분리된 프로세스로 돌리고, 진행 상황을 jobs/<이름>.json 에 남긴다.

    (서버가 알아서 호출) python analysis_jobs.py run <이름> --jobs jobs --data data --outputs outputs
상태: uploaded → running → done | failed | cancelled.  실패·취소 뒤에 다시 시작하면 끝난 단계는 건너뛴다.
서버를 닫거나 브라우저를 닫아도 러너 프로세스는 계속 돈다 (상태 파일이 진실의 원천).
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REQUIRED = [ROOT / "yolo26m.pt", ROOT / "outputs/ball/y26s_1280/weights/best.pt",
            ROOT / "outputs/action_cls/base/weights/best.pt", ROOT / "outputs/action_cls/frames_m/weights/best.pt"]
MIN_FREE_GB = 10
BIG_VIDEOS = ("tracked", "events", "ball")  # 다 끝나면 지우는 중간 영상 (뷰어는 안 쓰고, 영상당 수 GB)


class JobError(Exception):
    """사용자에게 그대로 보여 줄 수 있는 실패 사유"""


def stage_defs(video, outputs):
    """파이프라인 5단계. weight = 전체 진행률에서 차지하는 대략의 비중 (1·2·3세트 실행 시간을 보고 잡은 추정치),
    phases = 로그 진행 줄의 라벨 → 그 단계 안에서 차지하는 구간"""
    py, v, o = sys.executable, str(video), str(outputs)
    return [
        dict(name="court", label="코트 인식", weight=40, phases={"SAM3 코트": (0, 0.95)},
             argv=[py, "-u", "-c", f"from vbtrk.court_sam3 import video_courts; video_courts({v!r}, stride=10)"], env={}),
        dict(name="track", label="선수 추적", weight=25, phases={"추적": (0, 0.8), "저장": (0.8, 1.0)},
             argv=[py, "-u", "run.py", "--video", v, "--out", o], env={"CUDA_LAUNCH_BLOCKING": "1"}),  # 동기 모드: misaligned address 회피
        dict(name="events", label="이벤트 인식", weight=15, phases={"특징": (0, 0.7), "영상": (0.7, 1.0)},
             argv=[py, "-u", "events.py", "--video", v, "--out", o], env={}),
        dict(name="ball", label="공 추적", weight=17, phases={"공": (0, 0.85), "그리기": (0.85, 1.0)},
             argv=[py, "-u", "ball_video.py", "--video", v, "--out", o], env={}),
        dict(name="analysis", label="분석 그림", weight=3, phases={},
             argv=[py, "-u", "analysis.py", "--video", v, "--out", o], env={}),
    ]


# ---- 상태 파일 ----
def _path(jobs, stem):
    return Path(jobs) / f"{stem}.json"


def write_state(jobs, st):
    p = _path(jobs, st["stem"])
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, ensure_ascii=False), encoding="utf-8")
    for _ in range(40):  # Windows: 읽는 쪽이 파일을 잡고 있으면 잠깐 거절될 수 있다
        try:
            os.replace(tmp, p)
            return
        except PermissionError:
            time.sleep(0.05)
    os.replace(tmp, p)


def _read(jobs, stem):
    p = _path(jobs, stem)
    for _ in range(40):
        try:
            return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
        except (PermissionError, json.JSONDecodeError):
            time.sleep(0.05)
    return None


def alive(pid):
    if not pid:
        return False
    if os.name == "nt":  # os.kill(pid, 0) 은 Windows 에서 프로세스를 죽이므로 쓰면 안 된다
        import ctypes
        k = ctypes.windll.kernel32
        h = k.OpenProcess(0x1000, False, int(pid))  # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        ok = k.GetExitCodeProcess(h, ctypes.byref(code))
        k.CloseHandle(h)
        return bool(ok) and code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def kill_tree(pid):
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
    else:
        import signal
        os.killpg(os.getpgid(pid), signal.SIGTERM)


def get_state(jobs, stem):
    """상태를 읽는다. 돌고 있다고 적혀 있는데 프로세스가 없으면 '예기치 않게 종료'로 고쳐 둔다"""
    st = _read(jobs, stem)
    if st and st["status"] == "running" and not alive(st.get("pid")):
        st.update(status="failed", error="분석 프로세스가 예기치 않게 종료됐습니다 (다시 시작하면 끝난 단계는 건너뜁니다)")
        for s in st["stages"]:
            if s["status"] == "running":
                s["status"] = "failed"
        write_state(jobs, st)
    return st


def list_states(jobs):
    return [s for s in (get_state(jobs, p.stem) for p in sorted(Path(jobs).glob("*.json"))) if s]


def create(jobs, stem, video):
    Path(jobs).mkdir(parents=True, exist_ok=True)
    st = dict(stem=stem, video=str(video), status="uploaded", stage=None, error=None, pid=None, started=None, finished=None,
              stages=[dict(name=d["name"], label=d["label"], weight=d["weight"], status="pending", frac=0.0) for d in stage_defs(video, "outputs")])
    write_state(jobs, st)
    return st


# ---- 진행률 ----
def parse_progress(line, phases):
    """로그 한 줄 → 그 단계 안 진행률 0~1 (진행 줄이 아니면 None). '[진행] 라벨 a/b' 와 SAM3 의 'SAM3 코트 a/b'"""
    for label, (lo, hi) in phases.items():
        m = re.search(rf"{re.escape(label)}\s+(\d+)/(\d+)", line)
        if m and int(m[2]) > 0:
            return lo + (hi - lo) * min(1.0, int(m[1]) / int(m[2]))
    return None


def overall(st):
    tot = sum(s["weight"] for s in st["stages"])
    return sum(s["weight"] * (1.0 if s["status"] == "done" else s["frac"] if s["status"] == "running" else 0.0) for s in st["stages"]) / tot


# ---- 실행 ----
def safe_stem(filename):
    name = Path(str(filename).replace("\\", "/")).name
    if not name.lower().endswith(".mp4"):
        raise JobError("mp4 파일만 올릴 수 있습니다")
    stem = re.sub(r"[^0-9A-Za-z가-힣_-]+", "_", name[:-4]).strip("_")[:60]
    if not stem:
        raise JobError("영상 이름이 올바르지 않습니다")
    return stem


def _run_stage(sd, st, s, jobs, log):
    env = {**os.environ, **sd["env"], "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.Popen(sd["argv"], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
                            text=True, encoding="utf-8", errors="replace")
    tail, last = deque(maxlen=6), 0.0
    for line in proc.stdout:
        log.write(line)
        log.flush()
        if line.strip():
            tail.append(line.strip()[-200:])
        f = parse_progress(line, sd["phases"])
        if f is not None and time.time() - last > 1.0:
            s["frac"], last = f, time.time()
            write_state(jobs, st)
    return proc.wait(), list(tail)


def run(stem, jobs, data, outputs, stages=None):
    """러너: 끝나지 않은 단계를 차례로 돌린다. 실패하면 거기서 멈추고 사유를 남긴다"""
    st = get_state(jobs, stem) or {}
    stages = stages or stage_defs(st["video"], outputs)
    if [s["name"] for s in st["stages"]] != [d["name"] for d in stages]:  # (테스트용 가짜 단계 등) 단계 목록이 다르면 새로
        st["stages"] = [dict(name=d["name"], label=d["label"], weight=d["weight"], status="pending", frac=0.0) for d in stages]
    st.update(status="running", pid=os.getpid(), started=time.time(), finished=None, error=None)
    write_state(jobs, st)
    Path(outputs).mkdir(parents=True, exist_ok=True)
    with open(Path(outputs) / f"{stem}_pipeline.log", "a", encoding="utf-8") as log:
        for sd, s in zip(stages, st["stages"]):
            if s["status"] == "done":
                continue
            s.update(status="running", frac=0.0)
            st["stage"] = sd["name"]
            write_state(jobs, st)
            log.write(f"\n===== {sd['label']} ({time.strftime('%m-%d %H:%M:%S')}) =====\n")
            rc, tail = _run_stage(sd, st, s, jobs, log)
            if rc != 0:
                s["status"] = "failed"
                st.update(status="failed", error=f"{sd['label']} 단계 실패 (종료 코드 {rc}): " + " / ".join(tail[-3:]), finished=time.time())
                write_state(jobs, st)
                return st
            s.update(status="done", frac=1.0)
            write_state(jobs, st)
    for kind in BIG_VIDEOS:
        (Path(outputs) / f"{stem}_{kind}.mp4").unlink(missing_ok=True)
    st.update(status="done", stage=None, finished=time.time())
    write_state(jobs, st)
    return st


def start(stem, jobs, data, outputs, required=REQUIRED, min_free_gb=MIN_FREE_GB):
    """분리된 러너 프로세스를 띄운다 (서버가 죽어도 계속 돈다). 사유가 있으면 JobError"""
    st = get_state(jobs, stem)
    if st is None:
        raise JobError("없는 영상입니다")
    if any(s["status"] == "running" for s in list_states(jobs)):
        raise JobError("이미 분석 중인 영상이 있습니다 (GPU 때문에 한 번에 하나만 분석합니다)")
    missing = [Path(p).name for p in required if not Path(p).exists()]
    if missing:
        raise JobError("필요한 모델 파일이 없습니다: " + ", ".join(missing))
    if shutil.disk_usage(outputs if Path(outputs).exists() else ".").free < min_free_gb * 1e9:
        raise JobError(f"디스크 여유가 {min_free_gb}GB 미만입니다 (분석 중 중간 영상이 몇 GB 생깁니다)")
    argv = [sys.executable, str(ROOT / "analysis_jobs.py"), "run", stem, "--jobs", str(jobs), "--data", str(data), "--outputs", str(outputs)]
    kw = dict(cwd=ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, close_fds=True)
    if os.name == "nt":  # 숨은 콘솔 + 새 프로세스 그룹 (+ 서버가 속한 작업 객체에서 분리). 창이 안 뜨고, 서버를 닫아도 안 죽는다
        flags = 0x08000000 | 0x00000200
        try:
            proc = subprocess.Popen(argv, creationflags=flags | 0x01000000, **kw)
        except OSError:
            proc = subprocess.Popen(argv, creationflags=flags, **kw)
    else:
        proc = subprocess.Popen(argv, start_new_session=True, **kw)
    st.update(status="running", pid=proc.pid, error=None, started=time.time(), finished=None)
    write_state(jobs, st)
    return st


def cancel(stem, jobs):
    st = get_state(jobs, stem)
    if not st or st["status"] != "running":
        raise JobError("분석 중이 아닙니다")
    kill_tree(st["pid"])
    for s in st["stages"]:
        if s["status"] == "running":
            s.update(status="pending", frac=0.0)
    st.update(status="cancelled", stage=None, error=None, finished=time.time())
    write_state(jobs, st)
    return st


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["run"])
    p.add_argument("stem")
    p.add_argument("--jobs", default="jobs")
    p.add_argument("--data", default="data")
    p.add_argument("--outputs", default="outputs")
    a = p.parse_args()
    fake = os.environ.get("VB_JOB_STAGES_JSON")  # 테스트용: 가짜 단계 목록을 JSON 으로 주입
    run(a.stem, a.jobs, a.data, a.outputs, stages=json.loads(fake) if fake else None)


if __name__ == "__main__":
    main()
