"""배구 선수 트래킹 v1

파이프라인: YOLO 사람 검출 → ByteTrack ID 유지 → 발 위치 호모그래피 → 프리존 밖(관중석) 제거
           → 코트 사이드로 팀 판정 (점프 보정) → 영상 전체를 본 뒤 코트 안에 들어온 적 있는 트랙만 선수
           → 영상 + 미니맵 + CSV 저장

사용 예:
    # 1) 코트 보정 (처음 한 번, 같은 카메라 앵글이면 재사용)
    python run.py --video data/rally.mp4 --calibrate
    # 2) 트래킹
    python run.py --video data/rally.mp4 --show
"""
import argparse
import csv
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from vbtrk.court import FAR, Court
from vbtrk.team import TeamAssigner, player_ids, relink
from vbtrk.viz import draw_court_overlay, draw_players, make_minimap, paste_minimap, side_name


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--court", default=None, help="수동 코트 보정 json (기본: <영상이름>_court.json, 없으면 SAM3 자동)")
    p.add_argument("--calibrate", action="store_true", help="코트 꼭짓점을 새로 찍기")
    p.add_argument("--calib-frame", type=int, default=0, help="보정에 쓸 프레임 번호")
    p.add_argument("--model", default="yolo26m.pt")
    p.add_argument("--imgsz", type=int, default=1280, help="먼 쪽 선수가 작으므로 크게")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--tracker", default="botsort.yaml")
    p.add_argument("--device", default=None, help="0 = GPU, cpu = CPU")
    p.add_argument("--out", default="outputs")
    p.add_argument("--show", action="store_true")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--margin-side", type=float, default=5.0, help="코트 밖 추적 범위(m), 기본 FIVB 프리존")
    p.add_argument("--margin-end", type=float, default=6.5)
    p.add_argument("--court-stride", type=int, default=10,
                   help="SAM3 코트 인식 간격(프레임). 사이는 보간. 1 = 매 프레임 (1시간 영상 약 30시간 → 10이면 약 3시간)")
    return p.parse_args()


def track_step(model, teams, frame, court, a):
    """한 프레임: 검출+추적 → 코트 밖 인원 제거 → 팀 판정. 코트 없는 장면(관중 클로즈업 등)은 건너뜀.
    a: tracker, imgsz, conf, device 속성. 반환 (ids, boxes, [(team, pos, airborne)])"""
    none = [], np.zeros((0, 4)), []
    if court is None:
        return none
    r = model.track(frame, persist=True, tracker=a.tracker, classes=[0],
                    imgsz=a.imgsz, conf=a.conf, device=a.device, verbose=False)[0]
    if r.boxes is None or r.boxes.id is None:
        return none
    boxes = r.boxes.xyxy.cpu().numpy()
    ids = r.boxes.id.cpu().numpy().astype(int)
    feet = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]], axis=1)
    cpts = court.to_court(feet)
    keep = court.in_play_area(cpts)          # 관중석 제거 (심판·감독은 player_ids가 트랙 단위로 거름)
    ids, boxes, cpts = ids[keep], boxes[keep], cpts[keep]
    return ids, boxes, teams.update(ids, boxes, cpts, court.side_of(cpts), court)


def main():
    a = parse_args()
    video = Path(a.video)
    court_path = Path(a.court) if a.court else video.with_name(video.stem + "_court.json")

    margins = dict(margin_side=a.margin_side, margin_end=a.margin_end)
    if a.calibrate:
        from vbtrk.calibrate import calibrate
        calibrate(str(video), str(court_path), a.calib_frame)
        print("보정 완료. 이제 --calibrate 없이 실행하세요.")
        return
    if court_path.exists():  # 수동 보정 파일이 있으면 고정 카메라로 보고 그대로 사용
        fixed = Court.load(court_path, **margins)
        courts = None
    else:                    # 기본: SAM3로 프레임마다 코트 검출 (중계 카메라 패닝 대응)
        from vbtrk.court_sam3 import video_courts
        corners, _ = video_courts(video, a.max_frames, stride=a.court_stride)
        if all(c is None for c in corners):
            raise SystemExit("SAM3가 코트를 한 프레임도 찾지 못했습니다. --calibrate 로 수동 보정하세요.")
        courts = [None if c is None else Court(c, **margins) for c in corners]  # None = 코트 안 보이는 장면

    model = YOLO(a.model)
    teams = TeamAssigner()

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_dir / f"{video.stem}_tracked.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    csv_f = open(out_dir / f"{video.stem}_tracks.csv", "w", newline="", encoding="utf-8")
    log = csv.writer(csv_f)
    log.writerow(["frame", "track_id", "team", "x1", "y1", "x2", "y2", "court_x", "court_y", "airborne"])

    # 1차: 영상 전체 추적
    frames, t0 = [], time.time()
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total = min(total, a.max_frames) if a.max_frames else total
    while True:
        ok, frame = cap.read()
        if not ok or (a.max_frames and len(frames) >= a.max_frames):
            break
        court = fixed if courts is None else courts[min(len(frames), len(courts) - 1)]
        frames.append(track_step(model, teams, frame, court, a))
        if len(frames) % 300 == 0:
            print(f"[진행] 추적 {len(frames)}/{total}", flush=True)  # analysis_jobs 가 읽는 진행 신호
    fps_track = len(frames) / (time.time() - t0)
    n_raw = len({int(t) for f in frames for t in f[0]})
    frames = relink(frames)
    players = player_ids(frames)
    print(f"추적 {len(frames)} 프레임, {fps_track:.1f} fps, 트랙 {n_raw}개 → 재연결 후 "
          f"{len({int(t) for f in frames for t in f[0]})}개, 그중 선수 {len(players)}개")

    # 2차: 선수 트랙만 저장/그리기
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    trails = defaultdict(lambda: deque(maxlen=60))
    for frame_idx, (ids, boxes, results) in enumerate(frames):
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % 300 == 0:
            print(f"[진행] 저장 {frame_idx}/{len(frames)}", flush=True)
        keep = [i for i, t in enumerate(ids) if int(t) in players]
        ids, boxes, results = [ids[i] for i in keep], [boxes[i] for i in keep], [results[i] for i in keep]
        for tid, (_, pos, _) in zip(ids, results):
            trails[int(tid)].append(tuple(pos))
        court = fixed if courts is None else courts[min(frame_idx, len(courts) - 1)]
        left = court.left_team() if court is not None else FAR  # 화면 왼쪽 팀 → CSV·영상에 LEFT/RIGHT

        for tid, box, (team, pos, air) in zip(ids, boxes, results):
            log.writerow([frame_idx, tid, side_name(team, left), *np.round(box, 1),
                          *np.round(pos, 2), int(air)])

        vis = frame.copy()
        if court is not None:
            draw_court_overlay(vis, court)
        draw_players(vis, ids, boxes, results, left)
        paste_minimap(vis, make_minimap(ids, results, trails))
        cv2.putText(vis, f"frame {frame_idx}  players {len(ids)}",
                    (20, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        writer.write(vis)

        if a.show:
            cv2.imshow("vb_trk", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    writer.release()
    csv_f.close()
    cv2.destroyAllWindows()
    print(f"완료 → {out_dir}")


if __name__ == "__main__":
    main()
