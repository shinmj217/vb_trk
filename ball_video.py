"""영상에서 공 추적: 공 검출(YOLO26s 1280) → 궤적 후처리(ball_track.py, val에서 고른 값)

    python ball_video.py --video data/long_20.mp4
출력: outputs/<영상>_ball.csv (frame, x, y, score; 공 없으면 빈칸), outputs/<영상>_ball.mp4 (최근 궤적 꼬리)
"""
import argparse
import csv
from pathlib import Path

import cv2
from ultralytics import YOLO

from ball_track import choose, fill

D, GAP, THR = 60, 4, 0.22   # ball_track.py val 튜닝 결과 (2026-11-15)
TAIL = 12


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--model", default="outputs/ball/y26s_1280/weights/best.pt")
    p.add_argument("--out", default="outputs")
    a = p.parse_args()
    stem, out = Path(a.video).stem, Path(a.out)
    model = YOLO(a.model)
    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cands, done = [], False
    while not done:  # 16프레임씩 읽고 버림 (경기 전체도 메모리 일정)
        batch = []
        while len(batch) < 16:
            ok, f = cap.read()
            if not ok:
                done = True
                break
            batch.append(f)
        for r in (model.predict(batch, imgsz=1280, conf=0.01, max_det=5, verbose=False) if batch else []):
            xy, conf = r.boxes.xywh[:, :2].tolist(), r.boxes.conf.tolist()
            cands.append(sorted(((x, y, c) for (x, y), c in zip(xy, conf)), key=lambda c: -c[2]))
    pts = fill(choose(cands, D), THR, D, GAP)

    with open(out / f"{stem}_ball.csv", "w", newline="", encoding="utf-8") as fo:
        w = csv.writer(fo)
        w.writerow(["frame", "x", "y", "score"])
        for t, q in enumerate(pts):
            w.writerow([t, *(f"{v:.1f}" for v in q[:2]), f"{q[2]:.3f}"] if q else [t, "", "", ""])
    cap.release()
    cap = cv2.VideoCapture(a.video)  # 그리기는 영상을 다시 읽으며
    wd, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vw = cv2.VideoWriter(str(out / f"{stem}_ball.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (wd, h))
    for t in range(len(pts)):
        ok, f = cap.read()
        if not ok:
            break
        tail = [q for q in pts[max(0, t - TAIL):t + 1] if q]
        for k, (x0, y0) in enumerate(zip(tail, tail[1:])):
            cv2.line(f, (int(x0[0]), int(x0[1])), (int(y0[0]), int(y0[1])), (0, 255, 255), 2)
        if pts[t]:
            cv2.circle(f, (int(pts[t][0]), int(pts[t][1])), 12, (0, 255, 255), 2)
        vw.write(f)
    vw.release()
    seen = sum(q is not None for q in pts)
    print(f"공 보임 {seen}/{len(pts)} 프레임 → {out / f'{stem}_ball.csv'}, {out / f'{stem}_ball.mp4'}")


if __name__ == "__main__":
    main()
