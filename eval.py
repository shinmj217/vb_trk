"""Volleyball Dataset 정량 평가: 테스트 영상에서 레이블 프레임(시퀀스 가운데)의 파이프라인 출력을 정답과 비교

- 선수 검출: 코트 밖 제거까지 거친 최종 박스 vs 정답 박스 (IoU>=0.5, 탐욕 매칭) → 재현율/정밀도
- 팀 판정: l-/r- 팀 행동 프레임에서 그 행동을 하는 선수(pass=digging, set=setting, spike=spiking)가
          맞는 팀(왼쪽=FAR, 오른쪽=NEAR)으로 나오는지

    python eval.py --per-video 5          # 테스트 16영상 × 5시퀀스
"""
import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from vbtrk.court import Court, FAR, NEAR

TEST_VIDEOS = [4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47]  # 논문 공식 split
ACTOR = {"pass": "digging", "set": "setting", "spike": "spiking"}


def load_annotations(path):
    """annotations.txt → [(frame_id, group('l-pass' 등, '_'는 '-'로 통일), gt_boxes xyxy, actions)]"""
    out = []
    for line in Path(path).read_text().splitlines():
        p = line.split()
        if not p:
            continue
        people = [p[i:i + 5] for i in range(2, len(p) - 4, 5)]
        boxes = np.float32([[float(x), float(y), float(x) + float(w), float(y) + float(h)] for x, y, w, h, _ in people])
        out.append((int(p[0].split(".")[0]), p[1].replace("_", "-"), boxes.reshape(-1, 4), [q[4] for q in people]))
    return out


def iou(a, b):
    """a: (n,4), b: (m,4) xyxy → (n,m)"""
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(-1)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / (area(a)[:, None] + area(b)[None] - inter + 1e-9)


def match(gt, pred, thr=0.5):
    """탐욕 IoU 매칭 → {gt_idx: pred_idx}"""
    m = {}
    if len(gt) and len(pred):
        u = iou(gt, pred)
        for g, q in sorted(np.argwhere(u >= thr), key=lambda gq: -u[gq[0], gq[1]]):
            if g not in m and q not in m.values():
                m[g] = q
    return m


def sequences(root, per_video):
    for v in TEST_VIDEOS:
        ann = load_annotations(root / str(v) / "annotations.txt")
        for k in np.linspace(0, len(ann) - 1, per_video).round().astype(int):
            yield v, *ann[k]


def frames_of(root, v, fid, before, after):
    return [root / str(v) / str(fid) / f"{f}.jpg" for f in range(fid - before, fid + after + 1)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data/volleyball/videos")
    p.add_argument("--per-video", type=int, default=5)
    p.add_argument("--before", type=int, default=10, help="레이블 프레임 앞 몇 프레임부터 추적 (최대 20)")
    p.add_argument("--after", type=int, default=10, help="레이블 프레임 뒤 몇 프레임까지 (선수 판정용, 최대 20)")
    p.add_argument("--model", default="yolo26m.pt")
    p.add_argument("--tracker", default="botsort.yaml")
    p.add_argument("--out", default="outputs/eval")
    a = p.parse_args()
    root, out = Path(a.root), Path(a.out)
    (out / "courts").mkdir(parents=True, exist_ok=True)
    seqs = list(sequences(root, a.per_video))

    # 1) SAM3 코트 (캐시). YOLO와 GPU 메모리를 나눠 쓰지 않도록 먼저 다 돌리고 내린다
    det = None
    cache_of = lambda v, fid: out / "courts" / f"{v}_{fid}_{a.before}_{a.after}.json"
    for v, fid, *_ in seqs:
        cache = cache_of(v, fid)
        if cache.exists():
            continue
        if det is None:
            from vbtrk.court_sam3 import Sam3Court
            det = Sam3Court()
        raw = []
        for f in frames_of(root, v, fid, a.before, a.after):
            court, dbg = det(cv2.imread(str(f)))
            raw.append("no_court" if dbg.get("no_court") else None if court is None else court.image_corners.tolist())
        cache.write_text(json.dumps(raw))
        print("SAM3", v, fid, sum(isinstance(c, list) for c in raw), "/", len(raw))
    if det is not None:
        del det
        import torch
        torch.cuda.empty_cache()

    # 2) 시퀀스마다 새 트래커로 파이프라인 실행 → 선수 트랙 판정 → 레이블 프레임만 채점
    from ultralytics import YOLO
    from run import track_step
    from vbtrk.court_sam3 import clean_courts
    from vbtrk.team import TeamAssigner, player_ids, relink
    cfg = SimpleNamespace(tracker=a.tracker, imgsz=1280, conf=0.25, device=None)
    detector = YOLO(a.model)  # 트래커와 상태를 섞지 않는 별도 검출기
    rows = []
    for v, fid, group, gt, actions in seqs:
        corners, _ = clean_courts(json.loads(cache_of(v, fid).read_text()))
        model, teams = YOLO(a.model), TeamAssigner()
        paths = frames_of(root, v, fid, a.before, a.after)
        frames = relink([track_step(model, teams, cv2.imread(str(f)), None if c is None else Court(c), cfg)
                         for f, c in zip(paths, corners)])
        players = player_ids(frames)
        ids, boxes, res = frames[a.before]
        tracked_hit = len(match(gt, np.float32(boxes).reshape(-1, 4)))  # 선수 판정 전 (트래커 단계 손실 구분용)
        keep = [i for i, t in enumerate(ids) if int(t) in players]
        boxes, res = np.float32(boxes)[keep].reshape(-1, 4), [res[i] for i in keep]
        m = match(gt, boxes)
        # 코트와 무관한 YOLO 자체 재현율 (코트 실패로 트래킹이 멈춘 것과 검출 실패를 구분)
        raw = detector.predict(cv2.imread(str(paths[a.before])), classes=[0], imgsz=cfg.imgsz, conf=cfg.conf, verbose=False)[0]
        yolo_hit = len(match(gt, raw.boxes.xyxy.cpu().numpy()))
        side, act = group.split("-")
        actors = [g for g, x in enumerate(actions) if x == ACTOR.get(act)]
        want = FAR if side == "l" else NEAR
        rows.append(dict(video=v, frame=fid, group=group, court=corners[a.before] is not None, ids=len(players),
                         gt=len(gt), pred=len(boxes),
                         hit=len(m), tracked_hit=tracked_hit, yolo_hit=yolo_hit, actors=len(actors), actors_found=sum(g in m for g in actors),
                         team_ok=sum(g in m and res[m[g]][0] == want for g in actors)))
        print(rows[-1])

    with open(out / "results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    s = {k: sum(r[k] for r in rows) for k in ("court", "gt", "pred", "hit", "tracked_hit", "yolo_hit", "actors", "actors_found", "team_ok")}
    print(f"\n시퀀스 {len(rows)}개, 레이블 프레임에 코트 있음 {s['court']}/{len(rows)}, "
          f"시퀀스당 선수 ID {np.mean([r['ids'] for r in rows]):.1f}개 (정답 선수 {s['gt'] / len(rows):.1f}명)")
    print(f"선수 검출 재현율 {s['hit'] / s['gt']:.3f} ({s['hit']}/{s['gt']}), 정밀도 {s['hit'] / max(s['pred'], 1):.3f}"
          f"  (선수 판정 전 {s['tracked_hit'] / s['gt']:.3f}, YOLO 단독 {s['yolo_hit'] / s['gt']:.3f})")
    print(f"동작 선수 검출 {s['actors_found']}/{s['actors']}, 그중 팀 정답 {s['team_ok']}/{s['actors_found']}"
          f" = {s['team_ok'] / max(s['actors_found'], 1):.3f}")


if __name__ == "__main__":
    main()
