"""영상 → 동작 이벤트: 언제, 어느 팀이, 누가 리시브(pass)/토스(set)/스파이크(spike) 했는지, 득점(winpoint)

입력: 영상 + run.py 결과 (outputs/<영상>_tracks.csv, outputs/<영상>_tracked.mp4)
    python events.py --video data/long_20.mp4
    python events.py --video data/long_20.mp4 --gt 20 49840   # 데이터셋 이어붙인 영상: 레이블 프레임에서 채점
출력: outputs/<영상>_events.csv, outputs/<영상>_events.mp4

1) 프레임마다 특징: 선수 크롭(트랙 박스) → 동작 분류기 s, 프레임 전체 → 팀 행동 분류기 m
2) 프레임 t마다 ±8프레임 창: 선수별 5종 확률(시간축 헤드) + 프레임 8종 확률 → 융합 분류기 → 팀 행동 8종
   팀 = run.py 의 LEFT/RIGHT (화면 왼쪽/오른쪽 코트 팀). 학습 때 박스 x로 반씩 나눴던 것을 코트 좌표 팀으로 대체
3) ±4프레임 평균으로 다듬고 같은 팀 행동이 이어지는 구간 = 이벤트 1개, 확률 최고 프레임에서 그 팀의 행위자 1명
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import joblib
import numpy as np
import torch
from ultralytics import YOLO

from action_data import crop_padded
from action_frames import square
from action_team import ACTOR5, GROUPS, scene_feat
from action_temporal import NAMES5, OFFSETS, TemporalHead
from eval import iou, load_annotations

M = "outputs/action_cls"
PAD = 1000


class Featurizer:
    """분류기 마지막 층 직전 특징(1280) 뽑기"""

    def __init__(self, path, size):
        self.m = YOLO(path).model.cuda().half().eval()
        self.size, self.out = size, {}
        self.m.model[-1].linear.register_forward_hook(lambda m, i, o: self.out.__setitem__("x", i[0]))

    @torch.no_grad()
    def __call__(self, imgs):
        if not imgs:
            return np.zeros((0, 1280), np.float32)
        x = np.stack([cv2.resize(im, (self.size, self.size))[..., ::-1] for im in imgs])
        self.m(torch.from_numpy(x.copy()).cuda().half().permute(0, 3, 1, 2) / 255)
        return self.out["x"].float().cpu().numpy()


def load_tracks(path):
    tracks = defaultdict(list)  # frame -> [(tid, team, box, court_xy)]
    for r in csv.DictReader(open(path, encoding="utf-8")):
        tracks[int(r["frame"])].append((int(r["track_id"]), r["team"], np.float32([r["x1"], r["y1"], r["x2"], r["y2"]]),
                                        (float(r["court_x"]), float(r["court_y"]))))
    return tracks


def crop_box(img, box, scale=1.1):
    """action_data.crop_padded 와 같은 크롭 (박스 중심 정사각형, 프레임 밖은 0)을 프레임 전체를 패딩하지 않고 만든다.
    예전엔 프레임마다 사방 1000px 패딩(약 48MB 복사)을 했고 그게 이 단계 CPU 시간 대부분이었다"""
    x1, y1, x2, y2 = box
    cx, cy, s = (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1) * scale / 2
    x0, y0, side = int(cx - s), int(cy - s), int(2 * s)
    out = np.zeros((side, side, 3), img.dtype)
    h, w = img.shape[:2]
    sx0, sy0, sx1, sy1 = max(x0, 0), max(y0, 0), min(x0 + side, w), min(y0 + side, h)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img[sy0:sy1, sx0:sx1]
    return out


def read_frames(video, depth=16):
    """영상 디코딩을 별도 스레드로 (cv2는 GIL을 놓음) → GPU가 디코딩을 기다리지 않는다"""
    import queue
    import threading
    q = queue.Queue(depth)

    def work():
        cap = cv2.VideoCapture(str(video))
        while True:
            ok, img = cap.read()
            q.put(img if ok else None)
            if not ok:
                break

    threading.Thread(target=work, daemon=True).start()
    while (img := q.get()) is not None:
        yield img


def features(video, tracks, max_frames=0):
    player, frame = Featurizer(f"{M}/base/weights/best.pt", 224), Featurizer(f"{M}/frames_m/weights/best.pt", 448)
    # ponytail: 특징을 전부 메모리에 (1시간 영상이면 수 GB) → 창 단위로 흘려보내기
    pf, ff = {}, []
    for img in read_frames(video):
        t = len(ff)
        if max_frames and t >= max_frames:
            break
        rows = tracks.get(t, [])
        for (tid, *_), f in zip(rows, player([crop_box(img, box) for _, _, box, _ in rows])):
            pf[(t, tid)] = f
        ff.append(frame([square(img)])[0])
    return pf, np.stack(ff)


@torch.no_grad()
def classify(tracks, pf, ff):
    """프레임마다 팀 행동 8종 확률, 선수별 5종 확률"""
    head5 = TemporalHead(len(NAMES5)).cuda().eval()
    head5.load_state_dict(torch.load(f"{M}/temporal_head5.pt"))
    headf = TemporalHead(len(GROUPS)).cuda().eval()
    headf.load_state_dict(torch.load(f"{M}/frames_head_m.pt"))
    fusion = joblib.load(f"{M}/team_fusion.joblib")
    assert fusion["cues"] == ("f5",), fusion["cues"]
    seen = defaultdict(list)  # 트랙별로 보인 프레임
    for t in sorted(tracks):
        for tid, *_ in tracks[t]:
            seen[tid].append(t)
    n = len(ff)
    g_prob, p5 = np.full((n, len(GROUPS)), np.nan), {}
    for t in sorted(tracks):
        rows = tracks[t]
        if not rows:
            continue
        win = []
        for tid, *_ in rows:  # 창 안에서 트랙이 안 보이는 프레임은 가장 가까운 보인 프레임으로
            s = np.array(seen[tid])
            win.append([pf[(int(s[np.abs(s - (t + o)).argmin()]), tid)] for o in OFFSETS])
        p = head5(torch.from_numpy(np.array(win)).cuda()).softmax(1).cpu().numpy()
        fw = ff[np.clip([t + o for o in OFFSETS], 0, n - 1)][None]
        fp = headf(torch.from_numpy(fw).cuda()).softmax(1).cpu().numpy()[0]
        left = np.array([team == "LEFT" for _, team, *_ in rows])
        X = np.r_[scene_feat(p, left), np.log(fp + 1e-6)][None]
        g_prob[t] = fusion["clf"].predict_proba(X)[0]
        p5[t] = p
    return g_prob, p5


def actor(rows, p, group):
    """팀 행동이 pass/set/spike면 그 팀에서 해당 동작 확률이 가장 높은 선수 (행 번호)"""
    side, act = group.split("-")
    if act not in ACTOR5:
        return None
    team = [i for i, (_, tm, *_) in enumerate(rows) if tm == ("LEFT" if side == "l" else "RIGHT")]
    return max(team, key=lambda i: p[i, NAMES5.index(ACTOR5[act])]) if team else None


def segment(g_prob, smooth=4, min_len=5):
    """±smooth 프레임 평균 확률 → 같은 팀 행동이 이어지는 구간 [(시작, 끝, 팀 행동, 평균 확률 배열)]"""
    n = len(g_prob)
    valid = ~np.isnan(g_prob[:, 0])
    sm = np.full_like(g_prob, np.nan)
    for t in np.where(valid)[0]:
        w = g_prob[max(0, t - smooth):t + smooth + 1]
        sm[t] = np.nanmean(w, 0)
    lab = np.where(valid, np.nan_to_num(sm).argmax(1), -1)
    segs, s = [], 0
    for t in range(1, n + 1):
        if t == n or lab[t] != lab[s]:
            if lab[s] >= 0 and t - s >= min_len:
                segs.append((s, t - 1, GROUPS[lab[s]]))
            s = t
    return segs, sm, lab


def merge_winpoints(events, gap, ahead):
    """득점 판정 거르기 (영상의 동작·시간만 사용):
    ① 득점 뒤 ahead 프레임 안에 리시브·토스·스파이크가 이어지면 랠리 중이라 득점이 아님
    ② gap 프레임 안에 모인 득점은 하나로: 시각 = 처음 것 (진짜 득점이 먼저, 세리머니·리플레이가 뒤따라 켜짐), 팀 = 확신도 최고
    1세트 점수판 정답 44점으로 맞춤 (gap 10초, ahead 3초): 찾은 득점 38→37, 가짜 19→9, 팀 34/38→32/37, 랠리 시작 서브 팀 33/42→34/40.
    ponytail: 한 경기로 맞춘 값, 앞뒤 5초 안에 진짜 득점 두 개가 붙는 경기(타임아웃 없는 연속)는 놓침. 다른 경기로 다시 확인"""
    keep = [e for e in events if e["action"] != "winpoint"]
    starts = sorted(e["start"] for e in keep)
    groups = []
    for e in sorted((e for e in events if e["action"] == "winpoint"), key=lambda e: e["peak"]):
        if any(0 < t - e["peak"] <= ahead for t in starts):
            continue
        if groups and e["peak"] - groups[-1][-1]["peak"] <= gap:
            groups[-1].append(e)
        else:
            groups.append([e])
    # 시각은 처음 것, 팀은 확신도 가장 높은 것 (세리머니 동안 양 팀 번갈아 켜져 처음 것은 팀이 자주 틀림)
    keep += [dict(max(g, key=lambda e: e["conf"]), start=g[0]["start"], end=g[0]["end"], peak=g[0]["peak"]) for g in groups]
    return sorted(keep, key=lambda e: e["start"])


def score_gt(v, start, n, tracks, p5, lab, root="data/volleyball/videos"):
    """이어붙인 데이터셋 영상: 레이블 프레임마다 팀 행동, 행위자(박스 IoU≥0.5) 채점"""
    ann = [a for a in load_annotations(f"{root}/{v}/annotations.txt") if start <= a[0] < start + n]
    acts9 = {"pass": "digging", "set": "setting", "spike": "spiking"}
    ok_g = ok_a = n_a = 0
    for fid, g, boxes, acts in ann:
        t = fid - start
        pred = GROUPS[lab[t]] if lab[t] >= 0 else "없음"
        ok_g += pred == g
        want = acts9.get(g.split("-")[1])
        gt_actor = [b for b, a in zip(boxes, acts) if a == want]
        if gt_actor:
            n_a += 1
            i = actor(tracks[t], p5[t], pred) if pred == g else None
            ok_a += i is not None and iou(np.array(gt_actor), tracks[t][i][2][None]).max() >= 0.5
        print(f"  프레임 {t:4d} 정답 {g:11s} 예측 {pred}")
    print(f"레이블 프레임 {len(ann)}개: 팀 행동 {ok_g}/{len(ann)}, 행위자 있는 {n_a}개 중 팀 행동+행위자 {ok_a}/{n_a}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--out", default="outputs")
    p.add_argument("--gt", nargs=2, type=int, help="데이터셋 영상 번호, 첫 프레임 번호")
    a = p.parse_args()
    stem, out = Path(a.video).stem, Path(a.out)
    tracks = load_tracks(out / f"{stem}_tracks.csv")
    pf, ff = features(a.video, tracks)
    g_prob, p5 = classify(tracks, pf, ff)
    segs, sm, lab = segment(g_prob)

    events = []
    for s, e, g in segs:
        peak = s + int(np.nanargmax(sm[s:e + 1, GROUPS.index(g)]))
        i = actor(tracks[peak], p5[peak], g)
        tid, cxy = (tracks[peak][i][0], tracks[peak][i][3]) if i is not None else ("", ("", ""))
        events.append(dict(start=s, end=e, peak=peak, group=g, team="LEFT" if g[0] == "l" else "RIGHT",
                           action=g.split("-")[1], conf=round(float(sm[peak, GROUPS.index(g)]), 3),
                           actor=tid, court_x=cxy[0], court_y=cxy[1]))
    fps = cv2.VideoCapture(a.video).get(cv2.CAP_PROP_FPS) or 30
    events = merge_winpoints(events, int(10 * fps), int(3 * fps))
    with open(out / f"{stem}_events.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, ["start", "end", "peak", "group", "team", "action", "conf", "actor", "court_x", "court_y"])
        w.writeheader()
        w.writerows(events)
    for ev in events:
        print(ev)

    # 영상: 추적 결과 위에 현재 이벤트 + 행위자 강조
    cap = cv2.VideoCapture(str(out / f"{stem}_tracked.mp4"))
    W, H, fps = int(cap.get(3)), int(cap.get(4)), cap.get(5) or 30
    wr = cv2.VideoWriter(str(out / f"{stem}_events.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    t = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        for ev in events:
            if ev["start"] <= t <= ev["end"]:
                txt = f"{ev['team']} {ev['action'].upper()}" + (f"  #{ev['actor']}" if ev["actor"] != "" else "")
                cv2.putText(img, txt, (W // 2 - 200, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 8)
                cv2.putText(img, txt, (W // 2 - 200, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 255), 3)
                for tid, _, box, _ in tracks.get(t, []):
                    if tid == ev["actor"]:
                        x1, y1, x2, y2 = box.astype(int)
                        cv2.rectangle(img, (x1 - 4, y1 - 4), (x2 + 4, y2 + 4), (0, 255, 255), 4)
        wr.write(img)
        t += 1
    wr.release()
    print(f"이벤트 {len(events)}개 → {out / f'{stem}_events.csv'}, {out / f'{stem}_events.mp4'}")
    if a.gt:
        score_gt(*a.gt, len(ff), tracks, p5, lab)


if __name__ == "__main__":
    main()
