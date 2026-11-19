"""공 궤적 후처리: 앞뒤 프레임 후보와 이어지는 검출만 남기고, 짧은 구멍은 직선 보간

    python ball_track.py   → val(연속 프레임)에서 (거리 d, 구멍 gap, 임계값) 고르고 test 4px/10px F1
                             (ball_eval.py가 만든 후보 캐시 outputs/ball_preds_*.json 사용)

- 점수 = (자기 확신도 + 앞뒤 ±2프레임 안 d·|dt| px 이내 후보 중 최고 확신도) / 2
  → 혼자 튀는 오검출은 이웃 지지가 0이라 점수가 반으로, 공은 이웃이 받쳐줌. 프레임마다 점수 최고 후보 하나
- 임계값 넘은 점 사이 gap 프레임 이하 구멍은 직선 보간 (양 끝 거리 ≤ d·간격일 때만)
"""
import numpy as np

from ball_eval import CACHE, predict, score

DT = (-2, -1, 1, 2)


def segments(preds):
    """(이름 '영상_프레임', …) 목록 → 연속 프레임 구간마다 인덱스 목록"""
    key = sorted(range(len(preds)), key=lambda i: tuple(map(int, preds[i][0].split("_"))))
    out, prev = [], None
    for i in key:
        v, f = map(int, preds[i][0].split("_"))
        if prev != (v, f - 1):
            out.append([])
        out[-1].append(i)
        prev = (v, f)
    return out


def choose(cands, d):
    """구간 하나의 프레임별 후보 → 프레임별 (x, y, 점수) 또는 None"""
    n, out = len(cands), []
    for t, cs in enumerate(cands):
        best = None
        for x, y, c in cs:
            sup = max((c2 for dt in DT if 0 <= t + dt < n for x2, y2, c2 in cands[t + dt]
                       if np.hypot(x - x2, y - y2) <= d * abs(dt)), default=0.0)
            if best is None or (c + sup) / 2 > best[2]:
                best = (x, y, (c + sup) / 2)
        out.append(best)
    return out


def fill(pts, thr, d, gap):
    """임계값 아래 버리고, gap 이하 구멍은 양 끝 직선 보간 (보간 점 점수 = thr)"""
    out = [p if p is not None and p[2] >= thr else None for p in pts]
    last = None
    for t, p in enumerate(out):
        if p is None:
            continue
        if last is not None and 1 < t - last <= gap + 1 and np.hypot(p[0] - out[last][0], p[1] - out[last][1]) <= d * (t - last):
            x0, y0, _ = out[last]
            for u in range(last + 1, t):
                f = (u - last) / (t - last)
                out[u] = (x0 + f * (p[0] - x0), y0 + f * (p[1] - y0), thr)
        last = t
    return out


def run(preds, d, segs=None):
    """전체 예측 → 이름 순서 그대로 (정답, 선택 점) 목록 만들 함수 (thr, gap) → res"""
    segs = segs or segments(preds)
    chosen = [choose([preds[i][2] for i in s], d) for s in segs]

    def res(thr, gap):
        out = [None] * len(preds)
        for s, pts in zip(segs, chosen):
            for i, p in zip(s, fill(pts, thr, d, gap)):
                out[i] = (preds[i][1], p)
        return out
    return res


def main():
    val, test = (predict(None, s, cache=CACHE.format(s)) for s in ("val", "test"))
    vs, ts = segments(val), segments(test)
    best = (0, None)
    for d in (30, 60, 100):
        res = run(val, d, vs)
        for gap in (0, 2, 4):
            for thr in np.arange(0.02, 0.9, 0.02):
                f1 = score(res(thr, gap), 0, 4)["F1"]
                if f1 > best[0]:
                    best = (f1, (d, gap, thr))
    d, gap, thr = best[1]
    print(f"val 최적 d={d} gap={gap} thr={thr:.2f}: 4px F1 {best[0]:.3f}")
    res = run(test, d, ts)(thr, gap)
    for tol in (4, 10):
        s = score(res, 0, tol)
        print(f"test {len(res)}장 궤적 후처리, {tol}px: " + "  ".join(f"{k} {v:.3f}" for k, v in s.items()))


if __name__ == "__main__":
    main()
