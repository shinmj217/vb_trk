"""공 검출 평가 (WASB 논문과 같은 방식): 프레임당 확신도 최고 1개만, 정답 중심과 tol px 이내면 맞음

    python ball_eval.py --model outputs/ball/y26s_1280/weights/best.pt

- TP: 예측 있음 + 공 보임 + 거리 ≤ tol / FP: 예측 있음 + (공 안 보임 또는 거리 > tol) / FN: 예측 없음 + 공 보임 / TN: 둘 다 없음
- 확신도 임계값은 val에서 F1(4px) 최대인 값 → test에 그대로 적용
- 참고 (WASB 논문 volleyball test, 4px): WASB F1 88.0, TrackNetV2 83.6
"""
import argparse
import json
from pathlib import Path

import numpy as np
from ultralytics import YOLO

W, H = 1280, 720
CACHE = "outputs/ball_preds_{}.json"   # 모델 바꾸면 지울 것


def predict(model, split, imgsz=1280, bs=16, k=5, cache=None):
    """split의 이미지마다 (이름, 정답 xy 또는 None, 후보 [(x, y, conf)] 확신도 내림차순 최대 k개). cache json 있으면 재사용"""
    if cache and Path(cache).exists():
        return [(s, tuple(g) if g else None, [tuple(c) for c in cs]) for s, g, cs in json.loads(Path(cache).read_text())]
    imgs = sorted(Path(f"data/ball/images/{split}").glob("*.jpg"))
    out = []
    for i in range(0, len(imgs), bs):
        for img, r in zip(imgs[i:i + bs], model.predict(imgs[i:i + bs], imgsz=imgsz, conf=0.01, max_det=k, verbose=False)):
            lab = (Path(f"data/ball/labels/{split}") / f"{img.stem}.txt").read_text().split()
            gt = (float(lab[1]) * W, float(lab[2]) * H) if lab else None
            xy, conf = r.boxes.xywh[:, :2].tolist(), r.boxes.conf.tolist()
            out.append((img.stem, gt, sorted(((x, y, c) for (x, y), c in zip(xy, conf)), key=lambda c: -c[2])))
    if cache:
        Path(cache).write_text(json.dumps(out))
    return out


def top1(preds):
    """predict 결과 → score용 (정답, 최고 확신도 후보)"""
    return [(gt, cs[0] if cs else None) for _, gt, cs in preds]


def best_thr(res, tol=4):
    return max(np.arange(0.01, 0.95, 0.01), key=lambda t: score(res, t, tol)["F1"])


def score(res, thr, tol):
    tp = fp = fn = tn = 0
    for gt, pred in res:
        p = pred if pred is not None and pred[2] >= thr else None
        if p is None:
            fn += gt is not None
            tn += gt is None
        elif gt is not None and np.hypot(p[0] - gt[0], p[1] - gt[1]) <= tol:
            tp += 1
        else:
            fp += 1
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return dict(F1=f1, P=prec, R=rec, Acc=(tp + tn) / max(len(res), 1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="outputs/ball/y26s_1280/weights/best.pt")
    a = p.parse_args()
    model = YOLO(a.model)
    val = top1(predict(model, "val", cache=CACHE.format("val")))
    thr = best_thr(val)
    print(f"val 최적 확신도 {thr:.2f}: 4px {score(val, thr, 4)}")
    test = top1(predict(model, "test", cache=CACHE.format("test")))
    for tol in (4, 10):
        s = score(test, thr, tol)
        print(f"test {len(test)}장, {tol}px: " + "  ".join(f"{k} {v:.3f}" for k, v in s.items()))


if __name__ == "__main__":
    main()
