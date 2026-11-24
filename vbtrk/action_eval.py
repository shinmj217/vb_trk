"""동작 분류 평가 (test = 원래 분포): 전체 정확도, 클래스별 재현율, 평균 클래스 정확도(MCA), 혼동 행렬

    python -m vbtrk.action_eval --model outputs/action_cls/base/weights/best.pt
"""
import argparse
from pathlib import Path

import numpy as np
from ultralytics import YOLO


def report(names, y_true, y_pred):
    k = len(names)
    cm = np.zeros((k, k), int)
    np.add.at(cm, (y_true, y_pred), 1)
    rec = cm.diagonal() / np.maximum(cm.sum(1), 1)
    print(f"전체 정확도 {cm.trace() / cm.sum():.3f}, 평균 클래스 정확도(MCA) {rec.mean():.3f}  (n={cm.sum()})")
    w = max(map(len, names))
    print(" " * (w + 8) + " ".join(f"{n[:5]:>5}" for n in names))
    for i, n in enumerate(names):
        print(f"{n:>{w}} {rec[i]:.2f}  " + " ".join(f"{c:5d}" for c in cm[i]))
    return cm, rec


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="outputs/action_cls/base/weights/best.pt")
    p.add_argument("--data", default="data/actions/test")
    a = p.parse_args()
    model = YOLO(a.model)
    names = [model.names[i] for i in range(len(model.names))]
    files, y_true = [], []
    for i, n in enumerate(names):
        fs = sorted(Path(a.data, n).glob("*.[jp][pn]g"))
        files += fs
        y_true += [i] * len(fs)
    y_pred = []
    for s in range(0, len(files), 256):
        y_pred += [int(r.probs.top1) for r in model.predict(files[s:s + 256], imgsz=224, verbose=False)]
    report(names, np.array(y_true), np.array(y_pred))


if __name__ == "__main__":
    main()
