"""특징 단계 시간축 동작 분류

1) 추출: 1장 기준선 분류기(컬러)로 레이블 프레임 ±K 구간에서 STEP 간격 프레임마다 선수 특징(1280차원) 뽑기
   → data/action_feats/{split}.npz (feats [N, T, 1280] fp16, labels [N])
2) 학습: 프레임별 특징 → 작은 1D conv 시간축 헤드 → 9종. val 평균 클래스 정확도로 최고 epoch 선택
3) 평가: test (원래 분포). 비교용으로 가운데 프레임 + 기준선 원래 분류층 결과도 같이 출력 (= 기준선 재현)

    python action_temporal.py
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO

from action_data import collect, square_crop
from action_eval import report

K, STEP = 8, 2  # ±8프레임, 2프레임 간격 → 9프레임
OFFSETS = list(range(-K, K + 1, STEP))
FEATS = "data/action_feats"  # 선수 특징 폴더 (--feats 로 바꿈: 모델 크기별)


@torch.no_grad()
def extract(model_path, out=None, bs=512):
    out = out or FEATS
    cls = YOLO(model_path).model.cuda().half().eval()
    names = [cls.names[i] for i in range(len(cls.names))]
    feat = {}
    cls.model[-1].linear.register_forward_hook(lambda m, inp, o: feat.__setitem__("x", inp[0]))

    def run(crops):
        x = np.stack([cv2.resize(c, (224, 224))[..., ::-1] for c in crops])  # BGR→RGB, /255 (ultralytics 분류 전처리와 같음)
        x = torch.from_numpy(x.copy()).cuda().half().permute(0, 3, 1, 2) / 255
        cls(x)
        return feat["x"].float().cpu().numpy().astype(np.float16)

    Path(out).mkdir(parents=True, exist_ok=True)
    for split, by_act in collect().items():
        path = Path(out) / f"{split}.npz"
        if path.exists():
            continue
        rows = sorted(((d, fid, box, names.index(act)) for act, lst in by_act.items() for d, fid, box in lst),
                      key=lambda r: (r[0], r[1]))  # 같은 장면끼리 → 프레임 캐시
        feats, labels, crops = [], [], []
        for i, (d, fid, box, y) in enumerate(rows):
            # ponytail: 박스 고정 (정답은 가운데 프레임뿐). 트랙 박스로 바꾸면 크게 움직이는 선수도 따라감
            crops += [square_crop(f"{d}/{fid + o}.jpg", box) for o in OFFSETS]
            labels.append(y)
            if len(crops) >= bs or i == len(rows) - 1:
                feats.append(run(crops).reshape(-1, len(OFFSETS), 1280))
                crops = []
            if i % 2000 == 0:
                print(split, i, "/", len(rows), flush=True)
        np.savez(path, feats=np.concatenate(feats), labels=np.array(labels), names=np.array(names))
    return cls.model[-1].linear.float()


class TemporalHead(nn.Module):
    def __init__(self, n_cls, d=1280, h=256, p=0.3):
        super().__init__()
        self.inp = nn.Sequential(nn.Dropout(p), nn.Linear(d, h), nn.ReLU())
        self.conv = nn.Sequential(nn.Conv1d(h, h, 3, padding=1), nn.ReLU(), nn.Conv1d(h, h, 3, padding=1), nn.ReLU())
        self.out = nn.Sequential(nn.Dropout(p), nn.Linear(2 * h, n_cls))

    def forward(self, x):  # x [B, T, d]
        z = self.inp(x)
        c = self.conv(z.transpose(1, 2))  # [B, h, T]
        return self.out(torch.cat([c.mean(2), c[:, :, c.shape[2] // 2]], 1))  # 구간 평균 + 가운데 프레임


# 전력분석용 5종: 공을 다루는 동작만 남기고 나머지는 other
MERGE5 = {"spiking": "spike", "setting": "set", "digging": "dig", "blocking": "block"}
NAMES5 = ["block", "dig", "other", "set", "spike"]


def merge_map(names):
    """9종 인덱스 → 5종 인덱스 텐서"""
    return torch.tensor([NAMES5.index(MERGE5.get(n, "other")) for n in names])


def load(split, out=None, merge=False):
    out = out or FEATS
    z = np.load(Path(out) / f"{split}.npz")
    x, y, names = torch.from_numpy(z["feats"].astype(np.float32)), torch.from_numpy(z["labels"]), list(z["names"])
    return (x, merge_map(names)[y], NAMES5) if merge else (x, y, names)


def video_ids(split):
    """npz 행 순서(추출 때와 같은 정렬)대로 영상 번호"""
    rows = sorted(((d, fid) for lst in collect()[split].values() for d, fid, _ in lst))
    return torch.tensor([int(Path(d).parent.name) for d, _ in rows])


def predict(model, x, bs=4096):
    model.eval()
    with torch.no_grad():
        return torch.cat([model(x[i:i + bs].cuda()).argmax(1).cpu() for i in range(0, len(x), bs)])


def train(epochs=40, lr=1e-3, seed=0, head_data="train", holdout=(33, 46, 49, 51), merge=False):
    """head_data='train': 기준선이 외운 train 특징으로 학습 (과적합 확인용)
    head_data='val': 기준선이 본 적 없는 val 영상으로 학습, 그중 holdout 영상으로 epoch 선택"""
    torch.manual_seed(seed)
    if head_data == "train":
        xtr, ytr, names = load("train", merge=merge)
        xva, yva, _ = load("val", merge=merge)
    else:
        x, y, names = load("val", merge=merge)
        hold = torch.isin(video_ids("val"), torch.tensor(holdout))
        xtr, ytr, xva, yva = x[~hold], y[~hold], x[hold], y[hold]
    n = len(names)
    w = 1 / torch.bincount(ytr, minlength=n).float().sqrt()  # 불균형: 빈도 제곱근 역수 가중
    model = TemporalHead(n).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    loss_fn = nn.CrossEntropyLoss(weight=(w / w.mean()).cuda())
    best, best_state = -1, None
    for ep in range(epochs):
        model.train()
        for idx in torch.randperm(len(xtr)).split(256):
            loss = loss_fn(model(xtr[idx].cuda()), ytr[idx].cuda())
            opt.zero_grad()
            loss.backward()
            opt.step()
        pv = predict(model, xva)
        mca = np.mean([(pv[yva == c] == c).float().mean().item() for c in range(n)])
        print(f"epoch {ep + 1} val 정확도 {(pv == yva).float().mean():.3f} 평균 클래스 {mca:.3f}", flush=True)
        if mca > best:
            best, best_state = mca, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, names


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="outputs/action_cls/base/weights/best.pt")
    p.add_argument("--feats", default="data/action_feats")
    p.add_argument("--out", default="outputs/action_cls/temporal_head.pt")
    p.add_argument("--head-data", default="train", choices=["train", "val"])
    p.add_argument("--merge", action="store_true", help="전력분석용 5종 (block/dig/set/spike/other)")
    a = p.parse_args()
    global FEATS
    FEATS = a.feats
    base_linear = extract(a.base)
    xte, yte, names = load("test")
    with torch.no_grad():
        base_pred = base_linear.cuda()(xte[:, len(OFFSETS) // 2].cuda()).argmax(1).cpu()
    if a.merge:  # 9종 기준선 예측을 5종으로 합쳐서 비교
        m = merge_map(names)
        yte, base_pred, names = m[yte], m[base_pred], NAMES5
    print("\n[기준선 재현] 가운데 프레임 + 원래 분류층")
    report(names, yte.numpy(), base_pred.numpy())
    model, _ = train(head_data=a.head_data, merge=a.merge)
    torch.save(model.state_dict(), a.out)
    print(f"\n[시간축 헤드] ±{K}프레임, {len(OFFSETS)}장")
    report(names, yte.numpy(), predict(model, xte).numpy())


if __name__ == "__main__":
    main()
