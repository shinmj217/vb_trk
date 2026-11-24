"""프레임 전체로 팀 행동 8종 분류 (선수 크롭에 안 잡히는 단서: 세리머니, 모임, 카메라 움직임)

    python -m vbtrk.action_frames data      → data/team_frames/{train,val,test}/{팀 행동}/*.jpg
        가운데 프레임을 정사각형으로 패딩(양옆 잘림 방지). train은 좌우 뒤집은 사본 + l/r 레이블 교환
    yolo classify train model=yolo26s-cls.pt data=data/team_frames imgsz=448 fliplr=0 ...
    python -m vbtrk.action_frames feats s   → data/team_feats_s/{split}.npz : ±8프레임(9장) 특징 [장면, 9, 1280]
    python -m vbtrk.action_frames head s    → 시간축 헤드 학습(train), val 평균 클래스로 epoch 선택, test 출력
    (s/m = outputs/action_cls/frames_{s,m} 모델)
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO

from vbtrk.action_temporal import OFFSETS, TemporalHead, predict
from vbtrk.action_eval import report
from vbtrk.eval import TEST_VIDEOS, load_annotations
from vbtrk.action_data import VAL_VIDEOS

GROUPS = [f"{s}-{a}" for s in "lr" for a in ("pass", "set", "spike", "winpoint")]
SWAP = {g: ("r" if g[0] == "l" else "l") + g[1:] for g in GROUPS}
ROOT = "data/volleyball/videos"
MODEL = "outputs/action_cls/frames_{}/weights/best.pt"


def scene_list():
    """split -> [(프레임 폴더, 프레임 번호, 팀 행동)], action_team.scenes 와 같은 (폴더, 번호) 정렬"""
    out = {"train": [], "val": [], "test": []}
    for d in Path(ROOT).iterdir():
        if not d.name.isdigit():
            continue
        v = int(d.name)
        split = "test" if v in TEST_VIDEOS else "val" if v in VAL_VIDEOS else "train"
        for fid, g, _, _ in load_annotations(d / "annotations.txt"):
            out[split].append((f"{ROOT}/{v}/{fid}", fid, g))
    return {s: sorted(lst, key=lambda r: (r[0], r[1])) for s, lst in out.items()}


def square(img):
    h, w = img.shape[:2]
    t = (w - h) // 2
    return cv2.copyMakeBorder(img, t, w - h - t, 0, 0, cv2.BORDER_CONSTANT)


def make_data(out="data/team_frames"):
    for split, lst in scene_list().items():
        for i, (d, fid, g) in enumerate(lst):
            img = cv2.resize(square(cv2.imread(f"{d}/{fid}.jpg")), (448, 448))
            for flip in ((False, True) if split == "train" else (False,)):
                label, im = (SWAP[g], img[:, ::-1]) if flip else (g, img)
                p = Path(out) / split / label
                p.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(p / f"{i}_{int(flip)}.jpg"), im)
        print(split, len(lst))


@torch.no_grad()
def extract(size="s", bs=16):
    out = f"data/team_feats_{size}"
    cls = YOLO(MODEL.format(size)).model.cuda().half().eval()
    feat = {}
    cls.model[-1].linear.register_forward_hook(lambda m, inp, o: feat.__setitem__("x", inp[0]))
    names = [cls.names[i] for i in range(len(cls.names))]
    Path(out).mkdir(parents=True, exist_ok=True)
    for split, lst in scene_list().items():
        if (Path(out) / f"{split}.npz").exists():
            continue
        feats, buf = [], []
        for i, (d, fid, _) in enumerate(lst):
            buf += [cv2.resize(square(cv2.imread(f"{d}/{fid + o}.jpg")), (448, 448))[..., ::-1] for o in OFFSETS]
            if len(buf) >= bs * len(OFFSETS) or i == len(lst) - 1:
                x = torch.from_numpy(np.stack(buf).copy()).cuda().half().permute(0, 3, 1, 2) / 255
                cls(x)
                feats.append(feat["x"].float().cpu().numpy().astype(np.float16).reshape(-1, len(OFFSETS), 1280))
                buf = []
        y = np.array([names.index(g) for *_, g in lst])
        np.savez(Path(out) / f"{split}.npz", feats=np.concatenate(feats), labels=y, names=np.array(names))
        print(split, len(lst))


def load(split, size="s"):
    z = np.load(Path(f"data/team_feats_{size}") / f"{split}.npz")
    return torch.from_numpy(z["feats"].astype(np.float32)), torch.from_numpy(z["labels"]), list(z["names"])


def train_head(size="s", epochs=60, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    out = f"outputs/action_cls/frames_head_{size}.pt"
    xtr, ytr, names = load("train", size)
    xva, yva, _ = load("val", size)
    # 좌우 뒤집기 증강은 특징 단계에서 못 하므로 없음 (프레임 분류기 학습 때 이미 함)
    model = TemporalHead(len(names)).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    loss_fn = nn.CrossEntropyLoss()
    best, state = -1, None
    for ep in range(epochs):
        model.train()
        for idx in torch.randperm(len(xtr)).split(64):
            loss = loss_fn(model(xtr[idx].cuda()), ytr[idx].cuda())
            opt.zero_grad()
            loss.backward()
            opt.step()
        pv = predict(model, xva)
        mca = np.mean([(pv[yva == c] == c).float().mean().item() for c in range(len(names))])
        if mca > best:
            best, state = mca, {k: v.clone() for k, v in model.state_dict().items()}
            print(f"epoch {ep + 1} val 정확도 {(pv == yva).float().mean():.3f} 평균 클래스 {mca:.3f}", flush=True)
    model.load_state_dict(state)
    torch.save(state, out)
    xte, yte, _ = load("test", size)
    print("\n[프레임 1장 + 원래 분류층] test")
    lin = YOLO(MODEL.format(size)).model.model[-1].linear.float().cuda()
    with torch.no_grad():
        report(names, yte.numpy(), lin(xte[:, len(OFFSETS) // 2].cuda()).argmax(1).cpu().numpy())
    print("\n[프레임 시간축 헤드] test")
    report(names, yte.numpy(), predict(model, xte).numpy())


if __name__ == "__main__":
    {"data": make_data, "feats": extract, "head": train_head}[sys.argv[1]](*sys.argv[2:])
