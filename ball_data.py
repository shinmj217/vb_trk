"""공 검출 YOLO 데이터셋 (Perez et al. 2022 공 중심 레이블 → 고정 크기 박스)

    python ball_data.py     → data/ball/{images,labels}/{train,val,test}/  + data/ball/ball.yaml

- 이미지는 하드링크 (복사 없음, 같은 디스크). 이름 = 영상_프레임번호
- 레이블: 공 중심에 BOX px 정사각 박스 (공 지름 약 18px @1280x720). 0 0(안 보임) = 빈 레이블(배경)
- 시퀀스 41프레임 중 STEP 간격만 (연속 프레임은 거의 같음). 시퀀스끼리 겹치는 프레임은 한 번만
- split: 논문 공식 (train 24·val 15·test 16 영상). test는 전부 넣고 평가는 ball_eval.py
"""
import os
from pathlib import Path

from action_data import VAL_VIDEOS
from eval import TEST_VIDEOS

ROOT = Path("data/volleyball")
OUT = Path("data/ball")
BOX, W, H = 20, 1280, 720
STEP = {"train": 4, "val": 1, "test": 1}  # val 연속 프레임: 궤적 후처리 튜닝용 (학습 때는 10)


def ball_labels(v, fid):
    """시퀀스 하나 → {프레임 번호: (x, y) 또는 None}"""
    lines = (ROOT / "volleyball_ball_annotation" / str(v) / f"{fid}.txt").read_text().split("\n")
    out = {}
    for k, line in enumerate(l for l in lines if l.strip()):
        x, y = map(float, line.split()[:2])
        # 0 0 = 안 보임. 화면 밖으로 추정된 위치도 이 프레임엔 공이 없는 것으로
        out[fid - 20 + k] = (x, y) if 0 < x < W and 0 < y < H else None
    return out


def main():
    seen, n = set(), {}
    for v in sorted(int(d) for d in os.listdir(ROOT / "videos") if d.isdigit()):
        split = "test" if v in TEST_VIDEOS else "val" if v in VAL_VIDEOS else "train"
        for d in (OUT / "images" / split, OUT / "labels" / split):
            d.mkdir(parents=True, exist_ok=True)
        ann = ROOT / "volleyball_ball_annotation" / str(v)
        for f in sorted(ann.glob("*.txt"), key=lambda p: int(p.stem)):
            fid = int(f.stem)
            for k, (frame, xy) in enumerate(sorted(ball_labels(v, fid).items())):
                if k % STEP[split] or (v, frame) in seen:
                    continue
                src = ROOT / "videos" / str(v) / str(fid) / f"{frame}.jpg"
                if not src.exists():
                    continue
                seen.add((v, frame))
                name = f"{v}_{frame}"
                dst = OUT / "images" / split / f"{name}.jpg"
                if not dst.exists():
                    os.link(src, dst)
                lab = "" if xy is None else f"0 {xy[0] / W:.6f} {xy[1] / H:.6f} {BOX / W:.6f} {BOX / H:.6f}\n"
                (OUT / "labels" / split / f"{name}.txt").write_text(lab)
                n[split] = n.get(split, 0) + 1
    (OUT / "ball.yaml").write_text(f"path: {OUT.resolve().as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: ball\n")
    print(n)


if __name__ == "__main__":
    main()
