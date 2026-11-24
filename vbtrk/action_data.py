"""동작 분류용 선수 크롭 데이터셋 만들기 (Volleyball Dataset 레이블 프레임, 정답 박스)

    python -m vbtrk.action_data      → data/actions/{train,val,test}/{동작}/*.jpg (ultralytics 분류 폴더 형식)
    python -m vbtrk.action_data 5    → data/actions_t5/...png : 시간축 입력. 채널 = 흑백 (t-5, t, t+5) 프레임
                                  → 움직임이 색 번짐으로 보임, 사전학습 3채널 모델 그대로 사용

- 크롭: 박스 중심 기준 정사각형 (변 = 긴 변 x 1.1). 분류 모델이 가운데를 정사각형으로 자르므로 미리 맞춤
- 불균형: train/val은 클래스당 최대 CAP장 (standing 줄임), train은 MIN장 미만이면 복사해서 채움. test는 원래 분포
"""
import os
import random
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import cv2

from vbtrk.eval import TEST_VIDEOS, load_annotations

VAL_VIDEOS = [0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51]  # 논문 공식 split
CAP, MIN = 3000, 1000


@lru_cache(maxsize=12)  # 시간축 추출이 한 장면에서 9프레임을 돌려 씀
def padded(path, pad=1000):
    return cv2.copyMakeBorder(cv2.imread(path), pad, pad, pad, pad, cv2.BORDER_CONSTANT)  # 화면 가장자리 선수


def square_crop(path, box, scale=1.1):
    return crop_padded(padded(path), box, scale)


def crop_padded(img, box, scale=1.1, pad=1000):
    """img: 사방 pad 만큼 패딩된 프레임. 박스 중심 정사각형 (변 = 긴 변 x scale)"""
    x1, y1, x2, y2 = box
    cx, cy, s = (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1) * scale / 2
    a, b = int(cx - s) + pad, int(cy - s) + pad
    return img[b:b + int(2 * s), a:a + int(2 * s)]


def clip_crop(frame_dir, fid, box, k):
    if k == 0:
        return square_crop(f"{frame_dir}/{fid}.jpg", box)
    # ponytail: 박스 고정 (정답은 가운데 프레임뿐). k가 크면 선수가 박스 밖으로 나감 → 트랙 박스 쓰면 해결
    return cv2.merge([cv2.cvtColor(square_crop(f"{frame_dir}/{f}.jpg", box), cv2.COLOR_BGR2GRAY)
                      for f in (fid - k, fid, fid + k)])


def collect(root="data/volleyball/videos"):
    """split -> action -> [(프레임 폴더, 레이블 프레임 번호, 박스)]"""
    items = defaultdict(lambda: defaultdict(list))
    for v in sorted(int(d) for d in os.listdir(root) if d.isdigit()):
        split = "test" if v in TEST_VIDEOS else "val" if v in VAL_VIDEOS else "train"
        for fid, _, boxes, acts in load_annotations(f"{root}/{v}/annotations.txt"):
            for box, act in zip(boxes, acts):
                items[split][act].append((f"{root}/{v}/{fid}", fid, box))
    return items


def main(k=0, root="data/volleyball/videos", seed=0):
    out, ext = ("data/actions", "jpg") if k == 0 else (f"data/actions_t{k}", "png")  # jpg는 색 채널을 뭉갬
    random.seed(seed)
    items = collect(root)
    for split, by_act in items.items():
        for act, lst in by_act.items():
            random.shuffle(lst)
            if split != "test":
                lst = lst[:CAP]
            reps = max(1, -(-MIN // len(lst))) if split == "train" else 1
            d = Path(out) / split / act
            d.mkdir(parents=True, exist_ok=True)
            for i, (frame_dir, fid, box) in enumerate(sorted(lst, key=lambda x: x[0])):  # 같은 프레임 연속 → 캐시
                crop = clip_crop(frame_dir, fid, box, k)
                for r in range(reps):
                    cv2.imwrite(str(d / f"{i}_{r}.{ext}"), crop)
            print(split, act, len(lst), "x", reps)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
