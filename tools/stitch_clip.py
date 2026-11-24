"""데이터셋 레이블 폴더(각 41프레임)를 이어 붙여 연속 영상 만들기: python tools/stitch_clip.py <video_id> <start> <end> <out.mp4>"""
import os, sys
import cv2

root = "data/volleyball/videos"
v, start, end, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
src = {}
for d in os.listdir(f"{root}/{v}"):
    if d.isdigit():
        for f in range(int(d) - 20, int(d) + 21):
            src.setdefault(f, f"{root}/{v}/{d}/{f}.jpg")
missing = [f for f in range(start, end + 1) if f not in src or not os.path.exists(src[f])]
assert not missing, missing[:10]
h, w = cv2.imread(src[start]).shape[:2]
wr = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
for f in range(start, end + 1):
    wr.write(cv2.imread(src[f]))
wr.release()
print(out, end - start + 1, "frames")
