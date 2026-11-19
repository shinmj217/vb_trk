"""영상 첫 프레임에서 코트 꼭짓점 4개를 마우스로 찍어 저장한다.

클릭 순서: 먼쪽-왼쪽 → 먼쪽-오른쪽 → 가까운쪽-오른쪽 → 가까운쪽-왼쪽
조작: 왼쪽 클릭 = 점 추가, 오른쪽 클릭 = 마지막 점 취소, Enter = 저장, Esc = 취소
"""
import cv2
import numpy as np

from .court import Court

LABELS = ["far-left", "far-right", "near-right", "near-left"]
WIN = "calibrate (L-click: add, R-click: undo, Enter: save, Esc: quit)"


def read_frame(video_path, frame_idx=0):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"프레임을 읽을 수 없습니다: {video_path} @ {frame_idx}")
    return frame


def calibrate(video_path, out_path, frame_idx=0, max_width=1600):
    frame = read_frame(video_path, frame_idx)
    scale = min(1.0, max_width / frame.shape[1])
    disp_base = cv2.resize(frame, None, fx=scale, fy=scale)
    pts = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((x / scale, y / scale))
        elif event == cv2.EVENT_RBUTTONDOWN and pts:
            pts.pop()

    cv2.namedWindow(WIN)
    cv2.setMouseCallback(WIN, on_mouse)
    while True:
        disp = disp_base.copy()
        for i, (x, y) in enumerate(pts):
            p = (int(x * scale), int(y * scale))
            cv2.circle(disp, p, 6, (0, 0, 255), -1)
            cv2.putText(disp, LABELS[i], (p[0] + 8, p[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        if len(pts) == 4:
            poly = (np.array(pts) * scale).astype(np.int32)
            cv2.polylines(disp, [poly], True, (0, 255, 0), 2)
            msg = "Enter to save"
        else:
            msg = f"click: {LABELS[len(pts)]}"
        cv2.putText(disp, msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.imshow(WIN, disp)
        key = cv2.waitKey(20) & 0xFF
        if key == 27:
            cv2.destroyAllWindows()
            raise SystemExit("보정을 취소했습니다.")
        if key in (13, 10) and len(pts) == 4:
            break
    cv2.destroyAllWindows()

    court = Court(pts)
    court.save(out_path)
    print(f"코트 보정 저장: {out_path}")
    return court
