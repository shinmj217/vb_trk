"""배구 코트 좌표계와 호모그래피.

코트 좌표계 (단위: m)
    x: 사이드라인 방향, 0 ~ 9
    y: 엔드라인 방향, 0(먼 쪽 엔드라인) ~ 18(가까운 쪽 엔드라인), 네트는 y = 9

카메라가 엔드라인 뒤에서 찍는 중계 앵글 기준으로
    y < 9  → 먼 쪽 팀 (FAR)
    y > 9  → 가까운 쪽 팀 (NEAR)
"""
import json

import cv2
import numpy as np

COURT_W = 9.0
COURT_L = 18.0
NET_Y = COURT_L / 2

# 클릭 순서와 같아야 함: 먼쪽-왼쪽, 먼쪽-오른쪽, 가까운쪽-오른쪽, 가까운쪽-왼쪽
COURT_CORNERS = np.array(
    [[0, 0], [COURT_W, 0], [COURT_W, COURT_L], [0, COURT_L]], dtype=np.float32
)

FAR, NEAR = 0, 1


class Court:
    def __init__(self, image_corners, margin_side=5.0, margin_end=6.5):
        """image_corners: 영상 위 코트 꼭짓점 4개 (COURT_CORNERS와 같은 순서)
        margin_side / margin_end: 코트 밖 어디까지 추적할지 (m). 기본 = FIVB 국제대회 프리존.
        관중석만 자르는 넓은 범위이고, 심판·감독은 team.player_ids가 트랙 단위로 거른다.
        """
        self.image_corners = np.asarray(image_corners, dtype=np.float32)
        self.H = cv2.getPerspectiveTransform(self.image_corners, COURT_CORNERS)
        self.H_inv = np.linalg.inv(self.H)
        self.margin_side = margin_side
        self.margin_end = margin_end

    # ---------- 저장 / 불러오기 ----------
    @classmethod
    def load(cls, path, **kw):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(data["image_corners"], **kw)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"image_corners": self.image_corners.tolist()}, f, indent=2)

    # ---------- 좌표 변환 ----------
    def to_court(self, pts):
        """영상 좌표 (N,2) → 코트 좌표 (N,2)"""
        pts = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
        if len(pts) == 0:
            return np.zeros((0, 2), dtype=np.float32)
        return cv2.perspectiveTransform(pts, self.H).reshape(-1, 2)

    def to_image(self, pts):
        pts = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
        if len(pts) == 0:
            return np.zeros((0, 2), dtype=np.float32)
        return cv2.perspectiveTransform(pts, self.H_inv).reshape(-1, 2)

    # ---------- 판정 ----------
    def left_team(self):
        """화면 왼쪽에 보이는 코트 반쪽의 팀 (FAR = y<9 쪽, NEAR = y>9 쪽). 출력은 LEFT/RIGHT로 쓴다"""
        a, b = self.to_image([[COURT_W / 2, NET_Y / 2], [COURT_W / 2, NET_Y * 1.5]])[:, 0]
        return FAR if a < b else NEAR

    def in_play_area(self, court_pts):
        """코트 + 여유 영역 안에 있는지. 관중, 벤치, 심판 대부분을 걸러낸다."""
        x, y = court_pts[:, 0], court_pts[:, 1]
        return (
            (x >= -self.margin_side)
            & (x <= COURT_W + self.margin_side)
            & (y >= -self.margin_end)
            & (y <= COURT_L + self.margin_end)
        )

    @staticmethod
    def side_of(court_pts):
        return np.where(court_pts[:, 1] < NET_Y, FAR, NEAR)
