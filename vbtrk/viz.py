import cv2
import numpy as np

from .court import COURT_L, COURT_W, FAR, NEAR, NET_Y

TEAM_COLOR = {FAR: (60, 60, 230), NEAR: (230, 140, 40), None: (160, 160, 160)}  # BGR
SIDE_COLOR = {"LEFT": (60, 60, 230), "RIGHT": (230, 140, 40), "?": (160, 160, 160)}  # BGR


def side_name(team, left_team=FAR):
    """코트 반쪽 팀(FAR/NEAR) → 화면 기준 LEFT/RIGHT (left_team: Court.left_team())"""
    return "?" if team is None else "LEFT" if team == left_team else "RIGHT"


def draw_court_overlay(frame, court):
    poly = court.image_corners.astype(np.int32)
    cv2.polylines(frame, [poly], True, (0, 255, 0), 2)
    net = court.to_image([[0, NET_Y], [COURT_W, NET_Y]]).astype(np.int32)
    cv2.line(frame, tuple(net[0]), tuple(net[1]), (0, 255, 255), 2)


def draw_players(frame, ids, boxes, results, left_team=FAR):
    for tid, box, (team, _, airborne) in zip(ids, boxes, results):
        x1, y1, x2, y2 = map(int, box)
        name = side_name(team, left_team)
        c = SIDE_COLOR[name]
        cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)
        label = f"#{int(tid)} {name}" + (" J" if airborne else "")
        cv2.putText(frame, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)
        cv2.circle(frame, ((x1 + x2) // 2, y2), 4, c, -1)


def make_minimap(ids, results, trails, scale=20, pad=40):
    """위에서 내려다본 코트. 1m = scale px. trails: {track_id: [(x, y), ...]} 지금까지의 이동 경로"""
    w = int(COURT_W * scale + 2 * pad)
    h = int(COURT_L * scale + 2 * pad)
    img = np.full((h, w, 3), (40, 90, 40), np.uint8)

    def P(x, y):
        return int(pad + x * scale), int(pad + y * scale)

    cv2.rectangle(img, P(0, 0), P(COURT_W, COURT_L), (255, 255, 255), 2)
    cv2.line(img, P(0, NET_Y), P(COURT_W, NET_Y), (0, 255, 255), 3)
    for y in (NET_Y - 3, NET_Y + 3):  # 어택 라인
        cv2.line(img, P(0, y), P(COURT_W, y), (220, 220, 220), 1)

    for tid, (team, pos, _) in zip(ids, results):
        if pos is None:
            continue
        c = TEAM_COLOR[team]
        trail = list(trails.get(tid, []))
        for a, b in zip(trail[:-1], trail[1:]):
            cv2.line(img, P(*a), P(*b), c, 1)
        cv2.circle(img, P(*pos), 7, c, -1)
        cv2.putText(img, str(int(tid)), (P(*pos)[0] + 8, P(*pos)[1] + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return img


def paste_minimap(frame, minimap, max_h_ratio=0.45, margin=10):
    fh, fw = frame.shape[:2]
    mh, mw = minimap.shape[:2]
    s = min(1.0, fh * max_h_ratio / mh)
    mm = cv2.resize(minimap, (int(mw * s), int(mh * s)))
    mh, mw = mm.shape[:2]
    frame[margin:margin + mh, fw - mw - margin:fw - margin] = mm
    return frame
