"""SAM3 마스크 → 코트 라인 → 호모그래피 (옆에서 찍는 중계 앵글 기준).

코트 좌표는 court.py와 같다: x 0~9, y 0~18, 네트 y=9.
이 앵글에서는 x=0 위쪽(먼) 사이드라인, x=9 아래쪽(가까운) 사이드라인,
y=0 왼쪽 엔드라인, y=18 오른쪽 엔드라인 → FAR=왼쪽 팀, NEAR=오른쪽 팀.

1) 'volleyball court' 마스크의 볼록 껍질 → 위/아래 사이드라인, 화면 안에 보이는 엔드라인
2) 마스크 근처 흰 선 중 가파른 것(Hough) → 엔드/어택/센터라인 후보
3) 후보에 y값(0/6/9/12/18)을 붙이는 조합을 모두 시도: 'net post' 바닥점이 y≈9에 오고
   투영한 코트 라인이 흰 선과 가장 잘 겹치는 조합 선택
4) 가로선 × 사이드라인 교점 → findHomography, 투영한 코트 라인이 흰 선 위에 있는지로 검증
"""
import bisect
import json
import time
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np

from .court import COURT_CORNERS, COURT_W, NET_Y, Court

ATTACK = 3.0
LINE_YS = (0.0, NET_Y - ATTACK, NET_Y, NET_Y + ATTACK, 18.0)
COURT_LINES = [((0, 0), (0, 18)), ((9, 0), (9, 18))] + [((0, y), (9, y)) for y in (0, 6, 9, 12, 18)]


def _line(p, q):
    return np.cross([p[0], p[1], 1.0], [q[0], q[1], 1.0])


def _meet(l1, l2):
    x, y, w = np.cross(l1, l2)
    return None if abs(w) < 1e-9 else np.array([x / w, y / w])


def _angle(p, q):
    a = abs(np.degrees(np.arctan2(q[1] - p[1], q[0] - p[0])))
    return min(a, 180 - a)


def _clean(court_mask, post_mask):
    """코트 마스크에 섞인 네트/기둥 같은 얇은 돌출부 제거"""
    m = court_mask.astype(np.uint8) & ~cv2.dilate(post_mask.astype(np.uint8), np.ones((15, 15), np.uint8)) & 1
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))


def _court_color(frame, mask, tol=12):
    """SAM3 마스크는 코트 밖 바닥(프리존)까지 번지기도 한다 → 마스크 안에서 가장 많은 색상(H)만 남긴다.
    코트 바닥은 한 가지 색이고 프리존은 다른 색이라 둘을 가른다 (어택존이 더 진한 두 톤 코트도 색상은 같다)"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = mask.astype(bool)
    sel = mask & (hsv[..., 1] > 60)
    if sel.sum() < 1000:
        return mask
    hue = int(np.bincount(hsv[..., 0][sel], minlength=180).argmax())
    d = np.abs(hsv[..., 0].astype(int) - hue)
    colored = mask & (np.minimum(d, 180 - d) <= tol) & (hsv[..., 1] > 60)
    # 회색 코트 + 유채색 프리존(보라 등): 프리존이 코트를 둘러싸므로 (조각 전부의) 볼록 껍질이 더 작은 쪽이 코트.
    # 유니폼·라인 같은 작은 무채색 조각은 면적 조건(30%)으로 제외
    gray = mask & ~sel
    gray &= hsv[..., 2] < 235  # 흰 광고판·라인(V≈255) 제외, 회색 바닥은 V≈200
    if gray.sum() > 0.3 * colored.sum():
        area = []
        for m in (colored, gray):
            pts = np.argwhere(_clean(m, np.zeros_like(m)))[:, ::-1].astype(np.float32)
            area.append(cv2.contourArea(cv2.convexHull(pts)) if len(pts) > 2 else 0.0)
        if min(area) > 0:
            return gray if area[1] < area[0] else colored
    return colored


def _hull(m):
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    big = max(cv2.contourArea(c) for c in cnts)
    hull = cv2.convexHull(np.vstack([c for c in cnts if cv2.contourArea(c) > 0.1 * big]))  # 기둥 때문에 둘로 잘린 코트 합치기
    return cv2.approxPolyDP(hull, 0.01 * cv2.arcLength(hull, True), True).reshape(-1, 2).astype(float)


def _hull_lines(poly, shape):
    """껍질 변 중 화면 테두리를 따라가는 변은 버리고, 가장 긴 위/아래 가로 변 = 사이드라인"""
    h, w = shape

    def border(p):
        return {s for s, ok in (("l", p[0] < 10), ("t", p[1] < 10), ("r", p[0] > w - 11), ("b", p[1] > h - 11)) if ok}

    cy = poly[:, 1].mean()
    top = bot = None
    for p, q in zip(poly, np.roll(poly, -1, axis=0)):
        L = np.hypot(*(q - p))
        if L < 40 or border(p) & border(q):
            continue
        if _angle(p, q) < 25:
            slot = "top" if (p[1] + q[1]) / 2 < cy else "bot"
            cur = top if slot == "top" else bot
            if cur is None or L > cur[0]:
                if slot == "top":
                    top = (L, p, q)
                else:
                    bot = (L, p, q)
    if top is None or bot is None:
        return None, None
    return _line(top[1], top[2]), _line(bot[1], bot[2])


def _white(frame):
    """밝고 무채색 + 주변보다 밝은 것(top-hat). 밝은 회색 코트는 바닥 전체가 '흰색'이라 라인만 남긴다"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    L = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)[..., 0]
    lift = cv2.morphologyEx(L, cv2.MORPH_TOPHAT, np.ones((25, 25), np.uint8)) > 25
    return ((hsv[..., 2] > 170) & (hsv[..., 1] < 70) & lift).astype(np.uint8) * 255


def _support(near, p, q, n=60):
    """선분 p-q 위 샘플 중 흰 픽셀 비율. near = 흰 픽셀 맵을 몇 px 넓힌 것. 화면 안 샘플이 적으면 None"""
    h, w = near.shape
    pts = np.linspace(p, q, n).round().astype(int)
    pts = pts[(pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)]
    if len(pts) < 10:
        return None
    return float((near[pts[:, 1], pts[:, 0]] > 0).mean())


def _court_support(near, court, skip_net=False):
    """투영한 코트 라인 위 흰 픽셀 비율 평균. skip_net: 센터라인(네트·기둥에 가려 거의 안 보임) 제외"""
    sup = [_support(near, *court.to_image([p, q])) for p, q in COURT_LINES if not (skip_net and p[1] == q[1] == NET_Y)]
    sup = [s for s in sup if s is not None]
    return float(np.mean(sup)) if sup else 0.0


def _explained(cands, court, tol=15):
    """검출된 흰 선 후보 중 투영한 코트 가로선(0/6/9/12/18) 어느 하나와 겹치는 비율"""
    proj = [court.to_image([(0, y), (COURT_W, y)]) for y in LINE_YS]
    hit = [any(np.linalg.norm(pt - a) + np.linalg.norm(pb - b) < 2 * tol for a, b in proj) for pt, pb in cands]
    return float(np.mean(hit)) if hit else 0.0


def _court_from(lines):
    """{코트 y: (위 사이드라인 교점, 아래 사이드라인 교점)} → Court"""
    img = np.float32([p for y in lines for p in lines[y]])
    crt = np.float32([pt for y in lines for pt in ((0, y), (COURT_W, y))])
    H, _ = cv2.findHomography(img, crt, 0)
    if H is None or abs(np.linalg.det(H)) < 1e-12:
        return None
    corners = cv2.perspectiveTransform(COURT_CORNERS.reshape(-1, 1, 2), np.linalg.inv(H)).reshape(-1, 2)
    try:
        return Court(corners)
    except (cv2.error, np.linalg.LinAlgError):
        return None


def _refine(white, l, xr, bands=(20, 6)):
    """직선 l 근처(band px) 흰 픽셀로 다시 피팅. 껍질 변은 마스크 경계라 실제 라인에서 조금 비켜 있다"""
    ys, xs = np.nonzero(white)
    keep = (xs >= xr[0]) & (xs <= xr[1])
    xs, ys = xs[keep].astype(np.float32), ys[keep].astype(np.float32)
    for band in bands:
        a, b, c = l
        sel = np.abs(a * xs + b * ys + c) / np.hypot(a, b) < band
        if sel.sum() < 100:
            return l
        vx, vy, x0, y0 = cv2.fitLine(np.stack([xs[sel], ys[sel]], 1), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        l = _line((x0, y0), (x0 + vx, y0 + vy))
    return l


def _white_lines(white, mask, top, bot, poly, min_cover=0.35):
    """코트 안 흰색 가파른 선분 → 사이드라인과의 교점 (pt, pb) 묶음"""
    # 코트 마스크는 바깥 흰 라인을 포함하지 않으므로 넉넉히 넓혀서 찾는다
    white = white & cv2.dilate(mask.astype(np.uint8), np.ones((21, 21), np.uint8)) * 255
    segs = cv2.HoughLinesP(white, 1, np.pi / 180, 40, minLineLength=40, maxLineGap=8)
    cand = []
    for x1, y1, x2, y2 in ([] if segs is None else segs.reshape(-1, 4)):
        mid = (float(x1 + x2) / 2, float(y1 + y2) / 2)
        if _angle((x1, y1), (x2, y2)) < 35 or cv2.pointPolygonTest(poly.astype(np.float32), mid, True) < -15:
            continue
        l = _line((x1, y1), (x2, y2))
        pt, pb = _meet(l, top), _meet(l, bot)
        if pt is not None and pb is not None:
            cand.append((pt, pb, np.hypot(x2 - x1, y2 - y1)))
    # 사이드라인 사이 전체 길이 중 흰 픽셀이 충분히 덮인 것만 (흰 유니폼 다리 같은 짧은 선 제거)
    near = cv2.dilate(white, np.ones((7, 7), np.uint8))
    return [(pt, pb) for pt, pb in _cluster(cand) if (_support(near, pt, pb) or 0) >= min_cover]


def _cluster(cand, gap=20, min_len=60):
    cand = sorted(cand, key=lambda c: c[1][0])
    groups, out = [], []
    for c in cand:
        if groups and c[1][0] - groups[-1][-1][1][0] < gap:
            groups[-1].append(c)
        else:
            groups.append([c])
    for g in groups:
        wts = np.array([c[2] for c in g])
        if wts.sum() >= min_len:
            out.append((np.average([c[0] for c in g], axis=0, weights=wts),
                        np.average([c[1] for c in g], axis=0, weights=wts)))
    return out


def _post_foot(post_mask):
    ys, xs = np.nonzero(post_mask)
    if len(ys) == 0:
        return None
    bottom = ys >= ys.max() - 5
    return np.array([xs[bottom].mean(), ys.max()], float)


def _net_line(post_mask, min_frac=0.15):
    """센터라인(y=9) 직선 + 가까운/먼 기둥 바닥점.
    기둥 두 개가 따로 보이면 두 바닥점을 잇는다. 옆 앵글에선 두 기둥과 네트가 한 덩어리(네트 평면을 거의 옆에서 봄)
    → 그 덩어리의 주축 방향으로 가까운 기둥 바닥점을 지나는 직선, 먼 점은 덩어리 반대쪽 끝"""
    n, lab, st, _ = cv2.connectedComponentsWithStats(post_mask.astype(np.uint8))
    if n < 2:
        return None, []
    order = sorted(range(1, n), key=lambda i: -st[i, cv2.CC_STAT_AREA])
    big = st[order[0], cv2.CC_STAT_AREA]
    comps = [i for i in order[:2] if st[i, cv2.CC_STAT_AREA] >= min_frac * big]
    if len(comps) == 2:
        feet = sorted((_post_foot(lab == i) for i in comps), key=lambda f: -f[1])
        return _line(*feet), feet
    m = lab == comps[0]
    ys, xs = np.nonzero(m)
    vx, vy, _, _ = cv2.fitLine(np.stack([xs, ys], 1).astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).ravel()
    foot = _post_foot(m)
    top = np.array([xs[ys.argmin()], ys.min()], float)
    far = foot + np.array([vx, vy]) * np.dot(top - foot, [vx, vy])
    return _line(foot, far), [foot, far]


def _side(l, p):
    return float(np.dot(l, [p[0], p[1], 1.0]))


def _dir_diff(u, v):
    """두 방향 사이 각도 (0~90도, 방향 부호 무시)"""
    c = abs(np.dot(u, v)) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9)
    return float(np.degrees(np.arccos(min(c, 1.0))))


def _icp(court, white, radii=(30, 20, 12, 8)):
    """대략 맞춘 코트를 흰 라인에 붙이기: 코트 라인 위 점 → 가장 가까운 흰 픽셀(거리 변환) → RANSAC 호모그래피, 반경 줄이며 반복"""
    dist, lab = cv2.distanceTransformWithLabels(255 - white, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    pos = np.zeros((lab.max() + 1, 2), np.float32)  # 라벨 → 그 라벨의 흰 픽셀 좌표
    wy, wx = np.nonzero(white)
    pos[lab[wy, wx]] = np.stack([wx, wy], 1)
    h, w = white.shape
    model = np.float32([np.array(p) + (np.array(q) - np.array(p)) * s
                        for p, q in COURT_LINES for s in np.linspace(0, 1, 60)])
    for r in radii:
        img = court.to_image(model)
        ok = (img[:, 0] >= 0) & (img[:, 0] < w) & (img[:, 1] >= 0) & (img[:, 1] < h)
        xi, yi = img[ok].astype(int).T
        close = dist[yi, xi] < r
        if close.sum() < 12:
            break
        tgt = pos[lab[yi[close], xi[close]]]
        H, _ = cv2.findHomography(tgt, model[ok][close], cv2.RANSAC, 4.0)
        if H is None:
            break
        try:  # RANSAC이 퇴화한 호모그래피(역행렬 없음)를 낼 수 있음: 3세트 영상에서 8시간짜리 실행이 이걸로 죽었다
            corners = cv2.perspectiveTransform(COURT_CORNERS.reshape(-1, 1, 2), np.linalg.inv(H)).reshape(-1, 2)
            court = Court(corners)
        except (cv2.error, np.linalg.LinAlgError):
            break
    return court


def fit_court_edges(frame, court_mask, post_mask, min_support=0.45):
    """보이는 변 4개로 코트 맞추기 (앵글 무관):
    코트 영역(색 구분) 볼록 껍질의 변 중 화면 테두리가 아닌 것 → 네트 선(기둥 바닥 2개 = 센터라인 y=9)을
    가로지르는 두 변 = 사이드라인 (먼 기둥 쪽 x=0, 가까운 쪽 x=9), 안 가로지르는 가장 긴 변 = 보이는 엔드라인
    (화면 왼쪽 = y=0, 오른쪽 = 18). 흰 라인으로 다듬은 뒤 교점 4~6개 → 호모그래피, 투영 라인-흰 선 일치로 검증"""
    dbg = {"ok": False, "method": "edges"}
    court_mask = _clean(_court_color(frame, court_mask), post_mask)
    poly = _hull(court_mask)
    net, feet = _net_line(post_mask)
    h, w = court_mask.shape
    dbg.update(hull=poly, foot=feet[0] if feet else None, feet=feet,
               no_court=poly is None or cv2.contourArea(poly.astype(np.float32)) < 0.03 * h * w)
    if dbg["no_court"]:
        return None, dbg
    white = _white(frame)
    # 네트 선 후보: 기둥 바닥점으로 만든 것 + 가까운 기둥 바닥을 지나는 흰 센터라인 (기둥 하나만 보이거나
    # 기둥 마스크가 네트·심판과 뭉쳐 방향이 틀릴 때). 둘 다 맞춰 보고 흰 선과 더 잘 맞는 쪽
    cands = [(net, feet)] if net is not None else []
    if feet:
        alt = _net_from_white(white, poly, feet[0])
        if alt is not None:
            cands.append(alt)
    best = (None, dbg)
    for net, feet in cands:
        court, d = _fit_edges_net(frame, white, poly, net, feet, dict(dbg, foot=feet[0], feet=feet), min_support)
        if d.get("support", 0) > best[1].get("support", 0):
            best = (court, d)
    return best


def _net_from_white(white, poly, foot, max_d=30):
    """가까운 기둥 바닥 근처를 지나며 코트를 가로지르는 가장 긴 흰 선 = 센터라인 → (선, [바닥, 반대쪽 끝])"""
    m = np.zeros_like(white)
    cv2.fillConvexPoly(m, poly.astype(np.int32), 255)
    m = cv2.dilate(m, np.ones((31, 31), np.uint8))
    segs = cv2.HoughLinesP(white & m, 1, np.pi / 180, 60, minLineLength=80, maxLineGap=15)
    best = None
    for x1, y1, x2, y2 in ([] if segs is None else segs.reshape(-1, 4)):
        l = _line((x1, y1), (x2, y2))
        if abs(_side(l, foot)) / np.hypot(*l[:2]) > max_d:
            continue
        L = np.hypot(x2 - x1, y2 - y1)
        if best is None or L > best[0]:
            far = max(((x1, y1), (x2, y2)), key=lambda q: np.hypot(q[0] - foot[0], q[1] - foot[1]))
            best = (L, l, np.array(far, float))
    if best is None:
        return None
    return best[1], [np.asarray(foot, float), best[2]]


def _fit_edges_net(frame, white, poly, net, feet, dbg, min_support):
    h, w = white.shape

    def border(p):  # 화면 테두리 40px 안이면 어느 변인지
        return {s for s, ok in (("l", p[0] < 40), ("t", p[1] < 40), ("r", p[0] > w - 41), ("b", p[1] > h - 41)) if ok}

    sides, ends = [], []
    for p, q in zip(poly, np.roll(poly, -1, axis=0)):
        L = np.hypot(*(q - p))
        if L < 40 or border(p) & border(q):  # 같은 테두리를 따라가는 변 = 화면 끝이지 코트 라인 아님
            continue
        (sides if _side(net, p) * _side(net, q) < 0 else ends).append((L, p, q))
    if len(sides) != 2:
        return None, dbg
    xr = (poly[:, 0].min(), poly[:, 0].max())
    # 먼 기둥(feet[1])에 가까운 변 = 먼 사이드라인 x=0
    sides.sort(key=lambda s: abs(_side(_line(s[1], s[2]), feet[1])) / np.hypot(*_line(s[1], s[2])[:2]))
    far, nearl = (_refine(white, _line(s[1], s[2]), xr) for s in sides)
    lines = {}
    left = np.array([-1e4, (feet[0][1] + feet[1][1]) / 2])  # 화면 왼쪽 먼 점: 네트 기준 어느 쪽인지 판정용
    for L, p, q in sorted(ends, key=lambda e: -e[0]):
        if min(_dir_diff(q - p, s[2] - s[1]) for s in sides) < 15:
            continue  # 사이드라인과 거의 평행 = 사이드라인 조각
        y = 0.0 if _side(net, (p + q) / 2) * _side(net, left) > 0 else 18.0
        if y in lines:
            continue
        el = _refine(white, _line(p, q), (min(p[0], q[0]) - 30, max(p[0], q[0]) + 30))
        lines[y] = (_meet(el, far), _meet(el, nearl))
    lines[NET_Y] = (_meet(net, far), _meet(net, nearl))
    if len(lines) < 2 or any(v is None for pair in lines.values() for v in pair):
        return None, dbg
    # 변 하나가 틀리면 전체가 휘므로 2개 이상 조합마다 맞춰 보고 흰 선과 가장 잘 겹치는 것
    near = cv2.dilate(white, np.ones((7, 7), np.uint8))
    best = (0.0, None, {})
    for k in range(len(lines), 1, -1):
        for ys in combinations(sorted(lines), k):
            sub = {y: lines[y] for y in ys}
            raw = _court_from(sub)
            if raw is None:
                continue
            # ICP는 센터라인이 네트에 가려지면 길이 방향으로 미끄러질 수 있다(3m 라인에 붙음) → ICP 전/후 둘 다,
            # 가까운 기둥 바닥이 센터라인(y=9)에서 1.5m 넘게 벗어나면 버림
            for court in (_icp(raw, white), raw):
                if abs(court.to_court([feet[0]])[0, 1] - NET_Y) > 1.5:
                    continue
                s = _court_support(near, court, skip_net=True)
                if s > best[0] + 0.02:
                    best = (s, court, sub)
    support, court, used = best
    dbg.update(lines=used, support=support, ok=bool(court is not None and support >= min_support), court=court)
    return (court if dbg["ok"] else None), dbg


def fit_court(frame, court_mask, post_mask, min_support=1.3):
    """프레임 + SAM3 마스크 → (Court 또는 None, 디버그 dict). 보이는 변 방식 먼저, 안 되면 안쪽 라인 라벨 방식"""
    court, dbg = fit_court_edges(frame, court_mask, post_mask)
    if court is not None:
        return court, dbg
    return fit_court_lines(frame, court_mask, post_mask, min_support)


def fit_court_lines(frame, court_mask, post_mask, min_support=1.3):
    """안쪽 라인에 y 라벨(0/6/9/12/18)을 붙이는 조합 탐색 (데이터셋 옆 앵글용)"""
    dbg = {"ok": False}
    court_mask = _clean(_court_color(frame, court_mask), post_mask)
    poly = _hull(court_mask)
    foot = _post_foot(post_mask)
    # 코트가 거의 안 보이면(관중 클로즈업, 리플레이 등) '코트 없음' → 이 프레임을 넘어서 보간하지 않는다
    h, w = court_mask.shape
    dbg.update(hull=poly, foot=foot, no_court=poly is None or cv2.contourArea(poly.astype(np.float32)) < 0.03 * h * w)
    if dbg["no_court"] or foot is None:
        return None, dbg
    top, bot = _hull_lines(poly, court_mask.shape)
    if top is None:
        return None, dbg
    white = _white(frame)
    xr = (poly[:, 0].min(), poly[:, 0].max())
    top, bot = _refine(white, top, xr), _refine(white, bot, xr)

    near = cv2.dilate(white, np.ones((7, 7), np.uint8))
    cands = _white_lines(white, court_mask, top, bot, poly)
    cands = sorted(sorted(cands, key=lambda w: -_support(near, *w))[:6], key=lambda w: w[1][0])
    dbg.update(top=top, bot=bot, cands=cands)

    # 후보 선(왼→오)에 라벨(0/6/9/12/18, 증가 순)을 붙이는 모든 조합 중
    # 네트 기둥 바닥이 센터라인 근처에 오고, 투영한 코트 라인이 흰 선과 가장 잘 겹치는 것
    best = (-1.0, None, {})
    for k in range(2, min(len(cands), len(LINE_YS)) + 1):
        for idx in combinations(range(len(cands)), k):
            for ys in combinations(LINE_YS, k):
                lines = dict(zip(ys, (cands[i] for i in idx)))
                court = _court_from(lines)
                if court is None or abs(court.to_court([foot])[0][1] - NET_Y) > 0.6:
                    continue
                score = _court_support(near, court) + _explained(cands, court)
                if score + 0.01 * k > best[0] + 0.01 * len(best[2]):
                    best = (score, court, lines)
    support, court, lines = best
    dbg.update(lines=lines, support=max(support, 0.0), ok=bool(support >= min_support), court=court)  # support = 라인 일치 + 후보 설명 비율 (최대 2)
    return (court if dbg["ok"] else None), dbg


def draw_fit(frame, court, dbg):
    """디버그: 검출한 선(노랑), 기둥 바닥(빨강), 투영한 코트 라인(초록)"""
    vis = frame.copy()
    if dbg.get("hull") is not None:
        cv2.polylines(vis, [dbg["hull"].astype(np.int32)], True, (255, 0, 0), 2)
    for pt, pb in dbg.get("lines", {}).values():
        cv2.line(vis, tuple(map(int, pt)), tuple(map(int, pb)), (0, 255, 255), 1)
    if dbg.get("foot") is not None:
        cv2.circle(vis, tuple(map(int, dbg["foot"])), 6, (0, 0, 255), -1)
    fitted = dbg.get("court")
    if fitted is not None:  # 통과 = 초록, 검증 실패 = 빨강
        for p, q in COURT_LINES:
            a, b = fitted.to_image([p, q])
            cv2.line(vis, tuple(map(int, a)), tuple(map(int, b)), (0, 255, 0) if dbg["ok"] else (0, 0, 255), 2)
    cv2.putText(vis, f"lines {sorted(dbg.get('lines', {}))}  support {dbg.get('support', 0):.2f}  ok {dbg['ok']}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return vis


# ---------- 영상 단위: SAM3 결과 캐시 + 탈락 프레임 보간 ----------
NO_COURT = "no_court"  # 캐시 표기: 코트가 안 보이는 프레임
ANCHORS = np.float32([[0, 6], [COURT_W, 6], [0, 12], [COURT_W, 12]])  # 화면에 거의 항상 보이는 코트 중앙부


def drop_outliers(corners, window=8, max_px=40, trusted=None):
    """검증은 통과했지만 카메라 궤적에서 벗어난 프레임을 None으로 (중계 카메라는 부드럽게 움직인다).
    라벨을 잘못 붙인 프레임이 40% 가까이 섞이기도 해서 RANSAC으로 판단:
    ±window 안의 검출 프레임 두 개로 선형 궤적을 세우는 모든 조합 중 가장 많은 프레임이 max_px 안에 드는 것을 고르고,
    그 궤적에서 벗어나면 버린다. 이웃이 2개 미만이면 판단하지 않고 둔다.
    trusted[i] 가 참인 프레임(흰 라인과 아주 잘 맞는 맞춤)은 궤적에서 벗어나도 버리지 않는다: 위에서 내려다보는 리플레이
    카메라는 빠르게·비선형으로 움직여 선형 궤적 가정에 걸리지만 맞춤 자체는 맞다 (눈으로 확인)."""
    ok = [i for i, c in enumerate(corners) if c is not None]
    anchor = {i: Court(corners[i]).to_image(ANCHORS) for i in ok}
    out = list(corners)
    for i in ok:
        nb = [j for j in ok if abs(j - i) <= window]
        if len(nb) < 3:
            continue
        best = set()
        for a, b in combinations(nb, 2):
            inl = {j for j in nb
                   if np.abs(anchor[j] - (anchor[a] + (anchor[b] - anchor[a]) * (j - a) / (b - a))).max() <= max_px}
            if len(inl) > len(best):
                best = inl
        if i not in best and not (trusted is not None and trusted[i]):
            out[i] = None
    return out


def fill_gaps(corners, max_gap=10):
    """앞뒤로 코트를 인식한 프레임 사이의 짧은 공백(max_gap 프레임 이하)만 코트 중앙부 영상 위치 선형 보간으로 채운다.
    긴 공백(관중석·선수 클로즈업 등)과 영상 앞뒤 끝은 None 그대로 → 그 구간은 트래킹을 멈춘다."""
    ok = [i for i, c in enumerate(corners) if c is not None]
    anchor = {i: Court(corners[i]).to_image(ANCHORS) for i in ok}
    out = list(corners)
    for i, c in enumerate(corners):
        if c is not None:
            continue
        k = bisect.bisect(ok, i)
        if k == 0 or k == len(ok) or ok[k] - ok[k - 1] - 1 > max_gap:
            continue
        a, b = ok[k - 1], ok[k]
        t = (i - a) / (b - a)
        pts = (1 - t) * anchor[a] + t * anchor[b]
        H = cv2.getPerspectiveTransform(np.float32(pts), ANCHORS)
        out[i] = cv2.perspectiveTransform(COURT_CORNERS.reshape(-1, 1, 2), np.linalg.inv(H)).reshape(-1, 2)
    return out


def video_courts(video, max_frames=0, cache=None, stride=1):
    """영상 전체 프레임의 코트 꼭짓점(영상 좌표, 화면 밖일 수 있음, 코트 없는 프레임은 None) 리스트와 SAM3 검출 성공 여부.
    SAM3 결과는 <영상>.courts.json 에 캐시 (프레임당 ~1초라 다시 돌리지 않도록).
    stride: SAM3를 stride 프레임마다만 돌리고 사이는 보간 (1시간 영상: 1 → 약 30시간, 10 → 약 3시간)"""
    full = Path(cache or Path(video).with_suffix(".courts.json"))
    if full.exists():  # 매 프레임 결과가 이미 있으면 그게 제일 정확하고 공짜
        stride = 1
    cache = full if stride == 1 else full.with_name(full.name.replace(".courts.json", f".courts_s{stride}.json"))
    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n = min(n, max_frames) if max_frames else n
    raw = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else None
    # 영상 메타 프레임 수가 실제로 읽히는 프레임 수보다 조금 클 수 있음 (유튜브 구간 등) → 1% 안 차이는 캐시 그대로 씀
    if raw is None or len(raw) < 0.99 * n:
        det = Sam3Court()
        raw, due, calls = [], 0, 0
        t = time.time()
        for i in range(n):
            ok, frame = cap.read()
            if not ok:
                break
            if i < due and i != n - 1:  # 건너뛴 프레임 = 모름(None) → clean_courts 가 보간 (마지막 프레임은 항상 검출)
                raw.append(None)
                continue
            try:
                court, dbg = det(frame)
            except Exception as e:  # 긴 실행이 프레임 하나 때문에 죽지 않게: 그 프레임은 검출 실패로
                print(f"프레임 {i} 코트 검출 오류 → 실패 처리: {type(e).__name__}: {e}")
                court, dbg = None, {}
            calls += 1
            raw.append(NO_COURT if dbg.get("no_court") else None if court is None else court.image_corners.tolist())
            due = i + (2 if raw[-1] is None else stride)  # 검출 실패면 곧 다시 시도 (공백이 길어져 보간 못 하는 것 방지)
            if i % (20 * stride) == 0:
                print(f"SAM3 코트 {i}/{n}  {(time.time() - t) / (i + 1):.2f}s/frame")
        if stride > 1:  # 2차: 보간으로 못 메운 짧은 구멍(튀는 값 제거로 키프레임이 빠진 곳 등)만 촘촘히 다시
            todo = _holes(raw, stride)
            for i in todo:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ok, frame = cap.read()
                if ok:
                    try:
                        court, dbg = det(frame)
                    except Exception as e:
                        print(f"프레임 {i} 코트 검출 오류 → 실패 처리: {type(e).__name__}: {e}")
                        court, dbg = None, {}
                    raw[i] = NO_COURT if dbg.get("no_court") else None if court is None else court.image_corners.tolist()
            print(f"SAM3 1차 {calls}프레임 + 구멍 보강 {len(todo)}프레임 / 전체 {n} ({(calls + len(todo)) / n:.0%})")
        cache.write_text(json.dumps(raw), encoding="utf-8")
        del det
        import torch
        torch.cuda.empty_cache()
    cap.release()
    return clean_courts(raw[:n], stride, _supports(video, raw[:n], cache))


def _supports(video, raw, cache):
    """검출된 프레임마다 맞춤이 흰 라인과 얼마나 겹치는지 (센터라인 제외). 영상을 한 번 훑으므로 <캐시>.support.json 에 저장"""
    path = cache.with_name(cache.name.replace(".json", ".support.json"))
    if path.exists():
        sup = json.loads(path.read_text(encoding="utf-8"))
        if len(sup) == len(raw):
            return sup
    cap = cv2.VideoCapture(str(video))
    sup = [None] * len(raw)
    for i, c in enumerate(raw):
        if isinstance(c, list):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, frame = cap.read()
            if ok:
                near = cv2.dilate(_white(frame), np.ones((7, 7), np.uint8))
                sup[i] = float(_court_support(near, Court(np.float32(c)), skip_net=True) or 0.0)
    cap.release()
    path.write_text(json.dumps(sup), encoding="utf-8")
    return sup


def _holes(raw, stride, max_len=None):
    """stride 검출 + 정리 후에도 코트가 비는 구간 중 짧은 것(max_len 이하, '코트 없음' 표시 없음)에서
    아직 SAM3를 안 돌린 프레임을 2프레임 간격으로. 긴 구간은 진짜 코트 없는 장면으로 보고 건너뜀 (비용 제한)"""
    import contextlib
    import io
    max_len = max_len or 4 * stride
    with contextlib.redirect_stdout(io.StringIO()):
        corners, _ = clean_courts(raw, stride)
    todo, i, n = [], 0, len(raw)
    while i < n:
        if corners[i] is not None:
            i += 1
            continue
        j = i
        while j < n and corners[j] is None:
            j += 1
        if j - i <= max_len and NO_COURT not in raw[i:j]:
            todo += [k for k in range(i, j, 2) if raw[k] is None]
        i = j
    return todo


TRUST = 0.85  # 흰 라인 일치도(센터라인 제외)가 이 이상이면 궤적 필터에서 안 버림


def clean_courts(raw, stride=1, supports=None):
    """SAM3 프레임별 결과(꼭짓점 / None / NO_COURT) → (corners, detected).
    '코트 없음' 프레임에서 끊어 구간별로 튀는 값 제거 + 보간. 코트 없음 프레임은 None 그대로"""
    corners = [None if c in (None, NO_COURT) else np.float32(c) for c in raw]
    detected = [False] * len(raw)
    start = 0
    for end in [i for i, c in enumerate(raw) if c == NO_COURT] + [len(raw)]:
        # stride 프레임마다만 검출했으면 창·공백 허용도 그만큼 넓힌다 (이웃 검출 수를 비슷하게)
        tr = None if supports is None else [x is not None and x >= TRUST for x in supports[start:end]]
        seg = drop_outliers(corners[start:end], window=max(8, 4 * stride), max_px=40 if stride == 1 else 60, trusted=tr)
        detected[start:end] = [c is not None for c in seg]
        corners[start:end] = fill_gaps(seg, max_gap=max(10, 2 * stride))
        start = end + 1
    n_cut = raw.count(NO_COURT)
    print(f"코트 검출 {sum(isinstance(c, list) for c in raw)}/{len(raw)} 프레임, 튀는 값 제거 후 {sum(detected)}, "
          f"코트 없음 {n_cut}, 보간 포함 트래킹 {sum(c is not None for c in corners)} (나머지는 트래킹 멈춤)")
    return corners, detected


class Sam3Court:
    """SAM3로 코트/네트 기둥 마스크를 뽑아 fit_court 실행"""

    def __init__(self, court_thr=0.3, post_thr=0.4):
        import os
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        import torch
        from transformers import Sam3Model, Sam3Processor
        self.torch = torch
        self.model = Sam3Model.from_pretrained("facebook/sam3", torch_dtype=torch.float16).cuda().eval()
        self.proc = Sam3Processor.from_pretrained("facebook/sam3")
        self.court_thr, self.post_thr = court_thr, post_thr

    def mask(self, rgb, prompt, thr):
        inputs = self.proc(images=rgb, text=prompt, return_tensors="pt").to("cuda")
        inputs["pixel_values"] = inputs["pixel_values"].half()
        with self.torch.no_grad():
            out = self.model(**inputs)
        r = self.proc.post_process_instance_segmentation(
            out, threshold=thr, mask_threshold=0.5, target_sizes=inputs["original_sizes"].tolist())[0]
        m = np.zeros(rgb.shape[:2], bool)
        for x in r["masks"]:
            m |= x.cpu().numpy().astype(bool)
        return m

    def __call__(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        post = self.mask(rgb, "net post", self.post_thr)
        court, dbg = fit_court(frame, self.mask(rgb, "volleyball court", self.court_thr), post)
        if dbg["no_court"] and post.any():  # 회색 코트 등은 'volleyball court' 점수가 낮다 (0.1~0.2) → 일반 프롬프트로 재시도
            court, dbg = fit_court(frame, self.mask(rgb, "sports court", 0.5), post)
        return court, dbg


if __name__ == "__main__":
    # python -m vbtrk.court_sam3 [영상.mp4]  → 프리뷰 4장(+영상) 결과를 data/court_fit/ 에 저장
    import sys

    out = Path("data/court_fit")
    out.mkdir(parents=True, exist_ok=True)
    det = Sam3Court()
    for p in sorted(Path("data").glob("clip_*_preview.jpg")):
        f = cv2.imread(str(p))
        court, dbg = det(f)
        cv2.imwrite(str(out / p.name), draw_fit(f, court, dbg))
        print(p.name, sorted(dbg.get("lines", {})), round(dbg.get("support", 0), 3), dbg["ok"])
    if len(sys.argv) > 1:
        cap = cv2.VideoCapture(sys.argv[1])
        W, H = int(cap.get(3)), int(cap.get(4))
        wr = cv2.VideoWriter(str(out / (Path(sys.argv[1]).stem + "_fit.mp4")), cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H))
        n = ok = 0
        t = time.time()
        while True:
            r, f = cap.read()
            if not r:
                break
            court, dbg = det(f)
            wr.write(draw_fit(f, court, dbg))
            n += 1
            ok += dbg["ok"]
        wr.release()
        print(f"video: {ok}/{n} ok, {(time.time() - t) / max(n, 1):.2f}s/frame")
