"""트랙 ID별 팀 판정과 위치 안정화.

- 팀: 최근 N프레임 동안 어느 쪽 코트에 있었는지 다수결 → 네트 근처에서 흔들려도 팀이 안 바뀜
- 위치: 점프 중에는 발이 떠서 호모그래피 위치가 네트 반대쪽으로 튀므로,
  발 위치(박스 하단)가 직전보다 급하게 위로 올라가면 '공중'으로 보고 마지막 착지 위치를 유지
"""
from collections import Counter, defaultdict, deque

import numpy as np

from .court import COURT_L, COURT_W, FAR, NEAR


class TrackState:
    def __init__(self, history):
        self.sides = deque(maxlen=history)
        self.ground_pos = None      # 마지막 착지 위치 (코트 좌표)
        self.ground_foot_y = None   # 착지 시점의 영상 내 발 y
        self.ground_box_h = None
        self.air_frames = 0
        self.last_t = None          # 마지막으로 보인 프레임 (update 호출 순번)


class TeamAssigner:
    def __init__(self, history=30, jump_ratio=0.12, max_air=30):
        """jump_ratio: 발 y가 박스 높이의 이 비율 이상 위로 올라가면 점프로 판단
        max_air: 이 프레임 수보다 오래 '공중'이면 점프가 아니라 가림/ID 스위치로 보고 현재 위치로 리셋
                 (점프 체공 ~1초 미만, 30fps 기준. fps가 다르면 조정)
        """
        self.history = history
        self.jump_ratio = jump_ratio
        self.max_air = max_air
        self.states = defaultdict(lambda: TrackState(self.history))
        self.t = 0  # update 한 번 = 한 프레임

    def update(self, track_ids, boxes, court_pts, sides, court=None):
        """한 프레임 결과를 받아 (team, 안정화된 코트 좌표, 점프 여부) 리스트를 돌려준다.
        court: 이 프레임의 Court. 주면 마지막 착지 위치를 이 프레임 영상에 다시 투영해서 비교
               → 카메라가 움직여도 발 높이 변화를 점프로 착각하지 않는다."""
        self.t += 1
        out = []
        for tid, box, cpt, side in zip(track_ids, boxes, court_pts, sides):
            st = self.states[int(tid)]
            foot_y = box[3]
            box_h = box[3] - box[1]
            # 트랙이 끊겼다 다시 나타나면 ByteTrack이 다른 선수에게 ID를 붙였을 수 있다 → 이전 착지와 비교 안 함
            # ponytail: 끊긴 동안 시작된 점프는 놓침. 오프라인 ID 재연결을 붙이면 다시 볼 것
            fresh = st.last_t is None or self.t - st.last_t > 1
            st.last_t = self.t

            airborne = False
            if not fresh and st.ground_foot_y is not None and st.ground_box_h:
                ground_y = st.ground_foot_y if court is None else court.to_image([st.ground_pos])[0][1]
                rise = ground_y - foot_y
                if rise > self.jump_ratio * st.ground_box_h and st.air_frames < self.max_air:
                    airborne = True
            st.air_frames = st.air_frames + 1 if airborne else 0

            if airborne:
                pos = st.ground_pos
            else:
                pos = cpt
                st.ground_pos = cpt
                st.ground_foot_y = foot_y
                st.ground_box_h = box_h
                st.sides.append(int(side))

            team = self._vote(st)
            out.append((team, pos, airborne))
        return out

    @staticmethod
    def _vote(st):
        if not st.sides:
            return None
        far = sum(1 for s in st.sides if s == FAR)
        return FAR if far * 2 >= len(st.sides) else NEAR


def relink(frames, max_gap=30, base=2.0, speed=0.3, overlap=10):
    """끊긴 트랙 이어 붙이기 (오프라인). frames: [(ids, boxes, results)] → 같은 형식, ID만 대표 ID로 바뀜.
    A가 끝나고 max_gap 프레임 안에 B가 시작하고, A 마지막 위치와 B 첫 위치(코트 좌표)가
    base + speed*간격 m 안이고, 팀이 다르지 않으면 B를 A에 잇는다. 가까운 쌍부터 탐욕으로 1:1.
    speed 0.3m/프레임 = 30fps에서 9m/s (선수 최고 속도 위).
    B가 A 끝나기 전 overlap 프레임 안에 시작해도 (추적기가 옛 ID를 늦게 버린 경우) B 시작 순간 A 위치와
    base m 안이면 잇고, 겹친 프레임은 A를 남긴다.
    long_20·48 실험: 간격을 60~90으로 늘리면 한 영상은 좋아지고 다른 영상은 나빠짐 → 30 유지"""
    # ponytail: 위치만 본다. 같은 팀은 유니폼이 같아 색 특징은 무의미, ReID(BoT-SORT)도 들쭉날쭉 → 등번호 인식이 다음
    span, at = {}, {}  # tid -> [첫 프레임, 마지막 프레임, 첫 위치, 마지막 위치, 팀], (tid, 프레임) -> 위치
    for t, (ids, _, res) in enumerate(frames):
        for tid, (team, pos, _) in zip(ids, res):
            s = span.setdefault(int(tid), [t, t, pos, pos, team])
            s[1], s[3], s[4] = t, pos, team
            at[int(tid), t] = pos
    pairs = []
    for a, (a_start, a_end, _, a_pos, a_team) in span.items():
        for b, (b_start, b_end, b_pos, _, b_team) in span.items():
            gap = b_start - a_end
            if a == b or not -overlap <= gap <= max_gap or b_start <= a_start or b_end <= a_end:
                continue
            ref = a_pos if gap > 0 else at.get((a, b_start), a_pos)
            d = float(np.hypot(*(np.asarray(b_pos) - ref)))
            same_team = None in (a_team, b_team) or a_team == b_team
            if d <= base + speed * max(gap, 0) and same_team:
                pairs.append((d, a, b))
    nxt, prev = {}, {}
    for d, a, b in sorted(pairs):
        if a not in nxt and b not in prev:
            nxt[a], prev[b] = b, a
    root = {}
    for t in span:
        r = t
        while r in prev:
            r = prev[r]
        root[t] = r
    out = []
    for ids, boxes, res in frames:  # 겹친 프레임엔 같은 대표 ID가 둘 → 먼저 나온(A) 것만
        keep, seen = [], set()
        for i, t in enumerate(ids):
            if root[int(t)] not in seen:
                seen.add(root[int(t)])
                keep.append(i)
        out.append(([root[int(ids[i])] for i in keep], boxes if boxes is None else np.asarray(boxes)[keep],
                    [res[i] for i in keep]))
    return out


def player_ids(frames, min_in=3, serve=3.0, corner=1.0, min_frac=0.5, long_track=60):
    """영상 전체 결과 [(ids, boxes, results)] → 선수 트랙 ID 집합 (오프라인).
    선수 = 코트 안, 또는 엔드라인 뒤 서브 구역에 min_in 프레임 이상 있었고, 그 프레임이 트랙 전체의 min_frac 이상인 트랙.
    서브 구역은 코너 corner m를 뺀다 (코너 바깥에 선 선심 제외).
    감독·벤치·선심·심판은 대부분 시간 코트 밖(광고판 뒤)에 서 있는데, 코트 맞춤이 잠깐 틀리거나 카메라가 움직이는 순간
    코트 안 좌표로 찍히면 '한 번이라도 들어온' 규칙으론 영원히 선수가 됐다 (1세트: 감독 트랙 코트 안 비율 0.05~0.19,
    선수는 0.95 이상). 코트 안 비율로 가른다. 한 번 선수로 판정되면 코트 밖 잠깐 공을 쫓아가도 계속 선수 (짧은 트랙, long_track 프레임 미만은 비율 안 봄)."""
    n, total = Counter(), Counter()
    for ids, _, res in frames:
        for tid, (_, pos, _) in zip(ids, res):
            x, y = pos
            total[int(tid)] += 1
            in_court = 0 <= x <= COURT_W and 0 <= y <= COURT_L
            # ponytail: 코너 끝에서 서브하고 코트에 안 들어온 서버는 놓침 (짧은 영상에서만 문제)
            in_serve = corner <= x <= COURT_W - corner and -serve <= y <= COURT_L + serve
            if in_court or in_serve:
                n[int(tid)] += 1
    return {t for t, k in n.items() if k >= min_in and (total[t] < long_track or k >= min_frac * total[t])}
