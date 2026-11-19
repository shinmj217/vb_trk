"""영상 없이 돌리는 자가 점검: python test_vbtrk.py"""
import numpy as np

from vbtrk.court import Court, FAR, NEAR, COURT_CORNERS
from vbtrk.team import TeamAssigner
from vbtrk.viz import draw_court_overlay, draw_players, make_minimap, paste_minimap

# 가짜 중계 앵글: 먼 쪽이 좁은 사다리꼴 (1280x720)
CORNERS = [[400, 250], [880, 250], [1180, 650], [100, 650]]


def test_court():
    c = Court(CORNERS)
    assert np.allclose(c.to_court(CORNERS), COURT_CORNERS, atol=1e-3)
    pts = np.array([[2, 3], [7, 15], [-4.5, 9.5], [4, -6], [15, 9], [4, 25]], np.float32)
    assert np.allclose(c.to_court(c.to_image(pts)), pts, atol=1e-2)
    assert list(c.in_play_area(pts)) == [True, True, True, True, False, False]  # 프리존 5m / 6.5m
    assert list(Court.side_of(pts)) == [FAR, NEAR, NEAR, FAR, NEAR, NEAR]  # y=9(네트 위)는 NEAR
    assert c.to_court(np.zeros((0, 2))).shape == (0, 2)


def box(foot_y, h=200, x=600):
    return np.array([x - 30, foot_y - h, x + 30, foot_y], np.float32)


def test_jump_hold():
    t = TeamAssigner()
    t.update([1], [box(600)], [np.array([4.0, 12.0])], [NEAR])        # 착지
    team, pos, air = t.update([1], [box(540)], [np.array([4.0, 8.0])], [FAR])[0]
    assert air and team == NEAR and np.allclose(pos, [4, 12])        # 점프: 위치 유지, 팀 유지
    team, pos, air = t.update([1], [box(598)], [np.array([4.1, 11.9])], [NEAR])[0]
    assert not air and np.allclose(pos, [4.1, 11.9])                  # 착지 복귀


def test_net_jitter_keeps_team():
    t = TeamAssigner()
    for i in range(40):
        side = FAR if i % 5 == 0 else NEAR                            # 네트 근처에서 가끔 반대쪽으로 튐
        team = t.update([1], [box(400)], [np.array([4.0, 9.1])], [side])[0][0]
    assert team == NEAR


def test_stuck_airborne_recovers():
    """가림/ID 스위치로 발 y가 영구히 위로 바뀌어도 '공중' 상태에서 빠져나와야 한다."""
    t = TeamAssigner()
    t.update([1], [box(600)], [np.array([4.0, 15.0])], [NEAR])
    for _ in range(100):
        _, pos, air = t.update([1], [box(300)], [np.array([4.0, 3.0])], [FAR])[0]
    assert not air and np.allclose(pos, [4, 3])


def test_track_gap_resets_ground():
    """트랙이 끊겼다가 다른 선수 위치로 다시 나타나면(ID 스위치) 점프로 보지 않는다"""
    t = TeamAssigner()
    t.update([1], [box(600)], [np.array([4.0, 12.0])], [NEAR])
    t.update([], [], [], [])                                          # 한 프레임 놓침
    _, pos, air = t.update([1], [box(500)], [np.array([4.0, 10.0])], [NEAR])[0]
    assert not air and np.allclose(pos, [4, 10])


def test_camera_tilt_is_not_jump():
    """카메라가 위로 틸트해서 화면 속 발이 40px 올라가도, 같은 코트 위치면 점프가 아니다"""
    c0 = Court(CORNERS)
    c1 = Court(np.float32(CORNERS) - [0, 40])
    t = TeamAssigner()
    foot0 = c0.to_image([[4.0, 12.0]])[0]
    foot1 = c1.to_image([[4.0, 12.0]])[0]
    t.update([1], [box(foot0[1], x=foot0[0])], c0.to_court([foot0]), [NEAR], c0)
    _, _, air = t.update([1], [box(foot1[1], x=foot1[0])], c1.to_court([foot1]), [NEAR], c1)[0]
    assert not air


def test_viz_smoke():
    c = Court(CORNERS)
    t = TeamAssigner()
    frame = np.zeros((720, 1280, 3), np.uint8)
    ids, boxes = [1, 2], [box(600), box(300, h=80)]
    res = t.update(ids, boxes, c.to_court([[600, 600], [600, 300]]), [NEAR, FAR])
    draw_court_overlay(frame, c)
    draw_players(frame, ids, boxes, res)
    paste_minimap(frame, make_minimap(ids, res, {1: [(4, 12), (4.5, 12)]}))
    assert frame.any()


def test_relink():
    """끊긴 트랙: 가까우면 이어 붙이고, 멀거나 팀이 다르거나 겹치면 안 붙임. 이어 붙이면 선수 자격도 이어짐"""
    from vbtrk.team import relink, player_ids
    r = lambda team, *ps: [(team, np.array(p, float), False) for p in ps]
    far5 = (FAR, np.array((4, 3.0)), False)
    frames = [([1, 5], None, r(NEAR, (4, 12)) + [far5])] * 5                   # 1: NEAR, 5: FAR (계속 보임)
    frames += [([5], None, [far5])] * 4                                         # 1 가려짐 (4프레임)
    frames += [([2, 3, 5], None, r(NEAR, (4.5, 12.3), (4, 1)) + [far5])] * 3   # 2: 1번 근처 재등장, 3: 먼 곳
    assert relink(frames)[-1][0] == [1, 3, 5]
    # 팀이 다르면 가까워도 안 붙음
    f2 = [([1], None, r(NEAR, (4, 12)))] * 3 + [([2], None, r(FAR, (4, 12)))] * 3
    assert relink(f2)[-1][0] == [2]
    # 점프 중에 새로 잡힌 조각(발이 사이드라인 밖으로 투영): 이어 붙이면 앞 조각의 '코트 안' 기록으로 선수 인정
    f3 = [([1], None, r(NEAR, (0.5, 12)))] * 3 + [([2], None, r(NEAR, (-1.5, 12.2)))] * 5
    assert player_ids(f3) == {1}
    assert player_ids(relink(f3)) == {1} and relink(f3)[-1][0] == [1]
    # 추적기가 옛 ID(1)를 늦게 버려 새 ID(2)와 2프레임 겹침: 같은 자리면 잇고, 겹친 프레임은 하나만
    f4 = [([1], None, r(NEAR, (4, 12)))] * 3 + [([1, 2], None, r(NEAR, (4, 12), (4.2, 12)))] * 2 + [([2], None, r(NEAR, (4.3, 12)))] * 3
    assert [f[0] for f in relink(f4)] == [[1]] * 8


def test_player_ids():
    """코트 안에 들어온 적 있는 트랙만 선수: 밖으로 공 쫓아간 선수는 유지, 사이드 심판·서버는 구분"""
    from vbtrk.team import player_ids
    r = lambda *ps: [(NEAR, np.array(p, float), False) for p in ps]
    frames = [([1, 2, 3, 4, 5], None, r((4, 12), (10.5, 9), (4, 19.5), (11, 20), (0.3, -2)))] * 3  # 선수, 2심, 서버, 선심, 코너 선심
    frames += [([1, 2], None, r((13, 14), (10.5, 9)))] * 10                              # 1번 코트 밖 4m까지 수비
    assert player_ids(frames) == {1, 3}
    assert player_ids(frames[:2]) == set()                                               # 2프레임은 부족
    # 감독: 광고판 뒤(코트 밖)에 오래 서 있다가 코트 맞춤이 흔들린 몇 프레임만 코트 안 좌표 → 선수 아님
    coach = [([6], None, r((-1.2, 2)))] * 300 + [([6], None, r((2.0, 8.0)))] * 20
    assert player_ids(coach) == set()


def test_fit_court_synthetic():
    """옆 앵글, 오른쪽 엔드라인이 화면 밖인 합성 코트에서 호모그래피 복원"""
    import cv2
    from vbtrk.court_sam3 import fit_court
    truth = Court([[300, 250], [100, 650], [1600, 650], [1500, 250]])  # (0,0) (9,0) (9,18) (0,18)
    frame = np.full((720, 1280, 3), (150, 140, 20), np.uint8)
    mask = np.zeros((720, 1280), np.uint8)
    cv2.fillPoly(mask, [truth.image_corners.astype(np.int32)], 1)
    frame[mask > 0] = (40, 120, 230)
    for p, q in [((0, 0), (0, 18)), ((9, 0), (9, 18))] + [((0, y), (9, y)) for y in (0, 6, 9, 12, 18)]:
        a, b = truth.to_image([p, q]).astype(int)
        cv2.line(frame, tuple(map(int, a)), tuple(map(int, b)), (255, 255, 255), 3)
    post = np.zeros_like(mask)  # 옆 앵글 실제처럼: 가까운 기둥~먼 기둥(+네트)이 한 덩어리 띠
    (nx, ny), (fx, fy) = truth.to_image([[10, 9], [-1, 9]]).astype(int)
    cv2.line(post, (int(nx), int(ny)), (int(fx), int(fy)), 1, 10)
    court, dbg = fit_court(frame, mask, post)
    assert court is not None, dbg
    assert dbg.get("method") == "edges" and {0, 9} <= set(dbg["lines"]), (dbg.get("method"), sorted(dbg["lines"]))
    pts = truth.to_image([[2, 3], [7, 10], [4.5, 15]])
    assert np.allclose(court.to_court(pts), [[2, 3], [7, 10], [4.5, 15]], atol=0.15), court.to_court(pts)


def test_fill_gaps():
    """짧은 공백만 보간(앞뒤 중간), 긴 공백과 영상 앞뒤 끝은 비워둠 → 트래킹 멈춤"""
    from vbtrk.court_sam3 import fill_gaps
    a = Court([[300, 250], [100, 650], [1600, 650], [1500, 250]]).image_corners
    b = a + np.float32([100, 0])                         # 카메라가 100px 패닝
    out = fill_gaps([None, a, None, b, None])
    mid = Court(out[2]).to_image([[4.5, 9]])[0]
    exp = (Court(a).to_image([[4.5, 9]])[0] + Court(b).to_image([[4.5, 9]])[0]) / 2
    assert np.allclose(mid, exp, atol=1), (mid, exp)
    assert out[0] is None and out[4] is None
    long_gap = fill_gaps([a] + [None] * 11 + [b], max_gap=10)  # 관중석 컷 같은 긴 공백
    assert all(c is None for c in long_gap[1:12])
    assert fill_gaps([None, None]) == [None, None]


def test_drop_outliers():
    """부드러운 패닝 중 한 프레임만 엉뚱한 코트 → 버림, 나머지는 유지"""
    from vbtrk.court_sam3 import drop_outliers
    a = Court([[300, 250], [100, 650], [1600, 650], [1500, 250]]).image_corners
    seq = [a + np.float32([5 * i, 0]) for i in range(9)]
    seq[4] = a + np.float32([400, 80])
    seq[6] = None
    out = drop_outliers(seq)
    assert out[4] is None and out[6] is None
    assert all(out[i] is not None for i in (0, 1, 2, 3, 5, 7, 8))
    # 빠른 패닝(프레임당 30px): 검출 구간 끝 프레임도 버리면 안 됨
    fast = [a + np.float32([30 * i, 0]) for i in range(8)]
    assert all(c is not None for c in drop_outliers(fast))
    # 실제 클립처럼 엉뚱한 값이 40% 섞여도 정상 값은 남고 엉뚱한 값만 버린다
    bad = {2, 4, 5, 8, 11, 13}
    mixed = [a + np.float32([-20 * i, 0]) if i not in bad else a + np.float32([-300 + 7 * i, 40 * (i % 3)])
             for i in range(15)]
    out = drop_outliers(mixed)
    assert [i for i, c in enumerate(out) if c is None] == sorted(bad), [i for i, c in enumerate(out) if c is None]


def test_eval_parse_and_match():
    import tempfile, os
    from eval import load_annotations, match
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write("48075.jpg r_winpoint 372 442 86 130 falling 712 426 73 124 standing\n")
    try:
        (fid, group, boxes, acts), = load_annotations(f.name)
    finally:
        os.remove(f.name)
    assert (fid, group, acts) == (48075, "r-winpoint", ["falling", "standing"])
    assert np.allclose(boxes[0], [372, 442, 458, 572])
    pred = np.float32([[715, 430, 785, 550], [370, 440, 460, 570], [0, 0, 10, 10]])
    assert match(boxes, pred) == {0: 1, 1: 0}


def test_action_report():
    from action_eval import report
    cm, rec = report(["a", "b"], np.array([0, 0, 0, 1]), np.array([0, 0, 1, 1]))
    assert cm.tolist() == [[2, 1], [0, 1]] and np.allclose(rec, [2 / 3, 1])


def test_temporal_head_shapes():
    import torch
    from action_temporal import TemporalHead
    assert TemporalHead(9)(torch.zeros(4, 9, 1280)).shape == (4, 9)


def test_scene_feat():
    from action_team import scene_feat
    p = np.array([[0.1, 0.9], [0.5, 0.5], [0.8, 0.2]])
    f = scene_feat(p, np.array([True, True, False]))   # 왼쪽 팀 max, mean / 오른쪽 팀 max, mean
    assert np.allclose(f, [0.5, 0.9, 0.3, 0.7, 0.8, 0.2, 0.8, 0.2])


def test_event_segment_and_actor():
    from events import segment, actor
    from action_team import GROUPS
    g = np.zeros((30, 8))
    g[:12, GROUPS.index("l-pass")] = 1
    g[12:14, GROUPS.index("r-set")] = 1                 # 2프레임짜리 튐 → 다듬기로 사라짐
    g[14:, GROUPS.index("l-set")] = 1
    g[25:27] = np.nan                                    # 코트 없는 프레임
    segs, _, _ = segment(g)
    assert [s[2] for s in segs] == ["l-pass", "l-set"], segs
    rows = [(1, "LEFT", None, None), (2, "RIGHT", None, None), (3, "LEFT", None, None)]
    p = np.zeros((3, 5)); p[:, 1] = [0.2, 0.9, 0.6]      # dig 확률: RIGHT 선수가 제일 높아도 LEFT 팀에서 고름
    assert actor(rows, p, "l-pass") == 2 and actor(rows, p, "l-winpoint") is None


def test_court_holes():
    """stride 검출 후 보간 못 한 짧은 구멍만 다시 돌림. 긴 공백·코트 없음 구간은 안 건드림"""
    from vbtrk.court_sam3 import _holes, NO_COURT
    a = Court([[300, 250], [100, 650], [1600, 650], [1500, 250]]).image_corners.tolist()
    raw = [None] * 120
    for i in (0, 10, 40, 50):          # 10~40 사이 키프레임 실패 → 29프레임 구멍 (stride 10이면 보간 한도 20 초과)
        raw[i] = a
    raw[60] = NO_COURT                 # 50~60: 코트 없음 경계 → 안 건드림. 60 이후 코트 끝 → 안 건드림
    todo = _holes(raw, 10)
    assert todo and all(10 < k < 40 for k in todo) and todo[:2] == [11, 13], todo


def test_movement():
    """3m를 30프레임(1초)에 걸어감 + 점프 프레임 + 한 프레임 5m 튐(오차) → 거리 약 3m, 속도 약 3m/s"""
    from analysis import movement
    pts = [(f, 1.0, 2.0 + 0.1 * f, 0, "LEFT") for f in range(31)]
    pts[15] = (15, 1.0, 9.0, 1, "LEFT")          # 점프(공중) → 제외
    pts[20] = (20, 6.0, 4.0, 0, "LEFT")          # 튐 → 이동 평균 후에도 큰 걸음은 버림
    dist, v = movement(pts, 30)
    assert 2.0 < dist < 3.5 and 2.0 < v < 4.5, (dist, v)


def test_left_team():
    """화면 왼쪽에 보이는 코트 반쪽이 LEFT. 같은 코트를 반대편에서 찍으면(꼭짓점 좌우 반전) 반대"""
    from vbtrk.viz import side_name
    c = Court([[300, 250], [100, 650], [1600, 650], [1500, 250]])   # (0,0) 왼쪽 위 … y=18 쪽이 오른쪽
    assert c.left_team() == FAR and side_name(FAR, c.left_team()) == "LEFT" and side_name(NEAR, c.left_team()) == "RIGHT"
    m = Court([[1280 - x, y] for x, y in [[300, 250], [100, 650], [1600, 650], [1500, 250]]])
    assert m.left_team() == NEAR and side_name(FAR, m.left_team()) == "RIGHT"


def test_ball_score():
    from ball_eval import score
    res = [((100, 100), (102, 101, 0.9)),   # TP (2.2px)
           ((100, 100), (120, 100, 0.9)),   # FP (20px)
           ((100, 100), None),              # FN
           (None, (50, 50, 0.9)),           # FP (공 없는데 예측)
           (None, (50, 50, 0.1)),           # TN (임계값 아래)
           ((100, 100), (100, 100, 0.2))]   # FN (임계값 아래)
    s = score(res, 0.5, 4)
    assert (round(s["P"], 3), round(s["R"], 3), round(s["Acc"], 3)) == (0.333, 0.333, 0.333), s
    assert score(res, 0.5, 30)["P"] == 2 / 3


def test_ball_track():
    """혼자 튀는 오검출은 이웃 지지가 없어 버려지고, 공 궤적의 1프레임 구멍은 보간. 구간은 영상·연속 프레임 기준"""
    from ball_track import choose, fill, segments
    cands = [[(100, 100, 0.8)], [(110, 100, 0.8), (600, 300, 0.9)], [], [(130, 100, 0.8)], [(140, 100, 0.8)]]
    pts = fill(choose(cands, 30), 0.5, 30, 2)
    assert [round(p[0]) for p in pts] == [100, 110, 120, 130, 140], pts   # 600은 버림, t=2는 보간
    assert segments([("3_10", 0, 0), ("3_12", 0, 0), ("3_11", 0, 0), ("4_13", 0, 0)]) == [[0, 2, 1], [3]]


def test_courses():
    """스파이크 다음이 상대 리시브면 코스(직선/크로스·짧게/깊게), 같은 팀 이벤트가 이어지면 도착 없음"""
    from analysis import courses
    ev = lambda t, a, x, y, f: dict(team=t, action=a, court_x=x, court_y=y, peak=f, actor="1")
    c = courses([ev("LEFT", "spike", 1.0, 8.0, 10), ev("RIGHT", "pass", 7.0, 16.0, 20),
                 ev("RIGHT", "spike", 2.0, 10.0, 30), ev("LEFT", "pass", 2.5, 7.0, 40),
                 ev("LEFT", "spike", 5.0, 8.0, 50), ev("LEFT", "set", 5.0, 7.0, 60)])
    assert [x["kind"] for x in c] == ["크로스 깊게", "직선 짧게", "상대 리시브 없음"], c


def test_square_crop_edge():
    import cv2, tempfile, os
    from action_data import square_crop
    f = os.path.join(tempfile.mkdtemp(), "f.jpg")
    cv2.imwrite(f, np.full((720, 1280, 3), 200, np.uint8))
    c = square_crop(f, (0, 600, 40, 720))          # 화면 왼쪽 아래 구석 선수
    assert c.shape[0] == c.shape[1] == 132


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print("PASS", name)
            except Exception as e:
                fails += 1
                print("FAIL", name, type(e).__name__, e)
    raise SystemExit(fails)


def test_merge_winpoints():
    from events import merge_winpoints
    ev = lambda p, a, c=.9: dict(start=p, end=p, peak=p, action=a, conf=c)
    evs = [ev(0, "pass"), ev(100, "winpoint"), ev(120, "winpoint", .99), ev(130, "pass"),  # 100: 뒤에 동작 → 랠리 중
           ev(500, "winpoint"), ev(520, "winpoint"), ev(1000, "winpoint")]
    out = merge_winpoints(evs, 150, 60)
    # 100·120은 랠리 중(120 뒤 10프레임에 pass)·100 뒤 30프레임에 pass → 둘 다 제외, 500·520은 처음 것만, 1000 유지
    assert [(e["peak"], e["action"]) for e in out] == [(0, "pass"), (130, "pass"), (500, "winpoint"), (1000, "winpoint")]
    mixed = [dict(start=10, end=20, peak=15, action="winpoint", team="LEFT", conf=.6),
             dict(start=40, end=50, peak=45, action="winpoint", team="RIGHT", conf=.9)]
    (m,) = merge_winpoints(mixed, 150, 60)
    assert (m["peak"], m["team"]) == (15, "RIGHT")  # 시각은 처음 것, 팀은 확신도 높은 것


def test_rally_starts_from_points():
    from analysis import rally_starts
    # 프레임 0~299 코트에 선수 10명, 득점 (100, LEFT) → 다음 랠리는 선수 8명 이상이 2초(60프레임) 이어지는 첫 프레임
    tracks = {i: [(f, 4.0, 2.0 + i, 0, "LEFT" if i < 5 else "RIGHT") for f in range(150, 300)] for i in range(10)}
    r = rally_starts(tracks, [dict(peak=100, team="LEFT", action="winpoint")], 30, wait_s=0)
    assert r == [(150, "LEFT")]


def test_rallies():
    from analysis import rallies
    ev = lambda s, a: dict(start=s, end=s + 5, action=a)
    r = rallies([(0, "LEFT"), (300, "RIGHT"), (600, "RIGHT")], [ev(10, "pass"), ev(50, "spike"), ev(100, "winpoint"), ev(320, "pass")], 30)
    assert [(x["serve_team"], x["winner"], x["contacts"]) for x in r] == [("LEFT", "RIGHT", 2), ("RIGHT", "RIGHT", 1), ("RIGHT", None, 0)]


def test_serve_detect():
    """서빙 API: 헬스, 정상 이미지(사람 박스 형식), 깨진 파일은 400, 메트릭 노출"""
    import cv2
    import pytest
    from pathlib import Path
    if not Path("yolo26m.pt").exists():  # 가중치는 git 이 아니라 DVC 로 받는다 (CI 에선 없을 수 있음)
        pytest.skip("yolo26m.pt 없음 (dvc pull)")
    from fastapi.testclient import TestClient
    import serve
    with TestClient(serve.app) as c:
        assert c.get("/health").json() == {"ok": True}
        ok, jpg = cv2.imencode(".jpg", np.full((360, 640, 3), 90, np.uint8))
        r = c.post("/detect?conf=0.3", files={"file": ("a.jpg", jpg.tobytes(), "image/jpeg")})
        j = r.json()
        assert r.status_code == 200 and (j["width"], j["height"]) == (640, 360) and isinstance(j["persons"], list), j
        assert c.post("/detect", files={"file": ("a.jpg", b"not an image", "image/jpeg")}).status_code == 400
        assert c.post("/detect?conf=2", files={"file": ("a.jpg", jpg.tobytes(), "image/jpeg")}).status_code == 422
        m = c.get("/metrics").text
        assert 'vb_detect_requests_total{status="ok"}' in m and 'status="bad_image"' in m
