"""전력분석 결과물: 추적(run.py) + 이벤트(events.py) 결과 → 그림 한 장 + 표

    python analysis.py --video data/long_20.mp4
출력: outputs/<영상>_analysis.png, outputs/<영상>_players.csv

그림 (코트는 영상처럼 가로: 왼쪽 = LEFT 팀, 오른쪽 = RIGHT 팀, 가운데 세로선 = 네트)
  1) 팀별 위치 히트맵   2) 이벤트 위치 (리시브·토스·스파이크를 한 선수 자리, 랠리 순서 번호)
  3) 서브 리시브 포메이션 (랠리 첫 리시브 순간 리시브 팀 6명 자리)   5) 공격 코스 (스파이크 → 상대 리시브)
  7) 이벤트 타임라인
표: 4) 트랙별 이동 거리, 최고 속도, 주 위치, 이벤트 수   6) 스파이크별 코스 (직선/크로스, 짧게/깊게)
"""
import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from vbtrk.court import COURT_L, COURT_W

COLOR = {"LEFT": "#d62728", "RIGHT": "#1f77b4"}
ACT = {"pass": ("o", "리시브"), "set": ("s", "토스"), "spike": ("^", "스파이크"), "winpoint": ("*", "득점")}
plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]  # 윈도우 한글
plt.rcParams["axes.unicode_minus"] = False


def load(stem, out):
    rows = list(csv.DictReader(open(out / f"{stem}_tracks.csv", encoding="utf-8")))
    events = list(csv.DictReader(open(out / f"{stem}_events.csv", encoding="utf-8")))
    # 그림은 LEFT 팀 코트가 왼쪽(y<9)이 되게: 반대면 코트 좌표를 180도 돌림 (CSV 값은 그대로)
    flip = np.median([float(r["court_y"]) for r in rows if r["team"] == "LEFT"] or [0]) > COURT_L / 2
    rot = (lambda x, y: (COURT_W - x, COURT_L - y)) if flip else (lambda x, y: (x, y))
    tracks = defaultdict(list)  # tid -> [(frame, x, y, airborne, team)]
    for r in rows:
        x, y = rot(float(r["court_x"]), float(r["court_y"]))
        tracks[int(r["track_id"])].append((int(r["frame"]), x, y, int(r["airborne"]), r["team"]))
    for ev in events:
        if ev["court_x"] != "":
            ev["court_x"], ev["court_y"] = rot(float(ev["court_x"]), float(ev["court_y"]))
    return tracks, events


def rally_starts(tracks, events, fps, min_players=8, hold_s=2.0, wait_s=3.0):
    """랠리 시작 = 득점 이벤트 뒤 wait_s초 지나 처음으로 코트 위 선수가 min_players명 이상인 장면이 hold_s초 이어지는 첫 프레임.
    중계는 서브 직전 서버 클로즈업 → 서브 순간 와이드로 넘어가서 서버를 직접 보기 어렵다 (서버 인식 시도: 서브 팀 9/14).
    서브 팀 = 직전 득점 팀 (배구 규칙). → [(frame, 서브 팀)]"""
    on = Counter()
    for pts in tracks.values():
        for f, x, y, air, tm in pts:
            if -1 <= x <= COURT_W + 1 and -1 <= y <= COURT_L + 1:
                on[f] += 1
    wins = sorted((int(e["peak"]), e["team"]) for e in events if e["action"] == "winpoint")
    out = []
    for k, (w, team) in enumerate(wins):
        nxt = wins[k + 1][0] if k + 1 < len(wins) else max(on, default=0) + 1
        run = 0
        for f in range(w + int(wait_s * fps), nxt):
            run = run + 1 if on[f] >= min_players else 0
            if run >= hold_s * fps:
                out.append((f - run + 1, team))
                break
    return out


def rallies(starts, events, fps):
    """랠리 단위 기록 [(시작, 서브 팀, 이긴 팀, 접촉 수, 길이 초)]. 이긴 팀 = 다음 랠리 서브 팀 (득점 팀이 다음 서브),
    접촉 수 = 랠리 안 리시브·토스·스파이크 이벤트 수, 마지막 랠리는 이긴 팀 모름(None)"""
    out = []
    for k, (f, team) in enumerate(starts):
        win = starts[k + 1][1] if k + 1 < len(starts) else None
        end = starts[k + 1][0] if k + 1 < len(starts) else max((int(e["end"]) for e in events), default=f)
        n = sum(1 for e in events if e["action"] != "winpoint" and f <= int(e["start"]) < end)
        out.append(dict(start=f, serve_team=team, winner=win, contacts=n, seconds=round((end - f) / fps, 1)))
    return out


def movement(pts, fps, smooth=9, max_step=0.4, span=0.5):
    """코트 좌표 이동 거리(m)·최고 속도(m/s). 점프 중 프레임 제외, 이동 평균으로 떨림 제거,
    한 프레임에 max_step m(30fps면 12m/s) 넘게 튀는 건 오차로 버림.
    최고 속도 = span초 동안 이동한 거리 기준의 95% 값 (프레임 단위 떨림이 속도로 부풀지 않게)"""
    p = np.array([(f, x, y) for f, x, y, air, _ in pts if not air], float)
    if len(p) < smooth + 1:
        return 0.0, 0.0
    k = np.ones(smooth) / smooth
    q = np.pad(p, ((smooth // 2, smooth // 2), (0, 0)), mode="edge")  # 양 끝 이동이 잘려나가지 않게
    xy = np.stack([np.convolve(q[:, i], k, "valid") for i in (1, 2)], 1)
    fr = p[:, 0]
    step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    dt = np.diff(fr)
    ok = (dt > 0) & (step / np.maximum(dt, 1) <= max_step)
    dist = float(step[ok].sum())
    w = max(1, int(span * fps))
    d = np.linalg.norm(xy[w:] - xy[:-w], axis=1) if len(xy) > w else np.zeros(0)
    gap = fr[w:] - fr[:-w] if len(xy) > w else np.zeros(0)
    speed = d[gap <= 1.5 * w] / (gap[gap <= 1.5 * w] / fps)  # 트랙이 오래 끊긴 구간은 제외
    speed = speed[speed <= max_step * fps]
    return dist, float(np.percentile(speed, 95)) if len(speed) else 0.0


def courses(events, straight=3.0, short=3.0):
    """스파이크마다 공격 코스: 출발 = 스파이크한 선수 자리, 도착 = 바로 다음 이벤트가 상대 팀 리시브면 그 선수 자리.
    직선/크로스 = 폭 방향 이동 < straight m 여부, 짧게/깊게 = 도착이 네트에서 short m(공격선) 안인지.
    ponytail: 도착은 받은 선수 발 위치 근사 — 공이 바닥에 떨어진(득점) 스파이크는 도착 없음.
              공 궤적(ball_video.py)의 바운드 지점으로 채우는 건 득점 스파이크가 있는 영상에서"""
    out = []
    for ev, nxt in zip(events, events[1:] + [None]):
        if ev["action"] != "spike" or ev["court_x"] == "":
            continue
        c = dict(team=ev["team"], frame=int(ev["peak"]), actor=ev["actor"],
                 start=(float(ev["court_x"]), float(ev["court_y"])), end=None, receiver="", kind="상대 리시브 없음")
        if nxt and nxt["team"] != ev["team"] and nxt["action"] == "pass" and nxt["court_x"] != "":
            c["end"] = (float(nxt["court_x"]), float(nxt["court_y"]))
            c["receiver"] = nxt["actor"]
            c["kind"] = ("직선" if abs(c["end"][0] - c["start"][0]) < straight else "크로스") + \
                        (" 짧게" if abs(c["end"][1] - COURT_L / 2) < short else " 깊게")
        out.append(c)
    return out


def draw_court(ax, title):
    """가로 코트: 가로축 = 코트 길이(0~18, 왼쪽 = LEFT 팀 코트), 세로축 = 폭(x 0~9)"""
    ax.add_patch(plt.Rectangle((0, 0), COURT_L, COURT_W, fill=True, color="#f2c48d", zorder=0))
    ax.plot([0, COURT_L, COURT_L, 0, 0], [0, 0, COURT_W, COURT_W, 0], "w-", lw=2)
    ax.plot([9, 9], [-0.5, COURT_W + 0.5], "k-", lw=3)          # 네트
    for y in (6, 12):
        ax.plot([y, y], [0, COURT_W], "w-", lw=1.5)               # 공격선
    ax.set_xlim(-4, COURT_L + 4)
    ax.set_ylim(COURT_W + 3, -3)                                  # 먼 쪽 사이드라인(x=0)이 위
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    ax.text(4.5, -2.2, "LEFT 팀", ha="center", color=COLOR["LEFT"])
    ax.text(13.5, -2.2, "RIGHT 팀", ha="center", color=COLOR["RIGHT"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--out", default="outputs")
    p.add_argument("--fps", type=float, default=30)
    a = p.parse_args()
    stem, out = Path(a.video).stem, Path(a.out)
    tracks, events = load(stem, out)
    n_frames = max(f for pts in tracks.values() for f, *_ in pts) + 1

    fig = plt.figure(figsize=(16, 16))
    gs = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1, 0.35])

    # 1) 위치 히트맵 (팀별 색, 착지 위치만)
    ax = fig.add_subplot(gs[0, 0])
    draw_court(ax, "1) 위치 히트맵")
    for team, cmap in (("LEFT", "Reds"), ("RIGHT", "Blues")):
        xy = np.array([(y, x) for pts in tracks.values() for _, x, y, air, tm in pts if tm == team and not air])
        if len(xy):
            h, xe, ye = np.histogram2d(xy[:, 0], xy[:, 1], bins=[44, 30], range=[[-4, 22], [-3, 12]])
            h = np.ma.masked_less(h, 1)
            ax.pcolormesh(xe, ye, h.T, cmap=cmap, alpha=0.75, zorder=1)

    # 2) 이벤트 위치
    ax = fig.add_subplot(gs[0, 1])
    many = len(events) <= 40  # 경기 전체면 숫자·큰 마커 생략 (겹쳐서 안 읽힘)
    draw_court(ax, "2) 이벤트 위치" + (" (숫자 = 순서)" if many else ""))
    for k, ev in enumerate(events, 1):
        if ev["court_x"] == "":
            continue
        m, _ = ACT[ev["action"]]
        x, y = float(ev["court_x"]), float(ev["court_y"])
        ax.scatter(y, x, marker=m, s=160 if many else 50, c=COLOR[ev["team"]], edgecolors="k", zorder=3, alpha=1 if many else 0.6)
        if many:
            ax.annotate(str(k), (y, x), xytext=(6, -10), textcoords="offset points", fontsize=8)
    for act, (m, name) in ACT.items():
        ax.scatter([], [], marker=m, c="gray", edgecolors="k", label=name)
    ax.legend(loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.12), frameon=False)

    # 3) 서브 리시브 포메이션: 랠리 첫 리시브(시작 또는 득점 직후의 첫 pass) 시작 프레임의 리시브 팀 자리
    ax = fig.add_subplot(gs[1, 0])
    draw_court(ax, "3) 서브 리시브 포메이션 (전 랠리)")
    # 서브가 인식되면: 서브 순간 상대(리시브) 팀 자리. 없으면(랠리 중간부터인 클립) 득점 뒤 첫 pass 순간으로 대신
    sv = rally_starts(tracks, events, a.fps)
    rl = rallies(sv, events, a.fps)
    with open(out / f"{stem}_rallies.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, ["start", "serve_team", "winner", "contacts", "seconds"])
        w.writeheader()
        w.writerows(rl)
    summary = []
    for team in ("LEFT", "RIGHT"):
        srv = [r for r in rl if r["serve_team"] == team and r["winner"]]
        rcv = [r for r in rl if r["serve_team"] != team and r["winner"]]
        if srv and rcv:
            summary.append(f"{team}: 서브 득점률 {sum(r['winner'] == team for r in srv)}/{len(srv)}, "
                           f"사이드아웃률 {sum(r['winner'] == team for r in rcv)}/{len(rcv)}")
    print("랠리", len(rl), "|", " | ".join(summary))
    first = [dict(start=f, team="RIGHT" if tm == "LEFT" else "LEFT", serve=1) for f, tm in sv]
    if not sv:
        new_rally = True
        for ev in events:
            if ev["action"] == "winpoint":
                new_rally = True
            elif new_rally and ev["action"] == "pass":
                first.append(ev)
                new_rally = False
    def spots(ev):
        """리시브 팀이 자기 코트 반쪽 안(0.3m 여유)에 서 있는 트랙 위치. 선심·부심은 라인 밖 0.5m+ 라 빠지고,
        네트 앞 공격수는 y≈9.5 라 들어옴 (코트 ±1m 로 두면 선심·주심이 선수로 세어져 8명까지 나왔음)"""
        t = int(ev["start"])
        lo, hi = (-0.3, COURT_L / 2 + 0.3) if ev["team"] == "LEFT" else (COURT_L / 2 - 0.3, COURT_L + 0.3)
        return [(tid, xy[0]) for tid, pts in tracks.items()
                if (xy := [(x, y) for f, x, y, air, tm in pts if f == t and tm == ev["team"]
                           and -0.3 <= x <= COURT_W + 0.3 and lo <= y <= hi])]
    # 리시브 팀이 5~6명 보이는 랠리만 (중계는 서브 순간 컷·줌이 잦고, 7명 이상은 심판·벤치가 섞인 것)
    good = [(ev, sp) for ev in first if 5 <= len(sp := spots(ev)) <= 6]
    for ev, sp in good:  # 모든 랠리의 리시브 자리 (팀 색, 작은 점)
        for tid, (x, y) in sp:
            ax.scatter(y, x, s=28, c=COLOR[ev["team"]], alpha=0.45, edgecolors="none", zorder=3)
    ax.text(9, COURT_W + 2.2, f"서브 순간 리시브 자리 — {len(good)}/{len(first)} 랠리 (5~6명 보인 것)", ha="center")
    if not good:
        ax.text(9, 4.5, "선수 5~6명 보이는 서브 장면 없음", ha="center")

    # 표: 트랙별 이동
    rows = []
    actor_n = Counter(ev["actor"] for ev in events if ev["actor"] != "")
    for tid, pts in sorted(tracks.items()):
        dist, vmax = movement(pts, a.fps)
        team = Counter(tm for *_, tm in pts).most_common(1)[0][0]
        med = np.median([(x, y) for _, x, y, *_ in pts], 0)
        rows.append(dict(track_id=tid, team=team, frames=len(pts), seconds=round(len(pts) / a.fps, 1),
                         distance_m=round(dist, 1), top_speed_mps=round(vmax, 1),
                         median_x=round(med[0], 1), median_y=round(med[1], 1), events=actor_n.get(str(tid), 0)))
    with open(out / f"{stem}_players.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    ax = fig.add_subplot(gs[1, 1])
    ax.axis("off")
    top = sorted(rows, key=lambda r: -r["seconds"])[:14]
    ax.table(cellText=[[r["track_id"], r["team"], r["seconds"], r["distance_m"], r["top_speed_mps"], r["events"]] for r in top],
             colLabels=["트랙", "팀", "보인 시간(s)", "이동(m)", "최고속도(m/s)", "이벤트"], loc="center").scale(1, 1.3)
    ax.set_title("4) 선수(트랙)별 이동 — 오래 보인 순 14개")

    # 5) 공격 코스: 스파이크 자리 → 상대 리시브 자리 화살표 + 표
    cs = courses(events)
    ax = fig.add_subplot(gs[2, 0])
    draw_court(ax, "5) 공격 코스 (스파이크 → 상대 리시브" + (", 숫자 = 프레임)" if len(cs) <= 20 else ")"))
    for c in cs:
        (x0, y0), col = c["start"], COLOR[c["team"]]
        ax.scatter(y0, x0, marker="^", s=160, c=col, edgecolors="k", zorder=3)
        if c["end"]:
            x1, y1 = c["end"]
            ax.annotate("", (y1, x1), (y0, x0), arrowprops=dict(arrowstyle="-|>", color=col, lw=2.5), zorder=2)
            ax.scatter(y1, x1, marker="o", s=90, facecolors="none", edgecolors=col, lw=2, zorder=3)
        if len(cs) <= 20:
            ax.annotate(str(c["frame"]), (y0, x0), xytext=(6, -10), textcoords="offset points", fontsize=8)
    ax = fig.add_subplot(gs[2, 1])
    ax.axis("off")
    if cs:
        ax.table(cellText=[[c["frame"], c["team"], f"#{c['actor']}", f"#{c['receiver']}" if c["receiver"] else "-", c["kind"]] for c in cs[:14]],
                 colLabels=["프레임", "공격 팀", "공격 선수", "리시브 선수", "코스"], loc="center").scale(1, 1.3)
    kinds = Counter(c["kind"] for c in cs if c["end"])
    ax.set_title("6) 공격 코스 표 — " + ", ".join(f"{k} {n}" for k, n in kinds.most_common())
                 + (f"\n(전체 {len(cs)}개 중 처음 14개)" if len(cs) > 14 else ""))

    # 7) 타임라인
    ax = fig.add_subplot(gs[3, :])
    for ev in events:
        s, e = int(ev["start"]), int(ev["end"])
        ax.barh(0 if ev["team"] == "LEFT" else 1, e - s + 1, left=s, color=COLOR[ev["team"]], edgecolor="k")
        if len(events) <= 60:
            ax.text((s + e) / 2, 0 if ev["team"] == "LEFT" else 1, ACT[ev["action"]][1], ha="center", va="center",
                    color="w", fontsize=8)
    ax.set_yticks([0, 1], ["LEFT", "RIGHT"])
    ax.set_xlim(0, n_frames)
    ax.set_xlabel(f"프레임 ({a.fps:.0f}fps)")
    ax.set_title("7) 이벤트 타임라인")

    cnt = Counter((ev["team"], ev["action"]) for ev in events) + Counter((tm, "serve") for _, tm in sv)
    fig.suptitle(f"{stem}\n" + "\n".join(", ".join(f"{t} {(ACT[k][1] if k in ACT else '서브')} {n}" for (t, k), n in sorted(cnt.items()) if t == team)
                                         for team in ("LEFT", "RIGHT")) + "\n" + " | ".join(summary) + ("  (득점 팀 판정 정확도 약 85%라 ±2~3 오차)" if summary else ""), fontsize=13)
    fig.tight_layout()
    fig.savefig(out / f"{stem}_analysis.png", dpi=110)
    print(f"→ {out / f'{stem}_analysis.png'}, {out / f'{stem}_players.csv'}")


if __name__ == "__main__":
    main()
