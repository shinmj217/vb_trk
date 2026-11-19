"""실제 경기 결과 채점 (평가 전용): 득점·서브 팀을 정답 점수 파일과 비교.

    python score_match.py --video match1_set1 --gt data/match1_set1_points_gt.csv
    python score_match.py --video match1_set3 --gt data/match1_set3_points_gt.csv --exclude-gaps
정답 CSV(frame, team)는 방송 점수판이 바뀐 프레임을 읽어 만든 평가용 파일이다. 파이프라인(run/events/analysis)은 읽지 않는다.
--exclude-gaps: 점수판이 오래 가려졌다 한꺼번에 여러 점을 갱신한 구간은 득점 시각을 알 수 없어 채점에서 뺀다 (3세트).
출력: outputs/metrics/<영상>.json (MLflow 기록용), 표준 출력에 요약
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from analysis import load, rally_starts

FPS = 29.97


def score(stem, gt_path, out="outputs", exclude_gaps=False, fps=FPS):
    gt = [(int(r["frame"]), r["team"]) for r in csv.DictReader(open(gt_path, encoding="utf-8"))]
    last = gt[-1][0]
    zones = []
    if exclude_gaps:
        cnt = Counter(f for f, _ in gt)
        frames = sorted(cnt)
        zones = [((frames[i - 1] if i else 0), f) for i, f in enumerate(frames)
                 if cnt[f] > 1 or f - (frames[i - 1] if i else 0) > 1500]
    excl = lambda p: any(a <= p <= b for a, b in zones)
    g = [(f, t) for f, t in gt if not excl(f)]
    events = list(csv.DictReader(open(f"{out}/{stem}_events.csv", encoding="utf-8")))
    wins = [(int(e["peak"]), e["team"]) for e in events
            if e["action"] == "winpoint" and int(e["peak"]) <= last + 10 and not excl(int(e["peak"]))]
    used, hit, team = set(), 0, 0
    for f, t in g:  # 점수판은 득점 몇 초 뒤에 바뀌므로 [f-600, f+60] 안 가장 가까운 안 쓴 득점이 짝
        c = [(abs(f - p), p, w) for p, w in wins if -60 <= f - p <= 600 and p not in used]
        if c:
            _, p, w = min(c)
            used.add(p)
            hit += 1
            team += w == t
    tracks, ev = load(stem, Path(out))
    starts = [s for s in rally_starts(tracks, ev, fps) if s[0] <= last and not excl(s[0])]

    def serving(fr):  # 득점한 팀이 다음 서브: 점수판이 바뀐 지 몇 초 뒤부터 유효
        p = [x for x in gt if x[0] < fr - 150]  # 제외 구간의 점도 '누가 득점했나'는 맞으므로 전부 사용
        return p[-1][1] if p else None

    n_serve = sum(1 for fr, _ in starts if serving(fr))
    ok_serve = sum(1 for fr, tm in starts if serving(fr) == tm)
    return dict(video=stem, gt_points=len(g), winpoints=len(wins), found=hit, team_ok=team, false_winpoints=len(wins) - len(used),
                rally_starts=len(starts), serve_scored=n_serve, serve_team_ok=ok_serve, excluded_zones=len(zones),
                found_rate=round(hit / max(len(g), 1), 3), serve_team_rate=round(ok_serve / max(n_serve, 1), 3))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True, help="영상 이름 (확장자 없이, 예: match1_set1)")
    p.add_argument("--gt", required=True)
    p.add_argument("--out", default="outputs")
    p.add_argument("--exclude-gaps", action="store_true")
    a = p.parse_args()
    r = score(a.video, a.gt, a.out, a.exclude_gaps)
    Path(a.out, "metrics").mkdir(exist_ok=True)
    Path(a.out, "metrics", f"{a.video}.json").write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{a.video}: 득점 {r['found']}/{r['gt_points']} 찾음 (가짜 {r['false_winpoints']}), 팀 {r['team_ok']}/{r['found']}, "
          f"서브 팀 {r['serve_team_ok']}/{r['serve_scored']}")


if __name__ == "__main__":
    main()
