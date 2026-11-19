"""팀 행동(8종) + '공 만진 선수 1명' 문맥으로 동작 이벤트 정리

1) 선수별 5종 확률 (시간축 헤드 temporal_head5.pt)
2) 장면 특징: 왼쪽/오른쪽 팀(박스 x 정렬 후 반씩, Ibrahim 2016 방식)별 5종 확률의 최댓값·평균 → 20차원
   → 로지스틱 회귀로 팀 행동 8종 (l/r × pass/set/spike/winpoint). 헤드가 학습에 안 쓴 val 장면으로 학습
3) 이벤트: 예측 팀 행동이 pass/set/spike면 그 팀에서 해당 동작(dig/set/spike) 확률이 가장 높은 1명 = 행위자
4) 선수별 5종 재채점: dig/set/spike는 행위자 1명만 남기고 나머지는 other (block은 그대로)

    python action_team.py
"""
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from action_data import collect
from action_eval import report
from action_temporal import NAMES5, TemporalHead, load, merge_map
from eval import load_annotations

GROUPS = [f"{s}-{a}" for s in "lr" for a in ("pass", "set", "spike", "winpoint")]
ACTOR5 = {"pass": "dig", "set": "set", "spike": "spike"}
ACTOR9 = {"pass": "digging", "set": "setting", "spike": "spiking"}


def scenes(split, probs, root="data/volleyball/videos"):
    """npz 행 순서대로 장면 묶기 → [(정답 팀 행동, 행 번호들, 왼쪽 팀 여부, 9종 동작 이름들, 박스들)]"""
    by_act = collect(root)[split]
    rows = sorted(((d, fid, box, act) for act, lst in by_act.items() for d, fid, box in lst),
                  key=lambda r: (r[0], r[1]))  # action_temporal.extract 와 같은 순서
    group = {}
    for v in {Path(d).parent.name for d, *_ in rows}:
        for fid, g, _, _ in load_annotations(f"{root}/{v}/annotations.txt"):
            group[(f"{root}/{v}/{fid}", fid)] = g
    out, i = [], 0
    while i < len(rows):
        j = i
        while j < len(rows) and rows[j][:2] == rows[i][:2]:
            j += 1
        cx = np.array([(r[2][0] + r[2][2]) / 2 for r in rows[i:j]])
        # ponytail: x 순서로 반씩 나눔 (네트 위치 무시). 파이프라인에서는 코트 좌표 팀으로 대체
        left = np.zeros(j - i, bool)
        left[np.argsort(cx)[:(j - i) // 2]] = True
        out.append((group[rows[i][:2]], np.arange(i, j), left, [r[3] for r in rows[i:j]],
                    np.array([r[2] for r in rows[i:j]])))
        i = j
    return out


def scene_feat(p, left):
    return np.concatenate([f(p[m], 0) if m.any() else np.zeros(p.shape[1]) for m in (left, ~left) for f in (np.max, np.mean)])


def layout_feat(boxes, left, W=1280, H=720):
    """팀별 선수 배치: 발 x·y 평균/표준편차, 박스 높이 평균 (득점 후 모이거나 흩어짐)"""
    b = boxes / [W, H, W, H]
    pts = np.stack([(b[:, 0] + b[:, 2]) / 2, b[:, 3], b[:, 3] - b[:, 1]], 1)
    return np.concatenate([np.r_[pts[m, :2].mean(0), pts[m, :2].std(0), pts[m, 2].mean()] if m.any() else np.zeros(5)
                           for m in (left, ~left)])


def head_probs(x, head):
    with torch.no_grad():
        return torch.cat([head(x[i:i + 4096].cuda()).softmax(1).cpu() for i in range(0, len(x), 4096)]).numpy()


def player_probs(size):
    """선수 시간축 헤드 5종·9종 확률 (size: s = temporal_head*.pt, m = temporal_head*_m.pt)"""
    import action_temporal
    action_temporal.FEATS = "data/action_feats" + ("" if size == "s" else f"_{size}")
    sfx = "" if size == "s" else f"_{size}"
    out = {}
    for n, f in ((5, f"temporal_head5{sfx}.pt"), (9, f"temporal_head{sfx}.pt")):
        h = TemporalHead(n).cuda().eval()
        h.load_state_dict(torch.load(f"outputs/action_cls/{f}"))
        out[n] = {sp: head_probs(load(sp)[0], h) for sp in ("val", "test")}
    out["y9"], out["names9"] = load("test")[1:]
    return out


def frame_probs(size):
    import action_frames
    h = TemporalHead(len(GROUPS)).cuda().eval()
    h.load_state_dict(torch.load(f"outputs/action_cls/frames_head_{size}.pt"))
    return {sp: head_probs(action_frames.load(sp, size)[0], h) for sp in ("val", "test")}


def evaluate(pp, fp=None, cues=("f5", "f9", "lay"), verbose=False, save=None):
    """장면 특징(선수 단서 + 프레임 확률)로 팀 행동 → 행위자 1명 → 이벤트 정확도"""
    data = {}
    for sp in ("val", "test"):
        p5, p9 = pp[5][sp], pp[9][sp]
        sc = scenes(sp, p5)
        parts = {"f5": [scene_feat(p5[i], l) for _, i, l, *_ in sc], "f9": [scene_feat(p9[i], l) for _, i, l, *_ in sc],
                 "lay": [layout_feat(b, l) for _, i, l, _, b in sc]}
        X = np.hstack([np.stack(parts[c]) for c in cues] + ([np.log(fp[sp] + 1e-6)] if fp is not None else []))
        data[sp] = (p5, sc, X, np.array([GROUPS.index(g) for g, *_ in sc]))
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=0.1)).fit(data["val"][2], data["val"][3])
    if save:  # 파이프라인(events.py)용: 분류기 + 어떤 단서를 썼는지
        import joblib
        joblib.dump(dict(clf=clf, cues=cues), save)
    p, sc, X, y = data["test"]
    gp = clf.predict(X)
    if verbose:
        print("[팀 행동 8종] test 장면", len(y))
        report(GROUPS, y, gp)

    pred5 = p.argmax(1)
    new5 = pred5.copy()
    other, ok, ok_given, n_ev, n_gr = NAMES5.index("other"), 0, 0, 0, 0
    for (g, idx, left, acts, _), gi in zip(sc, gp):
        for c in ("dig", "set", "spike"):  # 예측된 dig/set/spike는 일단 모두 other
            new5[idx[pred5[idx] == NAMES5.index(c)]] = other
        side, act = GROUPS[gi].split("-")
        pick = None
        if act in ACTOR5:
            c = NAMES5.index(ACTOR5[act])
            team = left if side == "l" else ~left
            if team.any():
                pick = np.where(team)[0][p[idx[team], c].argmax()]
                new5[idx[pick]] = c
        ga = g.split("-")[1]
        if ga in ACTOR9 and ACTOR9[ga] in acts:
            n_ev += 1
            hit = pick is not None and acts[pick] == ACTOR9[ga]
            ok += hit and GROUPS[gi] == g
            if GROUPS[gi] == g:
                n_gr += 1
                ok_given += hit
    acc = (gp == y).mean()
    mca = np.mean([(gp[y == k] == k).mean() for k in range(len(GROUPS))])
    win = [(gp[y == GROUPS.index(k)] == GROUPS.index(k)).mean() for k in ("l-winpoint", "r-winpoint")]
    if verbose:
        print(f"\n[이벤트] 정답 행위자 있는 장면 {n_ev}개: 팀 행동+행위자 모두 맞음 {ok / n_ev:.3f}"
              f"  (팀 행동이 맞은 {n_gr}개 중 행위자 맞음 {ok_given / max(n_gr, 1):.3f})")
        y5 = merge_map(pp["names9"])[torch.as_tensor(pp["y9"])].numpy()
        print("\n[선수별 5종] 행위자 1명 규칙 적용")
        report(NAMES5, y5, new5)
    return acc, mca, win, ok / n_ev, ok_given / max(n_gr, 1)


def main():
    """선수 모델(s/m) × 프레임 모델(없음/s/m) 조합 비교 → 가장 좋은 조합 상세 출력"""
    pps = {sz: player_probs(sz) for sz in ("s", "m") if Path(f"outputs/action_cls/temporal_head5{'' if sz == 's' else '_' + sz}.pt").exists()}
    fps = {sz: frame_probs(sz) for sz in ("s", "m") if Path(f"outputs/action_cls/frames_head_{sz}.pt").exists()}
    rows = []
    for ps, pp in pps.items():
        for cues in (("f5",), ("f5", "f9", "lay")):
            rows.append((f"선수 {ps} {'+'.join(cues)}", pp, None, cues))
            for fs, fp in fps.items():
                rows.append((f"선수 {ps} {'+'.join(cues)} + 프레임 {fs}", pp, fp, cues))
    for fs, fp in fps.items():
        rows.append((f"프레임 {fs}만", next(iter(pps.values())), fp, ()))
    best = None
    print(f"{'조합':32s} 팀행동 평균클래스 l-win r-win 이벤트 (팀 맞을 때 행위자)")
    for name, pp, fp, cues in rows:
        acc, mca, win, ev, evg = evaluate(pp, fp, cues)
        print(f"{name:32s} {acc:.3f}  {mca:.3f}    {win[0]:.2f}  {win[1]:.2f}  {ev:.3f} ({evg:.3f})", flush=True)
        if best is None or ev > best[0]:
            best = (ev, name, pp, fp, cues)
    print(f"\n=== 최고 이벤트 정확도: {best[1]} ===")
    evaluate(*best[2:], verbose=True, save="outputs/action_cls/team_fusion.joblib")
    print("저장: outputs/action_cls/team_fusion.joblib (선수 s 5종 + 프레임 m 기준으로 events.py 가 씀)")


if __name__ == "__main__":
    main()
