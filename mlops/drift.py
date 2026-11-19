"""입력 분포 드리프트 감지 (Evidently) + 재학습 신호. MLOps 7단계

    .venv-mlops/Scripts/python mlops/drift.py --ref long_20 long_48 --cur match1_set1
기준(reference) = 학습·평가에 쓴 데이터셋 클립의 추적 결과, 현재(current) = 새로 들어온 영상의 추적 결과.
프레임 단위 특징 (카메라 줌에 안 흔들리는 것만): 프레임당 선수 수, 박스 가로세로 비율, 코트 좌표(x,y), 공중 비율. 컬럼 절반 이상이 드리프트면 재학습 신호.
출력: outputs/drift/<현재>.html (Evidently 리포트), outputs/drift/<현재>.json (요약 + retrain 플래그)
retrain 플래그는 CI/스케줄러가 읽어 재학습·재평가 작업을 띄우는 자리다 (여기서는 플래그까지만 만든다).
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

ROOT = Path(__file__).resolve().parent.parent
FEATURES = ["n_players", "aspect", "court_x", "court_y", "airborne"]  # 박스 픽셀 크기는 카메라 줌에 좌우돼 뺐다 (같은 데이터셋 클립끼리도 드리프트로 나왔음)


def frame_table(stem, max_rows=20000):
    """추적 CSV → 선수 한 명당 한 행 (+그 프레임의 선수 수). 표본은 시간 순서 무관 무작위"""
    df = pd.read_csv(ROOT / "outputs" / f"{stem}_tracks.csv")
    df["n_players"] = df.groupby("frame")["track_id"].transform("size")
    df["box_h"], df["box_w"] = df.y2 - df.y1, df.x2 - df.x1
    df["aspect"] = df.box_h / df.box_w.clip(lower=1)
    return df[FEATURES].sample(min(len(df), max_rows), random_state=0).reset_index(drop=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ref", nargs="+", required=True)
    p.add_argument("--cur", required=True)
    p.add_argument("--threshold", type=float, default=0.5, help="드리프트 컬럼 비율이 이 이상이면 재학습 신호")
    a = p.parse_args()
    ref = pd.concat([frame_table(s) for s in a.ref], ignore_index=True)
    cur = frame_table(a.cur)
    dd = DataDefinition(numerical_columns=FEATURES)
    snap = Report([DataDriftPreset()]).run(current_data=Dataset.from_pandas(cur, data_definition=dd),
                                           reference_data=Dataset.from_pandas(ref, data_definition=dd))
    out = ROOT / "outputs" / "drift"
    out.mkdir(parents=True, exist_ok=True)
    snap.save_html(str(out / f"{a.cur}.html"))
    res = snap.dict()
    drifted = {}
    for m in res["metrics"]:
        name = m.get("metric_name", "")
        if name.startswith("ValueDrift"):
            col = name.split("column=")[1].split(",")[0].rstrip(")")
            drifted[col] = round(float(m["value"]), 3)  # 값 = 정규화 Wasserstein 거리 (Evidently 기본: 0.1 이상이면 드리프트)
    # 요약 지표 DriftedColumnsCount 에서 비율을 읽는다
    share = next((m["value"]["share"] for m in res["metrics"] if m.get("metric_name", "").startswith("DriftedColumnsCount")), None)
    summary = dict(reference=a.ref, current=a.cur, rows_ref=len(ref), rows_cur=len(cur), drifted_share=share,
                   per_column_distance=drifted, threshold=a.threshold, retrain=bool(share is not None and share >= a.threshold))
    (out / f"{a.cur}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{a.cur}: 드리프트 컬럼 비율 {share}, 재학습 신호 {summary['retrain']} → {out / (a.cur + '.html')}")


if __name__ == "__main__":
    main()
