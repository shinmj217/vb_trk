"""평가 결과를 MLflow에 기록하고 모델을 레지스트리에 등록. MLOps 3단계

    .venv-mlops/Scripts/python mlops/log_mlflow.py           # 기록 (mlflow.db, mlartifacts/ 에 저장)
    .venv-mlops/Scripts/mlflow ui --backend-store-uri sqlite:///mlflow.db   # http://localhost:5000
실행 전: python score_match.py 로 outputs/metrics/<영상>.json 을 만들어 둘 것 (파이프라인 쪽 base 환경, 여기는 venv).

실험 "vb_trk": 데이터셋 평가 1회, 실제 경기 세트별 1회 (득점 규칙 값이 params), 공 검출 모델, 선수 검출 모델.
레지스트리: vb_ball_detector (파인튜닝 YOLO26s 1280), vb_player_detector (YOLO26m, COCO 사전학습) — alias 'champion'.
"""
import json
import re
from pathlib import Path

import mlflow
from mlflow import MlflowClient

ROOT = Path(__file__).resolve().parent.parent
mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
mlflow.set_registry_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
EXP = "vb_trk"
RULES = dict(winpoint_gap_s=10, winpoint_ahead_s=3, rally_min_players=8, rally_hold_s=2, rally_wait_s=3,
             court_trust_support=0.85, player_min_inside_frac=0.5)


def get_experiment():
    if not mlflow.get_experiment_by_name(EXP):
        mlflow.create_experiment(EXP, artifact_location=(ROOT / "mlartifacts").as_uri())
    return mlflow.set_experiment(EXP)


def dataset_metrics(log="outputs/eval_v4.log"):
    t = (ROOT / log).read_text(encoding="utf-8")
    court = re.search(r"코트 있음 (\d+)/(\d+)", t)
    rec = re.search(r"재현율 ([\d.]+) \((\d+)/(\d+)\), 정밀도 ([\d.]+)", t)
    team = re.search(r"팀 정답 (\d+)/(\d+) = ([\d.]+)", t)
    return dict(court_found=int(court[1]), court_total=int(court[2]), player_recall=float(rec[1]), player_precision=float(rec[4]),
                actor_team_acc=float(team[3]))


def register(name, path, run_id, tags):
    """가중치 파일을 이 런의 산출물로 올리고 레지스트리에 새 버전으로 등록, 'champion' alias.
    (MLflow 3 의 register_model('runs:/..') 는 mlflow 모델 포맷(logged model)을 요구해서, .pt 는 버전 생성 API 로 직접 등록)"""
    mlflow.log_artifact(str(path), "model")
    c = MlflowClient()
    try:
        c.create_registered_model(name)
    except mlflow.exceptions.MlflowException:
        pass  # 이미 있음
    src = f"{mlflow.get_run(run_id).info.artifact_uri}/model"
    mv = c.create_model_version(name, src, run_id=run_id, tags=tags)
    c.set_registered_model_alias(name, "champion", mv.version)
    return mv.version


def main():
    get_experiment()
    # 1) 데이터셋 평가
    with mlflow.start_run(run_name="dataset_eval_v4") as r:
        mlflow.set_tags({"kind": "eval", "data": "Volleyball Dataset test split"})
        mlflow.log_params({**RULES, "eval_log": "outputs/eval_v4.log"})
        mlflow.log_metrics(dataset_metrics())
    # 2) 실제 경기 세트별
    for f in sorted((ROOT / "outputs" / "metrics").glob("match1_set*.json")):
        m = json.loads(f.read_text(encoding="utf-8"))
        with mlflow.start_run(run_name=m["video"]):
            mlflow.set_tags({"kind": "real_match", "scoring": "score_match.py (평가 전용 점수판 정답)"})
            mlflow.log_params({**RULES, "excluded_zones": m["excluded_zones"]})
            mlflow.log_metrics({k: v for k, v in m.items() if isinstance(v, (int, float)) and k != "excluded_zones"})
            for art in (f"{m['video']}_analysis.png", f"{m['video']}_rallies.csv", f"{m['video']}_events.csv"):
                if (ROOT / "outputs" / art).exists():
                    mlflow.log_artifact(str(ROOT / "outputs" / art), "outputs")
    # 3) 모델: 공 검출 (파인튜닝) / 선수 검출 (사전학습)
    ball = ROOT / "outputs/ball/y26s_1280/weights/best.pt"
    if ball.exists():
        with mlflow.start_run(run_name="ball_detector_y26s_1280") as r:
            mlflow.set_tags({"kind": "model", "base": "yolo26s", "data": "Perez et al. 2022 volleyball ball annotations"})
            mlflow.log_params(dict(imgsz=1280, epochs=30, batch=4, box_px=20, postprocess_D=60, postprocess_gap=4, postprocess_thr=0.22))
            mlflow.log_metrics(dict(f1_4px=0.842, f1_4px_frame_only=0.831, f1_10px=0.897))  # ROADMAP·README에 기록된 test 43,628장 값
            v = register("vb_ball_detector", ball, r.info.run_id, {"stage": "finetuned"})
            print("vb_ball_detector v", v)
    player = ROOT / "yolo26m.pt"
    if player.exists():
        with mlflow.start_run(run_name="player_detector_yolo26m") as r:
            mlflow.set_tags({"kind": "model", "base": "yolo26m (COCO 사전학습, 파인튜닝 안 함)"})
            mlflow.log_params(dict(imgsz=1280, classes="person", tracker="botsort"))
            mlflow.log_metrics({k: v for k, v in dataset_metrics().items()})
            v = register("vb_player_detector", player, r.info.run_id, {"stage": "pretrained"})
            print("vb_player_detector v", v)
    print("기록 완료: mlflow ui --backend-store-uri", f"sqlite:///{ROOT / 'mlflow.db'}")


if __name__ == "__main__":
    main()
