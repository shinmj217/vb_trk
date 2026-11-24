# 평가·학습 재현

[← README로](../README.md)

각 단계의 학습·평가 명령. 데이터(Volleyball Dataset 등)와 GPU 환경이 필요하다. 설치는 README의 '빠른 시작' 참고.

```
python -m vbtrk.eval --per-video 5                            # 1차 검출·추적 평가
python -m vbtrk.action_data                                   # 선수 크롭 데이터셋
yolo classify train model=yolo26s-cls.pt data=data/actions imgsz=224 epochs=30 project=outputs/action_cls name=base
python -m vbtrk.action_temporal --merge --out outputs/action_cls/temporal_head5.pt
python -m vbtrk.action_temporal --out outputs/action_cls/temporal_head.pt
python -m vbtrk.action_frames data                            # 프레임 데이터셋 (좌우 반전 + l/r 교환)
yolo classify train model=yolo26m-cls.pt data=data/team_frames imgsz=448 fliplr=0 epochs=30 batch=32 project=outputs/action_cls name=frames_m
python -m vbtrk.action_frames feats m && python -m vbtrk.action_frames head m
python -m vbtrk.action_team                                   # 조합 비교 + 융합 분류기 저장
python -m vbtrk.ball_data && yolo detect train model=yolo26s.pt data=data/ball/ball.yaml imgsz=1280 epochs=30 batch=4 project=outputs/ball name=y26s_1280
python -m vbtrk.ball_eval && python -m vbtrk.ball_track               # 공 검출 평가 (단일 프레임 / 궤적 후처리)
python tools/stitch_clip.py 20 49840 50210 data/long_20.mp4   # 긴 테스트 영상 만들기
```
