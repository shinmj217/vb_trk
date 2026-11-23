# 평가·학습 재현

[← README로](../README.md)

각 단계의 학습·평가 명령. 데이터(Volleyball Dataset 등)와 GPU 환경이 필요하다. 설치는 README의 '빠른 시작' 참고.

```
python eval.py --per-video 5                            # 1차 검출·추적 평가
python action_data.py                                   # 선수 크롭 데이터셋
yolo classify train model=yolo26s-cls.pt data=data/actions imgsz=224 epochs=30 project=outputs/action_cls name=base
python action_temporal.py --merge --out outputs/action_cls/temporal_head5.pt
python action_temporal.py --out outputs/action_cls/temporal_head.pt
python action_frames.py data                            # 프레임 데이터셋 (좌우 반전 + l/r 교환)
yolo classify train model=yolo26m-cls.pt data=data/team_frames imgsz=448 fliplr=0 epochs=30 batch=32 project=outputs/action_cls name=frames_m
python action_frames.py feats m && python action_frames.py head m
python action_team.py                                   # 조합 비교 + 융합 분류기 저장
python ball_data.py && yolo detect train model=yolo26s.pt data=data/ball/ball.yaml imgsz=1280 epochs=30 batch=4 project=outputs/ball name=y26s_1280
python ball_eval.py && python ball_track.py               # 공 검출 평가 (단일 프레임 / 궤적 후처리)
python stitch_clip.py 20 49840 50210 data/long_20.mp4   # 긴 테스트 영상 만들기
```
