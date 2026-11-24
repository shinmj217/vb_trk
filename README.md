# vb_trk — 배구 중계 영상 전력분석

![ci](https://github.com/shinmj217/vb_trk/actions/workflows/ci.yml/badge.svg)

Offline volleyball tactical analysis from broadcast video: court and player tracking, event and rally detection, ball tracking, and a local replay viewer.

중계 영상 하나를 넣으면 **선수 추적 → 코트 위치(미터) → 팀 → 누가 언제 리시브·토스·스파이크했는지 → 공 궤적 → 공격 코스 → 랠리 기록**까지 뽑는 오프라인 파이프라인과, 그 결과를 영상과 함께 복기하는 로컬 뷰어입니다.
RTX 4060 Laptop (8GB) 한 대에서 학습·추론했습니다.

![demo](docs/demo.gif)

*Volleyball Dataset(Ibrahim et al., CVPR 2016)의 test 영상(학습에 안 쓴 Omaha 2015 경기)으로 만든 결과 화면입니다. 박스 = 선수(LEFT 빨강 / RIGHT 파랑), 위 글자 = 현재 이벤트, 노란 박스 = 공을 만진 선수, 노란 원·꼬리 = 공, 오른쪽 위 = 미니맵*

## 성능 한눈에

Volleyball Dataset 공식 test split 기준. 단계별 표와 실패한 시도는 [상세 결과](docs/RESULTS.md).

| 단계 | 지표 | 결과 |
|---|---|---|
| 선수 검출·추적 | 재현율 / 정밀도 (IoU≥0.5) | **0.847 / 0.973** |
| 개인 동작 5종 | 정확도 | **0.926** |
| 팀 행동 8종 | 정확도 | **0.924** |
| 이벤트 (팀 행동 + 공 만진 선수) | 정확도 | **0.874** |
| 공 검출 | F1, 중심 4px 이내 (TrackNetV2 0.836, WASB 0.880) | **0.842** |
| 속도 | 1시간 경기 (RTX 4060 Laptop) | 약 5시간 |

### 실제 중계 경기에서

학습에 안 쓴 방송 경기(2026 제주 한·중·일 여자배구 탑 클럽 슈퍼매치, 유튜브 구간)로 확인했습니다. 클로즈업·리플레이·풍경 컷이 섞이고 카메라가 계속 바뀌는 어려운 조건입니다. 득점 판정 규칙은 1세트로만 맞추고, 2·3세트는 손대지 않고 확인했습니다.

| | 1세트 (23분) | 2세트 앞 15분 | 3세트 (41분)* |
|---|---|---|---|
| 실제 득점 중 맞게 찾음 | 38 / 44 (가짜 9) | 26 / 27 (가짜 3) | 20 / 22 (가짜 5) |
| 랠리 시작의 서브 팀 | 34 / 38 | 21 / 25 | 16 / 21 |
| 코트 잡은 프레임 | 14,518 / 41,059 | 9,835 / 26,973 | 19,136 / 73,565 |

정답은 방송 점수판을 읽어 만든 **평가 전용** 파일이고, 파이프라인은 점수판을 쓰지 않습니다 (점수판 없는 영상에서도 돌아야 하므로). `score_match.py`로 재현합니다.
\* 3세트는 점수판이 최대 6분씩 가려져서 보이는 구간의 22점만 채점했습니다. 방송 화면은 저작권 때문에 저장소에 싣지 않았습니다.

![match1](docs/match1_set1_analysis.png)

*1세트 분석 (LEFT = 화면 왼쪽 코트): 위치 히트맵, 이벤트 위치, 서브 순간 포메이션, 선수별 이동, 공격 코스, 이벤트 타임라인 (영상 프레임 없이 좌표만 그린 그래프)*

## 파이프라인

설계 원칙은 **역할 분리**입니다. 코트는 학습 없는 SAM3, 사람은 YOLO, 동작은 시간축 분류기이고, 팀은 유니폼 색이 아니라 *코트 위치*로 판정합니다 (리베로·조명 변화에 강함).

```
영상 ─┬─ SAM3 (코트·네트 기둥 마스크, 10프레임마다) → 코트 변 + 네트 선 → 호모그래피 + 흰 라인 ICP
      │     → 흰 라인 일치도 검증 → 튀는 값 제거·보간
      └─ YOLO26m (사람) + BoT-SORT → 발 위치를 코트 좌표(m)로 → 팀 = 네트 기준 코트 반쪽 (LEFT/RIGHT)
             → 끊긴 트랙 재연결, 트랙 시간의 절반 이상 코트 안에 있는 트랙만 선수 (심판·코치·벤치 제외)   run.py

      ┌─ 선수 크롭 → YOLO26s-cls 특징 → ±8프레임 시간축 헤드 → 선수별 5종 (block/dig/set/spike/other)
      ├─ 프레임 전체 → YOLO26m-cls 특징 → ±8프레임 시간축 헤드 → 팀 행동 8종
      └─ 융합(로지스틱 회귀) → 이벤트 (팀 × pass/set/spike/winpoint) + 공 만진 선수
             → 득점 판정 거르기 (뒤에 동작이 이어지면 랠리 중, 10초 안 득점은 하나로)                events.py

영상 ─── YOLO26s 1280 (공, 파인튜닝) → 프레임당 후보 5개 → 앞뒤 프레임 지지·구멍 보간               ball_video.py

→ 히트맵, 이벤트 지도, 서브 순간 포메이션, 이동 거리·속도, 공격 코스, 타임라인,
  랠리 단위 기록 (서브 팀·이긴 팀·접촉 수, 서브 득점률·사이드아웃률)                              analysis.py
```

## 결과 뷰어

분석은 명령줄로 따로 돌리고, 결과는 브라우저에서 봅니다. 인터넷 없이 동작합니다.

```bash
python app.py        # http://localhost:8000 이 열린다 (--outputs, --data, --port, --no-browser)
```

- **복기 탭:** 랠리를 고르면 그 구간의 영상이 재생되고, 영상 위에 선수 박스·공 꼬리·이벤트 글자가 겹쳐 그려집니다. 코트 지도에는 선수 위치와 접촉 위치(번호순)가 나옵니다. 스페이스(재생), ←→(5초 이동), F(전체화면), ↑↓(이전·다음 랠리).
- **경기 통계 탭:** 팀별 득점, 서브 득점률, 사이드아웃률과 분석 그림.
- **영상 올리기·분석 탭:** mp4를 올리고 `분석 시작`을 누르면 5단계(코트 → 추적 → 이벤트 → 공 → 분석)가 서버와 분리된 프로세스로 돕니다. 단계별 진행바·남은 시간(추정)·취소·재시작(끝난 단계는 건너뜀)이 있고, 창을 닫아도 계속 돕니다. 분석 중에 다른 영상을 시작하면 대기열에 들어갑니다.
- **필요한 것:** `outputs/<이름>_{rallies,events,tracks,ball}.csv`(파이프라인 산출물)와 `data/<이름>.mp4`. 영상은 저장소에 없습니다.
- **솔직한 한계:** 득점 팀은 '다음 랠리의 서브 팀'으로 추정한 값이라 약 85%만 맞고, 화면 위에 상시 표시됩니다. 코트를 못 잡은 구간(클로즈업·리플레이)은 박스가 없고 타임라인에 회색으로 표시됩니다. 분석은 40분짜리 한 세트에 몇 시간 걸리고 GPU를 거의 다 씁니다. 브라우저는 Chrome·Edge 기준(VP9 재생).

**exe로 열기:** `build_exe.bat`가 `vb_viewer.exe`(약 80MB, 뷰어만)를 만듭니다. 프로젝트 폴더에 두고 더블클릭하면 열립니다. 분석 실행에는 이 폴더의 Python 환경이 필요하고(`VB_PYTHON`으로 지정 가능), 결과 보기·업로드·삭제는 Python 없이도 됩니다.

## 빠른 시작

```bash
conda create -n vbtrk python=3.11 -y && conda activate vbtrk
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
python -m pytest -q          # 영상·GPU 없이 도는 테스트 (66개; 모델 파일이 없으면 서빙 테스트는 건너뜀)
```

- SAM3: Hugging Face `facebook/sam3` (접근 승인 필요). 한 번 받아 두면 오프라인으로 씁니다.
- YOLO 가중치는 ultralytics가 처음 실행 때 자동으로 받습니다. 모델 파일(`*.pt`)은 저장소에 없고 DVC 포인터만 있습니다.
- 데이터: [Volleyball Dataset](https://github.com/mostafa-saad/deep-activity-rec) (연구용, 저장소에 포함 안 함).

```bash
python run.py --video data/long_20.mp4         # 1) 추적  → outputs/<영상>_tracks.csv
python events.py --video data/long_20.mp4      # 2) 이벤트 → _events.csv
python ball_video.py --video data/long_20.mp4  # 3) 공    → _ball.csv
python analysis.py --video data/long_20.mp4    # 4) 분석  → _analysis.png, _rallies.csv
```

학습·평가를 다시 돌리는 명령은 [docs/REPRODUCE.md](docs/REPRODUCE.md).

## MLOps

| 항목 | 구성 | 확인 |
|---|---|---|
| 서빙 | `serve.py` (FastAPI `/detect`, `/health`, `/metrics`) | 서버·테스트 |
| 컨테이너 | `Dockerfile`, `docker-compose.yml` (서비스 + Prometheus + Grafana) | Docker Desktop에서 빌드·기동 |
| 실험·모델 관리 | MLflow (`mlops/log_mlflow.py`), 모델 레지스트리 | 실행·조회 |
| 데이터·가중치 버전 | DVC (영상·가중치 추적, 원격에서 복원 검증) | 확인 |
| 모니터링 | Prometheus 경고 규칙 3개, Grafana 대시보드 | 컨테이너에서 쿼리 확인 |
| 드리프트 | Evidently (`mlops/drift.py`) → 리포트 + `retrain` 플래그 | 표본이 적어 보정 필요 |
| CI | GitHub Actions (CPU torch, pytest) | 통과 |

MLflow·DVC·Evidently는 파이프라인 환경을 건드리지 않게 별도 가상환경 `.venv-mlops`에 설치했습니다. `serve.py`는 프레임의 사람 전부(관중 포함)를 돌려주고, 선수 판정은 영상 전체를 보는 `run.py`가 합니다.

## 구조

```
vb_trk/
├─ run.py  events.py  ball_video.py  analysis.py   파이프라인 4단계 (진입점)
├─ app.py                                           결과 뷰어 실행 (실제 코드는 viewer/)
├─ serve.py   score_match.py                        검출 모델 서빙, 실제 경기 채점
├─ vbtrk/     코트(SAM3 → 호모그래피)·팀·시각화 + 동작·공 데이터·학습·평가 (action_*, ball_*, eval)
├─ viewer/    결과 뷰어: server.py, viewer_data.py, analysis_jobs.py, ui/
├─ mlops/     MLflow 기록, 드리프트 감지
├─ monitoring/  Prometheus·Grafana 설정   (Dockerfile, docker-compose.yml 은 루트)
├─ tests/     테스트 66개 (conftest.py 가 저장소 루트를 import 경로에 넣는다)
├─ tools/     보조 도구 (긴 테스트 영상 만들기)
├─ data/ models/ *.dvc   영상·가중치는 DVC 포인터만
└─ docs/      상세 결과, 재현 방법, design/ (뷰어 설계서·구현 계획)
```

학습·평가 도구는 패키지 안에 있어서 `python -m vbtrk.eval`, `python -m vbtrk.action_data`처럼 실행합니다 ([재현 방법](docs/REPRODUCE.md)).

## 한계

- 득점 팀 판정이 약 85%라 서브 득점률·사이드아웃률은 ±2~3 오차가 있습니다. 서브는 서버를 직접 보지 않고 '득점 뒤 코트에 선수 8명 이상 2초 + 직전 득점 팀이 서브'로 추정합니다 (중계가 서브 직전 클로즈업이라 서버가 거의 안 보임).
- 위에서 내려다보는 리플레이 앵글은 코트 인식이 36%입니다 (SAM3 마스크가 코트 전체를 못 잡음). 득점 판정에 가짜가 남습니다 (1세트 9개).
- 득점 규칙 값(3초·10초)은 한 경기로 맞췄습니다. 다른 대회·중계에는 재확인이 필요합니다.
- 먼 쪽 네트 앞 선수 트랙이 끊깁니다 (선수 ID 15~16 / 실제 12). 같은 유니폼이라 색·ReID로는 한계가 있습니다.
- 학습 데이터에 '동작 없음' 구간이 없어 모든 프레임이 8종 팀 행동 중 하나로 분류됩니다.
- 공격 코스 도착점은 상대 리시브 선수 발 위치 근사라, 블로커 터치도 리시브로 잡혀 '짧게'가 과대합니다.

개발 과정의 상세 기록(실패한 시도 포함)은 [ROADMAP.md](ROADMAP.md)에 있습니다.

## 라이선스·이용 조건

- 코드: **AGPL-3.0** ([LICENSE](LICENSE)). 선수·공 검출에 쓰는 Ultralytics YOLO가 AGPL-3.0이라 같은 조건을 따릅니다.
- 저장소에 **없는 것**: 모델 가중치, 실험 영상, 방송 중계 영상. SAM3(`facebook/sam3`)는 Hugging Face에서 이용 조건에 동의해야 받을 수 있습니다.
- 데이터: Volleyball Dataset (Ibrahim et al., CVPR 2016), 공 위치 레이블 (Perez et al., 2022). 연구용이며 각 출처의 조건과 인용 방식을 따릅니다 (공 레이블은 라이선스 표기가 없음).
- `docs/demo.gif`·`docs/analysis.png`는 Volleyball Dataset 영상으로 만든 결과 그림이고, 방송 경기 분석 그림(`match1_set1_analysis.png`)은 영상 프레임 없이 좌표만 그린 그래프입니다.
