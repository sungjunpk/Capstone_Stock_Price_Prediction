# src/training — 학습·분할·손실

> 루트 [`CLAUDE.md`](../../CLAUDE.md)의 절대 규칙이 우선한다.

## 분할 (leakage 방지 — 가장 중요)

- **날짜 기준 global train/val/test split 필수.** 종목별 분할이 아니다
- **정규화 통계는 train 구간에서만 계산한다.** val/test를 통계에 참여시키면 미래를 훔쳐본다
- 구간 경계에 **embargo**를 둔다. 타깃이 t+h를 보므로, embargo 없이 자르면
  train 마지막 샘플의 라벨이 val 구간을 훔쳐본다

현재 분할 (`configs/config.yaml`):

| 구간 | 기간 | 샘플 |
|---|---|---|
| train | ~2022-12-31 | 260,339 |
| val | ~2023-12-31 | 17,584 |
| test | 2024-01 ~ | 74,953 |

(146종목 기준. 8종목이던 시절 val이 968샘플이라 early stopping이 불안정할 우려가 있었는데,
유니버스 확대로 해소됐다.)

## 손실

Pinball(Quantile) Loss로 10/50/90 분위를 동시에 학습한다.

- `crossing_penalty`: 분위가 단조증가하지 않는 만큼 벌점. 이미 정렬돼 있으면 0
- 분위 교차를 방치하면 매매 신호의 기권 로직이 무너진다 (`src/trading/` 참고)

## Anti-overfitting (기존 프로젝트에서 이어지는 노하우)

- dropout, weight decay
- **Val Loss 기준 early stopping** — train loss로 판단하지 않는다
- 학습샘플 약 35만, 파라미터 1.88M. 과소적합이 확인되기 전에는 모델을 키우지 않는다

## Dataset 메모리 제약

학습샘플 35만 × lookback 120 × 피처 17 × 4바이트 = **2.9GB**.
윈도우를 미리 펼치면 터진다. 종목별 연속 배열은 28MB뿐이므로
`dataset.py` 는 배열을 한 번만 만들고 `__getitem__` 에서 구간을 잘라 준다.
이 계약은 `tests/test_dataset.py::test_no_window_duplication_in_memory` 가 지킨다.

종목마다 거래일이 다르다(공통 1,887 / 전체 2,857 — 거래정지 제거 때문).
공통 캘린더를 강요하면 34%가 날아가므로 **각 종목의 자기 행 위에서** 윈도우를 만든다.

## 학습 실행 위치

맥북(MPS)은 epoch당 약 19분(260k 샘플, batch 64)이라 전체 학습에 16시간이 걸린다.
학습은 외부 GPU로 옮긴다. **Kaggle 권장** — 주당 30시간, 세션 12시간이라 Colab 무료보다 길다.
절차: [`docs/KAGGLE_SETUP.md`](../../docs/KAGGLE_SETUP.md) / 노트북: `notebooks/train_kaggle.ipynb`
(Colab은 `notebooks/train_colab.ipynb`).
`train.py` 가 디바이스를 감지해 AMP/워커/pin_memory 를 자동 조정하므로
같은 명령이 양쪽에서 그대로 돈다.

## 워크플로 주의

학습 단계는 **결과를 직접 확인하면서** 진행한다.
자동 반복 실행으로 GPU/시간을 낭비하지 말 것 (루트 CLAUDE.md 워크플로 규칙).
