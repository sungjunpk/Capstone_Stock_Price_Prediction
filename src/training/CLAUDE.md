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
| train | ~2022-12-31 | 13,818 |
| val | ~2023-12-31 | 968 |
| test | 2024-01 ~ | 4,095 |

⚠️ val 968샘플은 early stopping 판단이 불안정할 수 있는 규모다.
유니버스를 늘리면 함께 커진다 — 종목 추가 시 다시 볼 것.

## 손실

Pinball(Quantile) Loss로 10/50/90 분위를 동시에 학습한다.

- `crossing_penalty`: 분위가 단조증가하지 않는 만큼 벌점. 이미 정렬돼 있으면 0
- 분위 교차를 방치하면 매매 신호의 기권 로직이 무너진다 (`src/trading/` 참고)

## Anti-overfitting (기존 프로젝트에서 이어지는 노하우)

- dropout, weight decay
- **Val Loss 기준 early stopping** — train loss로 판단하지 않는다
- 데이터가 작으므로(학습샘플 ~18,900) 모델을 키우기보다 정규화를 먼저 조인다

## 워크플로 주의

학습 단계는 **결과를 직접 확인하면서** 진행한다.
자동 반복 실행으로 GPU/시간을 낭비하지 말 것 (루트 CLAUDE.md 워크플로 규칙).
