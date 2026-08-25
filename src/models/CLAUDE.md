# src/models — 모델 아키텍처

> 루트 [`CLAUDE.md`](../../CLAUDE.md)의 절대 규칙이 우선한다.
> ⚠️ 이 폴더 작업은 **Plan Mode로 설계 승인 후** 시작한다.

## 단계적 구현 전략 — 반드시 이 순서로

### Phase 1 (필수 완성 목표)

```
입력 → RevIN 정규화 → Patch 토큰화
     → Self-Attention 인코더 (순수 Transformer)
     → 글로벌-로컬 크로스어텐션 (국내종목 시퀀스 Query, 매크로/글로벌 지표 시퀀스 Key/Value)
     → TFT 변수 선택 네트워크 (Variable Selection Network, static/dynamic 분리)
     → Quantile 출력 헤드 (10/50/90 분위, Pinball Loss)
```

| 구성요소 | 역할 | 파일 |
|---|---|---|
| RevIN | 종목별 인스턴스 정규화 → 출력 시 역변환. 비정상성(non-stationary) 데이터 필수 기법 | `revin.py` |
| Patch 토큰화 | PatchTST 방식. 시점별이 아닌 **구간(5일) 단위** 토큰화 | `patch_embed.py` |
| Self-Attention 인코더 | 순수 Transformer | `encoder.py` |
| 크로스어텐션 | 매크로/글로벌 지표를 별도 시퀀스로 인코딩 후 연결 | `cross_attention.py` |
| TFT 변수 선택망 | 피처 중요도 자동 학습 → **해석가능성 리포트의 근거** | `vsn.py` |
| Quantile 헤드 | 10/50/90 분위 동시 출력 | `quantile_head.py` |
| 조립 | 위를 엮은 Phase 1 모델 | `phase1.py` |

**구현 완료 (2026-08-25).** 1.88M 파라미터. 확정된 설계 판단 두 가지:

- **VSN 을 인코더 앞에 둔다** (TFT 원논문 순서). CLAUDE.md 본문의 나열 순서는
  "인코더 → 크로스어텐션 → VSN" 이지만, 그렇게 하면 인코더가 채널 수(17)만큼
  반복 실행돼 forward 가 5배 느려진다(실측 259ms → 52ms).
  VSN 은 패치 임베딩을 받아 채널을 가중합하므로 해석가능성 근거는 그대로 남는다.
  `tests/test_models.py::test_encoder_runs_once_not_per_channel` 이 순서를 고정한다.
- **매크로는 채널을 먼저 합치고 한 번만 인코딩한다.** 매크로에는 VSN 이 없어
  채널을 살려둘 이유가 없는데, 채널별로 인코딩하면 비용만 13배가 된다.
- **채널별 GRN 은 einsum 으로 묶는다**(`GroupedGRN`). 17개를 파이썬 루프로 돌면
  커널이 68회 뜨고 GPU 가 논다.
- **분위 단조성을 구조로 보장** — `q10, q10+softplus(δ1), …` 누적 방식.
  손실 페널티만으로는 교차가 남을 수 있고, 교차가 나면 `trading/signal.py` 가
  예외를 던져 백테스트가 통째로 죽는다. 페널티(`crossing_weight=0.01`)는 보조로 유지.

### Phase 2 (스트레치 목표 — 시간 남을 때만)

- Phase 1의 Self-Attention 인코더를 **Mamba(장기) + Transformer(단기) 병렬 전문가(expert) 구조**로 교체
- ⚠️ Mamba와 Transformer를 단순히 순서대로 쌓으면 **정보 간섭으로 성능 저하** —
  반드시 병렬 expert 구조로 분리할 것
- Phase 1 대비 성능/해석가능성 비교 실험을 보고서에 추가
- `mamba-ssm`은 CUDA 전용 — Mac에서는 순수 PyTorch 대체 구현이 필요하다

## 절대 원칙

**Phase 1이 안정적으로 완성되기 전에는 Phase 2로 넘어가지 않는다.**
Phase 2는 실패해도 졸업작품 제출에 지장이 없어야 한다.

## 출력 계약

Quantile 헤드는 `(B, ..., 3)` 형태로 [q10, q50, q90]을 낸다.
**분위 교차(q10 > q50)가 생기면 신뢰구간 폭 기반 기권 로직이 무너진다.**
`src/training/losses.py`의 `crossing_penalty`로 억제하고,
`src/trading/signal.py`의 `QuantilePrediction`이 교차를 발견하면 예외를 던진다.

## 데이터 규모 참고

현재 학습샘플 약 18,900개(8종목, lookback 120일). 딥러닝 기준 크지 않다.
파라미터를 보수적으로 가져가고 dropout·weight decay를 충분히 걸 것.
현재 config 기본값(`d_model=128`, `n_layers=3`, `dropout=0.2`)은 이 규모에 맞춰 잡은 것이다.
