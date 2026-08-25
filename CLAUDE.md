# 주가예측 딥러닝 캡스톤 프로젝트

## 프로젝트 배경

기존에 단순 Transformer로 만든 주가예측 프로젝트를 처음부터 갈아엎는 졸업작품.
"AI 원클릭"처럼 보이지 않도록 **구조적 참신함과 깊이를 확보**하는 것이 최우선 목표.
섹터는 특정 업종에 얽매이지 않음 — 데이터 수급 가능 여부가 우선.

핵심 차별점 두 가지:
1. **기권 로직** — "오를까/내릴까"뿐 아니라 "지금 판단해도 되는가"까지 모델이 결정
2. **해석가능성** — TFT 변수선택망으로 피처 중요도를 자동 학습해 리포트 근거로 사용

## 절대 규칙 (위반 금지)

1. **`live` 금지.** `KIWOOM_ENV`는 항상 `mock`. 실전 키는 저장조차 하지 않는다.
   `src/utils/config.py`가 `live`를 예외로 막고 있다 — 이 방어를 풀지 말 것.
2. **비밀값은 코드/커밋에 넣지 않는다.** `.env`만 사용하고 `.env.example`에는 키 이름만.
3. **Phase 1이 안정적으로 완성되기 전에 Phase 2로 넘어가지 않는다.**
   Phase 2는 실패해도 제출에 지장 없어야 한다.
4. **수집은 항상 증분(idempotent).** 같은 명령을 두 번 실행해도 중복 행이 생기면 안 된다.
5. **look-ahead 금지.** 새 피처가 t 시점 이후 정보를 참조하지 않는지 반드시 확인한다.
6. **정규화 통계는 train 구간에서만 계산한다.** 날짜 기준 global split 필수.
7. **매매 신호는 `src/trading/signal.py` 하나만 사용한다.**
   백테스트와 모의투자가 같은 코드를 공유 — 실행 경로별 분기 금지.
8. **실험 결과는 `outputs/reports/`에 날짜+설정 해시로 남긴다.** 덮어쓰지 않는다.

## 상세 문서 (작업할 폴더의 것만 읽으면 된다)

| 문서 | 내용 |
|---|---|
| [`src/data/CLAUDE.md`](src/data/CLAUDE.md) | 키움 API 제약, 수집 전략, TR 정의 규칙 |
| [`src/features/CLAUDE.md`](src/features/CLAUDE.md) | 피처 설계표, look-ahead 방지, 스케일 정규화 |
| [`src/models/CLAUDE.md`](src/models/CLAUDE.md) | Phase 1/2 아키텍처 상세 |
| [`src/training/CLAUDE.md`](src/training/CLAUDE.md) | 분할 규칙, 손실, anti-overfitting |
| [`src/evaluation/CLAUDE.md`](src/evaluation/CLAUDE.md) | 평가 지표, walk-forward 백테스트 |
| [`src/trading/CLAUDE.md`](src/trading/CLAUDE.md) | 매매 판단 5단계, 기권 로직 |
| [`docs/KIWOOM_VERIFY.md`](docs/KIWOOM_VERIFY.md) | 실제 API 응답 검증 결과와 함정 |
| [`docs/KAGGLE_SETUP.md`](docs/KAGGLE_SETUP.md) | 클라우드 GPU 학습 절차 (수집은 로컬, 학습만 외부) |
| [`README.md`](README.md) | 현재 진행 상태, 다음 할 일, 세션 시작 절차 |

**진행 상황은 이 문서가 아니라 `README.md`에 적는다.** 여기는 규칙만 담는다.

## 개발 환경

- 파이썬 3.11, 가상환경 `.venv/`, 의존성은 `requirements.txt` 단일 관리
- Mac (MPS 백엔드), VS Code + Claude Code
- 설정은 `configs/config.yaml` 단일 소스 — 코드에 숫자를 하드코딩하지 않는다

## 작업 방식

- **큰 작업(모델 아키텍처 등)은 Plan Mode로 설계 검토 후 승인받고 진행**
- 데이터 수집/전처리처럼 반복적이고 리스크 낮은 작업은 자동 진행 가능
- **학습/백테스트는 결과를 직접 확인하면서 진행** — 자동 반복 실행으로 GPU/시간 낭비 금지
- 커밋은 작업 단위로. 커밋 메시지가 곧 작업 이력이다

## 저장소 구조

```
Capstone_Stock_Price_Prediction/
├── CLAUDE.md                # 이 문서 — 규칙과 인덱스
├── README.md                # 현재 상태 / 다음 할 일
├── configs/config.yaml      # 기간·하이퍼파라미터·임계값 단일 소스
├── data/{raw,interim,processed}/   # git 미추적
├── docs/                    # API 검증 결과 등
├── src/
│   ├── utils/       parsing(키움 문자열), ratelimit(토큰버킷), config, logging, seed
│   ├── data/        kiwoom/{client,endpoints,collect}, storage(parquet 증분)
│   ├── features/    technical(지표), build(조립)
│   ├── models/      revin, patch_embed, encoder, cross_attention, vsn, quantile_head, phase1
│   ├── training/    dataset, split(날짜 기준), losses(pinball), train
│   ├── evaluation/  metrics(Sharpe/Calmar/Sortino), backtest(walk-forward)
│   └── trading/     signal(분위→신호), risk(오버레이), paper_trader(모의투자 실행)
├── scripts/         CLI (collect / peek / build_features / train / backtest / paper_trade)
├── tests/           핵심 유틸·모델 스모크 테스트
└── outputs/         checkpoints, figures, reports, logs (git 미추적)
```
