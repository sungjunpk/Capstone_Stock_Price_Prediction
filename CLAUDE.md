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
   추론도 마찬가지다: `src/models/inference.py` 하나만 쓴다.
8. **실험 결과는 `outputs/reports/`에 날짜+설정 해시로 남긴다.** 덮어쓰지 않는다.
9. **주문은 기본이 dry-run 이다.** `scripts/paper_trade.py --execute` 로만 나간다.
   대시보드에서 주문을 낼 수 있게 만들지 않는다 — 화면은 읽기 전용이다.
   **자동 매매(2026-08-26~)도 이 규칙 안에 있다.** `scripts/daily_trade.sh` 가
   평일 15:15 에 도는데, 하는 일은 위 명령을 부르는 것뿐이다 — 새 주문 경로를
   만들지 않는다. 설치는 수집 자동화와 **분리**돼 있다
   (`install_daily_trade.sh`) — 주문을 내는 자동화는 스스로 켜는 동작이어야 한다.

## 작업 원칙

> 이 지침은 **속도보다 신중함** 쪽으로 치우쳐 있다. 사소한 작업에는 판단해서 적용한다.

### 1. 코딩 전에 생각한다

가정하지 않는다. 헷갈리는 것을 숨기지 않는다. 트레이드오프를 드러낸다.

구현을 시작하기 전에:

- **가정을 명시한다.** 확신이 없으면 묻는다.
- 해석이 여러 갈래면 **전부 제시한다.** 혼자 하나를 고르고 넘어가지 않는다.
- 더 단순한 방법이 있으면 말한다. 근거가 있으면 **반대 의견을 낸다.**
- 불분명하면 **멈춘다.** 무엇이 헷갈리는지 이름 붙이고 묻는다.

### 2. 단순함이 먼저다

문제를 푸는 최소한의 코드. 미리 짐작해서 만들어두지 않는다.

- 요청받지 않은 기능을 넣지 않는다.
- 한 번만 쓰는 코드에 추상화를 만들지 않는다.
- 요청받지 않은 "유연성"·"설정 가능성"을 넣지 않는다.
- 일어날 수 없는 상황에 대한 예외 처리를 넣지 않는다.
- 200줄을 썼는데 50줄로 되겠다면 다시 쓴다.

스스로에게 묻는다: **"시니어 엔지니어가 이걸 보고 과하다고 할까?"** 그렇다면 단순하게 만든다.

### 3. 최소 침습 변경

건드려야 하는 것만 건드린다. 치우는 건 내가 어지른 것만.

기존 코드를 수정할 때:

- 옆에 있는 코드·주석·서식을 "개선"하지 않는다.
- 안 깨진 것을 리팩터링하지 않는다.
- 내 취향과 달라도 **기존 스타일에 맞춘다.**
- 무관한 죽은 코드를 발견하면 **말만 한다.** 지우지 않는다.

내 변경이 고아를 만들었을 때:

- **내 변경 때문에** 안 쓰이게 된 import·변수·함수는 지운다.
- 원래부터 죽어 있던 코드는 요청받기 전엔 건드리지 않는다.

판정 기준: **바뀐 모든 줄이 사용자의 요청으로 곧장 추적되는가.**

### 4. 목표 기반 실행

성공 기준을 먼저 정의하고, 검증될 때까지 돈다.

작업을 검증 가능한 목표로 바꾼다:

| 모호한 요청 | 검증 가능한 목표 |
|---|---|
| "검증 로직 추가" | 잘못된 입력에 대한 테스트를 먼저 쓰고, 통과시킨다 |
| "버그 수정" | 버그를 재현하는 테스트를 먼저 쓰고, 통과시킨다 |
| "X 리팩터링" | 리팩터링 전후로 테스트가 모두 통과하는지 확인한다 |

여러 단계짜리 작업은 짧은 계획을 먼저 밝힌다:

```
1. [단계] → 검증: [무엇을 확인하면 끝난 것인가]
2. [단계] → 검증: [...]
```

**성공 기준이 강하면 혼자 반복해서 끝낼 수 있다.**
"동작하게 해줘" 같은 약한 기준은 매 단계 되물어야 한다.

### 이 프로젝트에서는

- **큰 작업(모델 아키텍처 등)은 Plan Mode로 설계 검토 후 승인받고 진행**
- 데이터 수집/전처리처럼 반복적이고 리스크 낮은 작업은 자동 진행 가능
- **학습/백테스트는 결과를 직접 확인하면서 진행** — 자동 반복 실행으로 GPU/시간 낭비 금지
- 커밋은 작업 단위로. 커밋 메시지가 곧 작업 이력이다

### 잘 지켜지고 있다는 신호

- diff 에 불필요한 변경이 줄어든다
- 과하게 만들어서 다시 쓰는 일이 줄어든다
- 확인 질문이 **실수한 뒤가 아니라 구현 전에** 나온다

## 상세 문서 (작업할 폴더의 것만 읽으면 된다)

| 문서 | 내용 |
|---|---|
| [`src/data/CLAUDE.md`](src/data/CLAUDE.md) | 키움 API 제약, 수집 전략, TR 정의 규칙 |
| [`src/features/CLAUDE.md`](src/features/CLAUDE.md) | 피처 설계표, look-ahead 방지, 스케일 정규화 |
| [`src/models/CLAUDE.md`](src/models/CLAUDE.md) | Phase 1/2 아키텍처 상세 |
| [`src/training/CLAUDE.md`](src/training/CLAUDE.md) | 분할 규칙, 손실, anti-overfitting |
| [`src/evaluation/CLAUDE.md`](src/evaluation/CLAUDE.md) | 평가 지표, walk-forward 백테스트 |
| [`src/trading/CLAUDE.md`](src/trading/CLAUDE.md) | 매매 판단 5단계, 기권 로직, 모의투자 실행 |
| [`src/webapp/CLAUDE.md`](src/webapp/CLAUDE.md) | 대시보드 — 읽기 전용 원칙 |
| [`docs/KIWOOM_VERIFY.md`](docs/KIWOOM_VERIFY.md) | 실제 API 응답 검증 결과와 함정 |
| [`docs/KAGGLE_SETUP.md`](docs/KAGGLE_SETUP.md) | 클라우드 GPU 학습 절차 (수집은 로컬, 학습만 외부) |
| [`README.md`](README.md) | 현재 진행 상태, 다음 할 일, 세션 시작 절차 |

**진행 상황은 이 문서가 아니라 `README.md`에 적는다.** 여기는 규칙만 담는다.

> Phase 1 구현에서 확정된 설계 변경 하나: **VSN 은 인코더 앞에 둔다**(TFT 원논문 순서).
> 성능상 5배 차이가 나고 해석가능성은 유지된다. 근거는 `src/models/CLAUDE.md`.

## 개발 환경

- 파이썬 3.11, 가상환경 `.venv/`, 의존성은 `requirements.txt` 단일 관리
- Mac (MPS 백엔드), VS Code + Claude Code
- 설정은 `configs/config.yaml` 단일 소스 — 코드에 숫자를 하드코딩하지 않는다

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
│   ├── models/      revin, patch_embed, encoder, cross_attention, vsn, quantile_head,
│   │                phase1, inference(백테스트·모의투자 공용 추론)
│   ├── training/    dataset, split(날짜 기준), losses(pinball), train
│   ├── evaluation/  metrics(Sharpe/Calmar/Sortino), backtest(walk-forward)
│   ├── trading/     signal(분위→신호), risk(오버레이), broker(키움 계좌·주문),
│   │                paper_trader(모의투자 실행)
│   └── webapp/      대시보드 (collect/render/server — 읽기 전용, 의존성 없음)
├── scripts/         CLI (collect / peek / build_features / train / backtest /
│                         verify_trading_trs / paper_trade / dashboard)
├── tests/           핵심 유틸·모델 스모크 테스트
└── outputs/         checkpoints, figures, reports, logs (git 미추적)
```
