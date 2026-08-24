# 주가예측 딥러닝 캡스톤 프로젝트

## 프로젝트 배경

기존에 단순 Transformer로 만든 주가예측 프로젝트를 처음부터 갈아엎는 졸업작품.
"AI 원클릭"처럼 보이지 않도록 구조적 참신함과 깊이를 확보하는 것이 최우선 목표.
섹터는 특정 업종(반도체 등)에 얽매이지 않음 — 데이터 수급 가능 여부가 우선.

## 데이터 파이프라인

- **데이터 소스: 키움증권 신규 REST API 하나로 통일** (과거 데이터 수집 + 모의투자 실행 모두)
  - KRX Data Marketplace, 한국투자증권 API, pykrx도 검토했으나 최종적으로 키움 단일 소스로 결정
  - 이유: 이미 키움 계좌 보유, 별도 계좌 개설 불필요, 모의투자까지 한 파이프라인으로 연결 가능
- **키움 API 연결은 MCP 서버로 수급** (`kiwoom-rest-api` npm 패키지, Claude Code MCP 도구로 등록)
  - 등록: `claude mcp add kiwoom -- npx -y kiwoom-rest-api`
  - 환경변수 `KIWOOM_ENV=mock`으로 모의투자, `live`로 실전투자 — 이 프로젝트는 항상 `mock` 사용, 실전 키는 이 서버에 넣지 않음
  - App Key/Secret은 `.env`로 분리, `.gitignore` 등록 필수 (코드에 하드코딩 금지)
  - 서드파티 커뮤니티 패키지이므로 실제 호출/응답을 코드에 반영하기 전 한 번씩 직접 검증할 것
- **키움 REST API 제약사항 (코드 작성 시 반드시 고려)**
  - TR(요청)별 초당 호출 제한 있음 → 레이트 리미터(토큰 버킷 또는 sleep 기반) 필수
  - 종목별 반복 호출 구조 — 여러 종목 동시 조회하는 벌크 엔드포인트 없음
  - 응답값이 전부 문자열 (`"+70000"`, `"1,234,567"` 형식) → 부호/쉼표 제거 후 숫자 변환하는 공통 파싱 유틸 필요
  - 실전투자 / 모의투자 base URL이 분리되어 있음 → 환경변수로 명확히 구분, 절대 혼용 금지
- **수집 전략**
  - 한 번 받은 데이터는 로컬(CSV/parquet 또는 SQLite)에 저장, 매번 재수집 금지
  - 이후에는 매일 장마감 후 당일치만 추가하는 증분 수집 구조로 전환
- **모의투자**는 모델 학습이 어느 정도 안정화된 후반부에 별도 파이프라인으로 구축

## 피처 설계

**원칙: 키움 REST API로만 수급 가능한 피처만 사용**

| 카테고리 | 피처 | 키움 API 확보 방법 |
|---|---|---|
| 가격/거래량 | OHLCV(수정주가), 거래대금, 등락률 → 리턴(수익률)으로 변환해 정상성 확보 | 일봉 차트 조회 TR |
| 기술적 지표 | MA(5/20/60), MACD, RSI, Bollinger Band, ATR, CCI | OHLCV로부터 직접 계산 (API 의존성 없음) |
| 매크로 지표 | KOSPI/KOSDAQ 지수, SOXX 등 해외지수/ETF (섹터에 따라 선택) | 지수 조회 TR, 해외주식 시세 조회 TR |
| 수급 데이터 | 외국인/기관/개인 순매수, PER, PBR | 투자자별매매동향 TR, 현재가 조회 TR(PER/PBR 포함) |
| 종목 고정 특성 (Static Covariate) | 업종, 시가총액 구간, 요일 임베딩 | 종목 기본정보 조회 TR |

## 모델 아키텍처

**단계적 구현 전략 — 반드시 이 순서로 진행:**

### Phase 1 (필수 완성 목표)
```
입력 → RevIN 정규화 → Patch 토큰화
     → Self-Attention 인코더 (순수 Transformer)
     → 글로벌-로컬 크로스어텐션 (국내종목 시퀀스 Query, 매크로/글로벌 지표 시퀀스 Key/Value)
     → TFT 변수 선택 네트워크 (Variable Selection Network, static/dynamic 분리)
     → Quantile 출력 헤드 (10/50/90 분위, Pinball Loss)
```
- RevIN: 종목별 인스턴스 정규화 → 출력 시 역변환. 비정상성(non-stationary) 데이터 필수 기법.
- Patch 토큰화: PatchTST 방식, 시점별이 아닌 구간(예: 5일) 단위로 토큰화
- 크로스어텐션: 반도체주라면 SOXX 등을 별도 시퀀스로 인코딩 후 연결
- TFT 변수 선택 네트워크: 피처 중요도를 자동 학습 → 해석가능성 리포트의 근거로 활용

### Phase 2 (시간 남으면 확장 — 스트레치 목표)
- Phase 1의 Self-Attention 인코더를 **Mamba(장기) + Transformer(단기) 병렬 전문가(expert) 구조**로 교체
- 주의: Mamba와 Transformer를 단순히 순서대로 쌓으면 정보 간섭으로 성능 저하 — 반드시 병렬 expert 구조로 분리
- Phase 1 대비 성능/해석가능성 비교 실험으로 보고서에 추가

**절대 원칙**: Phase 1이 안정적으로 완성되기 전에는 Phase 2로 넘어가지 않는다. Phase 2는 실패해도 졸업작품 제출에 지장 없어야 함.

## 자동매매 판단 기준 (모의투자 전용, 실전매매 아님)

모델의 Quantile 출력(10/50/90분위)을 아래 순서로 규칙화해서 매매 신호로 변환한다:

```
1. [기권 판단] 90분위 - 10분위 신뢰구간 폭이 임계값보다 넓으면 → 거래 안 함, 관망
2. [방향 판단] 신뢰구간이 충분히 좁으면 → 50분위(중앙값) 예측 수익률 확인
   - 임계값 이상이면 매수 신호, 이하이면 매도 신호
   - 임계값은 반드시 거래비용(수수료+슬리피지)보다 커야 함
3. [포지션 사이징] 신뢰구간이 좁을수록(확신할수록) 매수/매도 금액 비중을 키움
   - 균등 배분이 아니라 확신도에 비례 — Kelly Criterion 응용 가능
4. [리스크 오버레이] 최종 주문 전 반드시 아래 안전장치를 적용
   - 개별 종목 보유비중 상한 (예: 전체 자산의 10% 이내)
   - 손절매(stop-loss): 진입가 대비 -N% 하락 시 강제 청산
   - 일일 최대 거래횟수 제한 (과적합 신호로 인한 과도거래 방지)
5. 최종 결정된 주문을 키움 REST API 모의투자 매수/매도 엔드포인트로 전송
```

- 핵심 설계 원칙: "오를까/내릴까"뿐 아니라 "지금 판단해도 되는가"까지 모델이 결정하게 한다 — 신뢰구간 폭 기반 기권 로직이 이 프로젝트의 핵심 차별점 중 하나이므로 생략하지 말 것
- 임계값/포지션 사이징 공식의 정확한 수치는 처음부터 확정하지 말고 백테스트하면서 튜닝할 대상으로 취급

## 학습/평가 원칙 (기존 프로젝트에서 이어지는 노하우)

- Anti-overfitting: dropout, weight decay, Val Loss 기준 early stopping
- **날짜 기준 global train/val/test split 필수** — leakage 방지, 정규화 통계는 train 구간 기준으로만 계산
- 평가는 단순 방향 정확도가 아니라 **Sharpe/Calmar/Sortino 등 리스크 조정 지표** 중심으로
- 거래비용을 반영한 walk-forward 백테스트
- "하락장에서도 이긴다"는 프레이밍은 지양 — 리스크 조정 성과로 서술

## 코딩 시 참고

- Mac 환경, VS Code + Claude Code로 개발
- 큰 작업(Mamba 하이브리드 인코더 등) 시작 전엔 Plan Mode로 먼저 설계 검토 후 승인받고 진행
- 데이터 수집/전처리처럼 반복적이고 리스크 낮은 작업은 자동 진행 가능
- 학습/백테스트 단계는 결과를 직접 확인하면서 진행 (자동 반복 실행으로 GPU/시간 낭비 방지)

---

## 저장소 구조

```
Capstone_Stock_Price_Prediction/
├── CLAUDE.md                # 이 문서 — 프로젝트 헌법
├── configs/config.yaml      # 기간/하이퍼파라미터/임계값 단일 소스
├── data/{raw,interim,processed}/   # git 미추적
├── src/
│   ├── utils/       parsing(키움 문자열 파싱), ratelimit(토큰버킷), config, logging
│   ├── data/        kiwoom/{client,endpoints,collect}, storage(parquet 증분 저장)
│   ├── features/    technical(지표), build(피처 조립)
│   ├── models/      revin, patch_embed, encoder, cross_attention, vsn, quantile_head, phase1
│   ├── training/    dataset, split(날짜 기준), losses(pinball), train
│   ├── evaluation/  metrics(Sharpe/Calmar/Sortino), backtest(walk-forward, 거래비용 반영)
│   └── trading/     signal(분위→신호), risk(오버레이), paper_trader(모의투자 실행)
├── scripts/         CLI 엔트리포인트 (collect / build_features / train / backtest / paper_trade)
├── tests/           핵심 유틸·모델 shape 스모크 테스트
└── outputs/         checkpoints, figures, reports, logs (git 미추적)
```

## 개발 규칙

- 파이썬 3.11, 의존성은 `requirements.txt` 단일 관리. 가상환경은 `.venv/`.
- **비밀값은 절대 코드/커밋에 넣지 않는다.** `.env`만 사용하고 `.env.example`로 키 이름만 공유.
- `KIWOOM_ENV`(`mock` | `live`)로 base URL을 분기한다. 코드에 URL 하드코딩 금지.
  **이 프로젝트에서 `live`는 사용하지 않는다** — 기본값은 항상 `mock`, 실전 키는 저장조차 하지 않는다.
- MCP(`kiwoom-rest-api`)로 확인한 실제 응답 스키마는 `src/data/kiwoom/endpoints.py`의 TR 정의에 반영하고,
  검증 전까지는 해당 항목에 `# UNVERIFIED` 주석을 남긴다.
- 수집 스크립트는 항상 **증분(idempotent)** — 같은 명령을 두 번 실행해도 중복 행이 생기면 안 된다.
- 새 피처 추가 시 t 시점 이후 정보를 참조하지 않는지(look-ahead) 반드시 확인.
- 자동매매 신호 로직은 백테스트에서 쓰는 함수와 모의투자에서 쓰는 함수가 **동일한 코드**여야 한다
  (`src/trading/signal.py` 하나만 사용 — 백테스트/실행 로직 분기 금지).
- 실험 결과는 `outputs/reports/`에 날짜+설정 해시로 남긴다. 덮어쓰지 않는다.
