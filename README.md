# 주가예측 딥러닝 캡스톤

Quantile 예측 기반 주가 방향 예측 + 모의투자 파이프라인.
설계 원칙과 규칙은 [`CLAUDE.md`](CLAUDE.md), 데이터 소스 검증은 [`docs/KIWOOM_VERIFY.md`](docs/KIWOOM_VERIFY.md).

## 셋업

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env   # 키움 모의투자 APP_KEY / APP_SECRET 채우기
pytest -q              # 131 tests
```

## 파이프라인

```bash
python scripts/build_universe.py        # 유니버스 자동 선정 → configs/universe.yaml
python scripts/collect.py --dry-run     # 수집 계획 확인 (API 호출 없음)
python scripts/collect.py --tr chart info   # 일봉+종목정보만 (빠름, 약 15분)
python scripts/collect.py --tr flow     # 수급만 (느림, 종목당 30~50초)
python scripts/collect.py               # 전부
python scripts/peek.py                  # 뭐가 얼마나 쌓였는지 확인
python scripts/build_features.py        # → panel / macro / static parquet
python scripts/train.py --smoke         # Phase 1 배관 점검 (6종목 2epoch)
python scripts/train.py                 # Phase 1 학습
python scripts/sweep.py                 # 정규화 강도 비교 (한 세션에서 여러 설정)
python scripts/backtest.py              # 거래비용 반영 백테스트 (체크포인트 필요)
python scripts/verify_trading_trs.py    # 계좌/주문 TR 응답 스키마 검증 (읽기 전용)
python scripts/paper_trade.py           # 모의투자 — 계획만 출력 (주문 안 나감)
python scripts/paper_trade.py --execute # 모의투자 주문 전송
python scripts/dashboard.py             # 결과 대시보드 (http://127.0.0.1:8765)
```

### 하루 한 번 도는 자동매매

```bash
python scripts/collect.py --tr chart \
  && python scripts/build_features.py \
  && python scripts/paper_trade.py --execute
```

`--execute` 없이 돌리면 **계획만 출력하고 주문은 나가지 않는다**(기본값).
리밸런싱 주기(5일)가 아닌 날은 신규 진입 없이 손절/익절만 본다.

모든 수집은 **증분(idempotent)** — 같은 명령을 두 번 돌려도 중복 행이 생기지 않는다.

수집 결과는 `data/` 에 parquet 으로 쌓인다. 이진 파일이라 에디터로 열리지 않으니
`scripts/peek.py` 로 확인한다.

```
data/raw/{TR이름}/{종목코드}.parquet   ← API 응답 원본
data/processed/features.parquet        ← 지표·라벨까지 계산된 학습용 테이블
```

## 현재 상태

| 레이어 | 모듈 | 상태 |
|---|---|---|
| 유틸 | `parsing` `ratelimit` `config` `logging` `seed` | ✅ 완료 + 테스트 |
| 수집 | `kiwoom/client` `kiwoom/collect` `storage` | ✅ 완료 — 실호출 검증(005930, 2015~현재 2,857행) |
| 피처 | `features/technical` | ✅ 완료 + look-ahead 테스트 |
| 피처 | `features/build` (종목×매크로 결합, static covariate) | ✅ 완료 |
| 모델 | `revin` `patch_embed` `encoder` `cross_attention` `vsn` `quantile_head` `phase1` | ✅ 완료 + 테스트 (**0.33M** — 스윕으로 축소 확정) |
| 학습 | `split` `losses` | ✅ 완료 + 테스트 |
| 학습 | `dataset` `train` | ✅ 완료 — GPU 학습 실행, 기준선 대비 **+3.34%** |
| 평가 | `metrics` `backtest` | ✅ 완료 + 테스트 — 랭크 IC 진단 포함 |
| 매매 | `trading/signal` | ✅ 완료 + 테스트 17개 (횡단면 순위 모드) |
| 매매 | `trading/risk` | ✅ 완료 + 테스트 9개 |
| 매매 | `trading/broker` (키움 계좌·주문) | ✅ 완료 + 테스트 12개 |
| 매매 | `trading/paper_trader` | ✅ 완료 + 테스트 15개 |
| 추론 | `models/inference` (백테스트·모의투자 공용) | ✅ 완료 |
| 화면 | `webapp` 대시보드 | ✅ 완료 + 테스트 7개 |

### 2026-08-25 학습·백테스트 결과

**학습은 재현됐다.** 캐글 GPU 에서 기준선(무조건부 분위수) 대비 **+3.34%**,
best epoch 3 — 로컬 스윕(+3.35%)과 소수점 둘째 자리까지 일치.

**백테스트는 전략이 거래를 거의 안 했다.** 2.1년 체결 64건, 평균 노출 8.9%.
진 게 아니라 참여를 안 했다. 원인은 기권 로직이 아니라 **절대 임계값**이었다 —
모델 예측 중앙값(-0.15%)이 실제 평균(+0.80%)과 어긋나 매수 임계값(+0.40%)에
구조적으로 못 닿았다. 상세는 [`src/trading/CLAUDE.md`](src/trading/CLAUDE.md).

→ 매매 신호를 **횡단면 순위**로 전환했다. 전략이 살아났다:
체결 64 → **1,247**, 평균 노출 8.9% → **66.5%**, Sharpe 0.14 → **0.88**,
CAGR 0.03% → **12.28%**.

### 모델은 방향을 배웠다

```
랭크 IC  +0.0243   t = +4.00   515일   양수비율 55.7%
```

IC 0.024는 주식 횡단면 예측에서 정상 범위(0.02~0.05)이고 **t=4.0이면 통계적으로 명확하다.**
피처 중요도가 변동성 계열에 몰려 있는 건 사실이지만 방향 정보도 실제로 들어 있었다.

### 그런데도 매수후보유(CAGR 49.2%)에 진다 — 원인 셋

| # | 원인 | 근거 |
|---|---|---|
| 1 | 거래비용 | 연 회전율 46.9 × 편도 평균 15.5bp ≈ 연 7% |
| 2 | 스프레드 < 비용 | 십분위 스프레드 +0.19%(t=1.07) < 왕복비용 0.31% |
| 3 | 저변동성 편향 | 변동성 14.3% vs 시장 28.5% — **베타 0.5** |

**리스크는 절반을 졌는데 수익은 4분의 1만 가져왔다.**
기권 필터가 신뢰구간이 좁은 = 저변동성 종목만 남기는데, CAGR 49% 폭등장에서
저변동성은 구조적으로 뒤처진다.

→ **이력 버퍼 + 최소 거래폭**으로 회전율을 낮췄고, 비용·보유일수·차단사유를
실측으로 남기게 했다. `python scripts/backtest.py --compare` 가 규칙 5가지를
한 세션에서 비교한다.

**2026-08-25 실측 (phase1_0db568ae, test 구간)**

| | 버퍼 전 | 버퍼+밴드 |
|---|---|---|
| 연 회전율 | 46.9 | **32.5** |
| 평균 보유일수 | — | **10.0일** (리밸런싱 주기 5일의 2배) |
| 실지불 비용(연) | 추정 7% | **4.06%** (실측) |
| Sharpe | 0.88 | **1.10** |
| CAGR | 12.28% | **14.97%** |

매수후보유(Sharpe 1.55 / CAGR 49.2%)에는 여전히 진다 — 베타 0.5 라 폭등장에서
구조적으로 뒤처진다. 다만 **최대낙폭은 -13.5% vs -25.7%** 로 절반이다.

⚠️ 회전율을 낮춰도 **알파가 생기지는 않는다.** 이 작업의 산출물은 알파가 아니라
**귀속**이다 — 매수후보유와의 격차 중 얼마가 비용이고 얼마가 종목선택인지 가르는 것.

## 다음 할 일

1. ~~TR 4종 검증~~ ✅ 완료 — 상세는 `docs/KIWOOM_VERIFY.md`
2. ~~유니버스 8종목 + 지수 2종 + ETF 수집~~ ✅ 완료 (아래 "수집 현황")
3. **`features/build.py`** — 종목 피처에 매크로(KOSPI/KOSDAQ/ETF) 시퀀스와
   static covariate(업종·시총구간·요일)를 결합. 크로스어텐션 입력을 여기서 만든다.
   ⚠️ ETF(390390)는 2021-06-30 상장이라 앞 구간이 없다 → 마스킹 필요
4. ~~Phase 1 모델~~ ✅ 완료 — `--smoke` 로 학습이 끝까지 도는 것 확인
5. ~~GPU 학습 실행~~ ✅ 완료 — 기준선 대비 +3.34%, 스윕으로 0.33M 확정
6. ~~백테스트 구축~~ ✅ 완료 — 절대 임계값의 한계를 실측으로 확인
7. ~~순위 방식 검증~~ ✅ 완료 — 랭크 IC t=4.00, 전략이 실제로 거래한다
8. ~~기본 규칙 실측~~ ✅ 완료 — 로컬에서 `scripts/backtest.py` 실행,
   버퍼+밴드 결과가 위 표에 있다. `--compare` 5개 변형 비교는 아직 —
   `notebooks/kaggle_all_in_one.ipynb` 를 Import 하면 한 세션에서 끝난다
9. walk-forward 다구간 백테스트 — test 가 2024-07~2026-08 한 국면(역사적 폭등장)뿐이다.
   베타 0.5짜리 전략을 이 구간 하나로 판단할 수 없다
10. ~~`src/trading/paper_trader.py` 모의투자 실행~~ ✅ 완료 — 대시보드까지
11. **매매 TR 검증** ← 지금 할 일. 계좌/주문 TR 은 아직 `UNVERIFIED` 다.
    ```bash
    python scripts/verify_trading_trs.py          # 조회계만 (읽기 전용, 안전)
    python scripts/verify_trading_trs.py --order 005930   # 1주 매수까지 (실제 주문)
    ```
    ⚠️ **잔고 필드명이 틀리면 보유수량이 0 으로 읽히고 중복 매수가 나간다.**
    수집 TR 과 달리 조용히 틀리는 쪽이라 주문 전에 반드시 통과시킬 것.
12. 모의투자 첫 실행 — `python scripts/paper_trade.py` 로 계획을 눈으로 확인한 뒤
    `--execute`. 결과는 `python scripts/dashboard.py` 로 본다

## 클라우드 GPU 학습

맥북(MPS)은 epoch당 약 19분이라 50 epoch에 16시간이 걸린다. 학습만 외부 GPU로 옮긴다.
**수집은 로컬에서만 한다** — 키움 API는 등록된 IP에서만 호출되기 때문이다.

```bash
python scripts/package_data.py     # outputs/train_bundle.zip (35MB)
```

그 다음 노트북을 연다. **Kaggle 권장** — 주당 30시간, 세션 12시간이라 Colab 무료보다 길다.

노트북은 **`notebooks/kaggle_all_in_one.ipynb` 하나뿐이다** — 학습과 백테스트를
같은 세션에서 끝내므로 체크포인트를 내려받았다 다시 올릴 필요가 없다.
절차: [`docs/KAGGLE_SETUP.md`](docs/KAGGLE_SETUP.md)

코드는 GitHub에서 클론되고 이 zip만 올리면 된다.
**`.env`는 필요 없다** — 학습 과정에 API 호출이 전혀 없다.

학습 코드가 CUDA를 감지하면 혼합정밀(AMP)과 DataLoader 워커를 자동으로 켠다 —
같은 명령이 로컬/클라우드 양쪽에서 그대로 돈다.

### 수집 현황 (2026-08-24)

| 종류 | 대상 | 행 수 |
|---|---|---|
| 일봉 | 유니버스 **146종목** | 415,948행 (2015-01 ~ 2026-08) |
| 종목정보 | 유니버스 146종목 | 각 1행 (조회시점 스냅샷) |
| 수급 | 8종목 (나머지는 수집 대기) | 22,398행 |
| 지수 | KOSPI(001), KOSDAQ(101) | 각 2,857행 |
| ETF | KODEX 미국반도체MV(390390) | 1,260행 (2021-06 상장) |

**학습샘플 352,876개** (lookback 120일 기준)

| 구간 | 기간 | 샘플 |
|---|---|---|
| train | 2015-03 ~ 2022-12 | 260,339 |
| val | 2023-01 ~ 2023-12 | 17,584 |
| test | 2024-01 ~ 2026-08 | 74,953 |

유니버스 선정 기준: 2015-01-01 이전 상장 / 감사의견 정상 / 보통주 / 시가총액 상위
(ETF·리츠·스팩·인프라펀드 제외). 21개 섹터 분산.

## 다음 세션 시작하기

며칠 뒤 돌아왔을 때 이 순서대로 하면 된다.

```bash
cd ~/Desktop/Capstone_Stock_Price_Prediction
claude --resume          # 이전 대화 이어가기 (목록에서 선택)
```

새 대화로 시작한다면 아래 4개만 읽으면 맥락이 복구된다:

1. `CLAUDE.md` — 규칙·설계 원칙 (자동 로드됨)
2. `README.md` 의 **현재 상태** 표 + **다음 할 일**
3. `docs/KIWOOM_VERIFY.md` — 키움 TR 검증 현황
4. `.venv/bin/python -m pytest -q` — 통과하면 코드 건강함

**기록 원칙** (이걸 지켜야 위 4개가 믿을 만해진다)
- `CLAUDE.md` = 규칙만. 진행 일지를 여기 쓰지 않는다 (길어지면 정작 규칙이 묻힌다)
- 진행 상황 = 이 README 의 상태표. 뭔가 끝내면 여기 표를 고친다
- 검증·실험 결과 = `docs/` 와 `outputs/reports/`
- 작업 단위로 커밋. 커밋 메시지가 곧 작업 이력이다

## 안전장치

- `KIWOOM_ENV=live` 는 `src/utils/config.py` 에서 예외를 던진다. 실전투자 경로 없음.
  `PaperBroker` 가 생성 시점에 한 번 더 확인한다 (2중 방어).
- `.env` 는 gitignore. `.env.example` 에는 키 **이름만** 둔다.
- 매매 신호는 `src/trading/signal.py` 하나만 사용 — 백테스트/모의투자가 같은 코드를 공유한다.
  추론도 `src/models/inference.py` 하나만 쓴다.
- `scripts/paper_trade.py` 는 **기본이 dry-run**. `--execute` 를 붙여야 주문이 나간다.
- 대시보드는 **읽기 전용**이다. 화면에서 주문을 낼 수 없고, 기본 바인딩은 127.0.0.1 이다.
- 패널이 영업일 3일 이상 낡으면 실주문을 막는다(`--ignore-stale` 로만 강행).
