# 주가예측 딥러닝 캡스톤

Quantile 예측 기반 주가 방향 예측 + 모의투자 파이프라인.
설계 원칙과 규칙은 [`CLAUDE.md`](CLAUDE.md), 데이터 소스 검증은 [`docs/KIWOOM_VERIFY.md`](docs/KIWOOM_VERIFY.md).

## 셋업

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env   # 키움 모의투자 APP_KEY / APP_SECRET 채우기
pytest -q              # 21 tests
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
python scripts/paper_trade.py           # (미구현) 모의투자 실행
```

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
| 매매 | `trading/paper_trader` | ⬜ |

### 2026-08-25 학습·백테스트 결과

**학습은 재현됐다.** 캐글 GPU 에서 기준선(무조건부 분위수) 대비 **+3.34%**,
best epoch 3 — 로컬 스윕(+3.35%)과 소수점 둘째 자리까지 일치.

**백테스트는 전략이 거래를 거의 안 했다.** 2.1년 체결 64건, 평균 노출 8.9%.
진 게 아니라 참여를 안 했다. 원인은 기권 로직이 아니라 **절대 임계값**이었다 —
모델 예측 중앙값(-0.15%)이 실제 평균(+0.80%)과 어긋나 매수 임계값(+0.40%)에
구조적으로 못 닿았다. 상세는 [`src/trading/CLAUDE.md`](src/trading/CLAUDE.md).

→ 매매 신호를 **횡단면 순위**로 전환했다. 공통 편차가 상쇄되어 이 문제를 안 받는다.
검증 결과 대기 중.

⚠️ **주의해서 읽어야 할 지점**: 피처 중요도 상위 5개가 전부 변동성 계열이고,
q50(방향)의 분산이 구간폭(변동성)의 1/25 이다. pinball 개선의 대부분이 변동성에서
나왔을 수 있다. **랭크 IC** 가 그 둘을 갈라놓는다 —
[`src/evaluation/CLAUDE.md`](src/evaluation/CLAUDE.md).

## 다음 할 일

1. ~~TR 4종 검증~~ ✅ 완료 — 상세는 `docs/KIWOOM_VERIFY.md`
2. ~~유니버스 8종목 + 지수 2종 + ETF 수집~~ ✅ 완료 (아래 "수집 현황")
3. **`features/build.py`** — 종목 피처에 매크로(KOSPI/KOSDAQ/ETF) 시퀀스와
   static covariate(업종·시총구간·요일)를 결합. 크로스어텐션 입력을 여기서 만든다.
   ⚠️ ETF(390390)는 2021-06-30 상장이라 앞 구간이 없다 → 마스킹 필요
4. ~~Phase 1 모델~~ ✅ 완료 — `--smoke` 로 학습이 끝까지 도는 것 확인
5. ~~GPU 학습 실행~~ ✅ 완료 — 기준선 대비 +3.34%, 스윕으로 0.33M 확정
6. ~~백테스트 구축~~ ✅ 완료 — 절대 임계값의 한계를 실측으로 확인
7. **캐글에서 순위 방식 검증** ← 지금 할 일.
   `notebooks/kaggle_all_in_one.ipynb` 를 위에서부터 실행하면 매매 규칙 3가지가
   나란히 비교된다. **랭크 IC 의 t 값을 성과보다 먼저 볼 것**
8. walk-forward 다구간 백테스트 — test 가 2024-07~2026-08 한 국면뿐이다
9. `src/trading/paper_trader.py` 모의투자 실행

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
- `.env` 는 gitignore. `.env.example` 에는 키 **이름만** 둔다.
- 매매 신호는 `src/trading/signal.py` 하나만 사용 — 백테스트/모의투자가 같은 코드를 공유한다.
