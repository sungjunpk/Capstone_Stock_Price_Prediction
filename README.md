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
python scripts/collect.py --dry-run     # 수집 계획 확인 (API 호출 없음)
python scripts/collect.py               # 일봉/수급/종목정보/지수 증분 수집
python scripts/peek.py                  # 뭐가 얼마나 쌓였는지 확인
python scripts/build_features.py        # 지표 + 라벨 → data/processed/features.parquet
python scripts/train.py                 # (미구현) Phase 1 모델 학습
python scripts/backtest.py              # (미구현) walk-forward 백테스트
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
| 피처 | `features/build` (종목×매크로 결합, static covariate) | ⬜ |
| 모델 | `revin` `patch_embed` `encoder` `cross_attention` `vsn` `quantile_head` `phase1` | ⬜ |
| 학습 | `split` `losses` | ✅ 완료 + 테스트 |
| 학습 | `dataset` `train` | ⬜ |
| 평가 | `metrics` `backtest` | ⬜ |
| 매매 | `trading/signal` | ✅ 완료 + 테스트 |
| 매매 | `trading/risk` `trading/paper_trader` | ⬜ |

## 다음 할 일

1. ~~일봉 TR 검증~~ ✅ 완료 (mock 도 1985년부터 전체 이력 제공 — 데이터 부족 이슈 없음)
2. **나머지 TR 검증** — `stock_info`(ka10001) / `investor_flow`(ka10059) / `index_daily`(ka20006).
   절차는 `docs/KIWOOM_VERIFY.md`.
3. 유니버스 8종목 전체 수집 → 매크로 지수 + 국내상장 해외ETF 결합 (`features/build.py`)
4. Phase 1 모델 설계 → **Plan Mode 로 승인 후** 구현

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
