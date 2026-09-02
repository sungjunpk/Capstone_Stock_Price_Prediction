---
name: lookahead
description: 새 피처·타깃·평가지표를 추가하거나 수정했을 때 look-ahead(미래 정보 참조)를 감사한다. 피처 추가, 타깃 변경, 정규화·분할 코드 수정, 백테스트 로직 변경 시 사용한다. 조용히 틀리는 실패라 반드시 돈다.
---

# look-ahead 감사

**이 프로젝트에서 가장 위험한 버그 종류다.** 에러가 나지 않고, 오히려 **지표가 좋아진다.**
그래서 눈으로 보는 대신 절차로 잡는다 (CLAUDE.md 절대 규칙 5·6).

## 1. 무엇이 바뀌었는지 먼저 좁힌다

```bash
git diff --stat
git diff -- src/features/ src/training/split.py src/evaluation/ src/trading/
```

## 2. 다섯 자리를 순서대로 본다

| # | 확인 | 어긋나면 나타나는 증상 |
|---|---|---|
| 1 | **rolling 창이 t 를 포함하되 t+1 이후를 안 보는가.** `shift(-n)`, `center=True`, `bfill`, 역순 정렬 뒤 rolling 이 전형적인 범인 | 랭크 IC 가 갑자기 0.05 이상으로 뛴다 |
| 2 | **타깃이 피처에 새지 않는가.** t+1~t+h 수익률이 입력 채널에 흘러들어갔는가 | IC 가 0.1 을 넘으면 거의 확실히 누수 |
| 3 | **정규화 통계가 train 에서만 나왔는가.** `fit_normalizer` 는 train 구간만 받아야 한다 (`inference.load_features`) | val/test 성능이 train 보다 좋다 |
| 4 | **구간 경계에 embargo 가 있는가.** `split.embargo_days`(현재 5) 가 지평만큼 비워야 한다 | 경계 근처 예측만 유난히 정확 |
| 5 | **체결이 t 가 아니라 t+lag 인가.** `backtest.execution_lag_days`(기본 1). t 종가 체결은 방금 본 가격에 거래하는 것 | Sharpe 가 비현실적으로 높다 |

## 3. 테스트를 먼저 쓴다

기존 look-ahead 테스트를 본따 **새 피처에 대한 것을 추가한다.**

```bash
grep -rn "look\|shift\|lookahead" tests/test_technical.py | head
```

패턴: *뒤쪽 데이터를 잘라내고 계산한 값이, 전체로 계산한 값의 앞부분과 같아야 한다.*
같지 않으면 그 피처는 미래를 본다.

## 4. 숫자로 교차 확인

```bash
python scripts/feature_diagnostics.py     # 단변량 IC — 새 피처만 유난히 크면 의심
python scripts/backtest.py --split val    # val 에서도 같이 좋아지는가
```

⚠️ **test 에서만 좋아지고 val 에서 안 좋아지면 누수가 아니라 과적합이다.**
반대로 **양쪽 다 갑자기 좋아지면 누수를 먼저 의심한다.** 우리 기준선은 랭크 IC 0.024다 —
새 피처 하나로 그게 배로 뛰면 축하할 일이 아니라 찾아야 할 일이다.

## 5. 통과 기준

- [ ] 위 다섯 자리 전부 확인
- [ ] 새 피처에 대한 look-ahead 테스트 추가, `pytest -q` 통과
- [ ] val/test 지표 변화가 설명 가능한 크기
