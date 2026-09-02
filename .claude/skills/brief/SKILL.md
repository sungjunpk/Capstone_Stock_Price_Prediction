---
name: brief
description: 세션 시작 브리핑 — 이 프로젝트가 지금 어디까지 왔고 오늘 무엇이 걸려 있는지. 캡스톤 작업을 새로 시작할 때, "지금 상태 알려줘" / "어디까지 했지" / "오늘 뭐 해야 해" 같은 요청에 사용한다.
---

# 세션 시작 브리핑

매번 같은 파일을 같은 순서로 읽는 대신 이 절차를 돈다. **읽기 전용이다.**

## 1. 상태 수집 (한 번에 실행)

```bash
git log --oneline -5
sed -n '/## 현재 상태/,/^### /p' README.md | head -40
ls -t outputs/reports/ | head -5
tail -2 outputs/paper_trading/equity.jsonl
python3.11 -c "
import json,pandas as pd
p=json.load(open('outputs/paper_trading/performance.json'))
print('실거래', p['start_date'], '~', p['end_date'], '| n_days', p['n_days'],
      '| 누적', f\"{p['total_return']:+.2%}\", '| reliable', p['reliable'])
d=pd.read_parquet('data/processed/panel.parquet', columns=['date'])
print('패널 최종일', d['date'].max())
"
ls -t outputs/logs/daily/ | head -3
```

## 2. 보고할 것 — 이 다섯 줄이면 충분하다

| 항목 | 어디서 |
|---|---|
| 마지막 커밋과 그 작업 | `git log` |
| 패널 최종일이 오늘/전 거래일인가 | 위 스크립트. **1세션 넘게 낡으면 리밸런싱이 막힌다** |
| 실거래 경과일·누적수익률 | `performance.json`. `reliable: false` 면 Sharpe·MDD 를 인용하지 않는다 |
| **다음 리밸런싱일** | 마지막 리밸런싱 + `backtest.rebalance_days`(현재 10거래일) |
| 자동화가 어제 정상이었나 | `outputs/logs/daily/` 최신 파일 끝에 `완료` 가 있는지 |

## 3. 브리핑에 넣지 말 것

- 20일 미만 표본의 Sharpe·MDD (`reliable: false` 일 때) — 착시다
- "최근 N개월" 성과를 창 명시 없이 인용 — 시작일 닷새 차이로 부호가 뒤집힌 전례가 있다
- 백테스트 CAGR 을 모델 성과로 서술 — 순열검정이 p=0.25 로 기각했다
