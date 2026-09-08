#!/usr/bin/env python
"""리밸런싱 달력 민감도 — 예측은 한 번, 시작일만 하루씩 민다.

같은 모델·같은 규칙·같은 종료일인데 **첫 리밸런싱이 붙는 날**만 다르다.
방향 알파가 있으면 결과가 시작일에 크게 흔들리지 않아야 한다. 흔들린다면
백테스트 한 숫자를 인용할 근거가 없다는 뜻이다.

사용:
    python scripts/calendar_sensitivity.py
    python scripts/calendar_sensitivity.py --checkpoint outputs/checkpoints/phase1_ab4910e5.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from scripts.backtest import find_checkpoint  # noqa: E402
from src.evaluation.backtest import buy_and_hold, run_backtest  # noqa: E402
from src.evaluation.metrics import summarize  # noqa: E402
from src.models.inference import load_features, load_model, predict_split  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.logging import setup_logging  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    args = ap.parse_args()

    setup_logging(run_name="calendar_sensitivity")
    cfg = load_config().raw

    loaded = load_model(find_checkpoint(args.checkpoint, cfg))
    preds, prices = predict_split(loaded, load_features(cfg, loaded), cfg, args.split)

    starts = sorted(preds["date"].unique())
    end = max(prices["date"])
    # 주기가 10거래일이면 10칸이 달력 한 바퀴다 — 그 이상은 같은 위상이 반복된다.
    n_shift = int(cfg["backtest"]["rebalance_days"])

    print(f"\n{args.split} 예측 {starts[0]} ~ {starts[-1]} | 가격 끝 {end}")
    print(f"{'시작일':<14}{'일수':>6}{'누적':>10}{'CAGR':>9}{'Sharpe':>9}{'MDD':>9}{'회전율':>8}")

    rows = []
    for start in starts[:n_shift]:
        pad = (pd.Timestamp(start) - pd.DateOffset(days=20)).date()
        res = run_backtest(preds[preds["date"] >= start],
                           prices[(prices["date"] >= pad) & (prices["date"] <= end)], cfg)
        m = res.metrics
        rows.append(m)
        print(f"{str(start):<14}{m['n_days']:>6}{m['total_return']:>10.2%}"
              f"{m['cagr']:>9.2%}{m['sharpe']:>9.3f}{m['max_drawdown']:>9.2%}"
              f"{res.signal_stats['annual_turnover']:>8.1f}")

    s = pd.DataFrame(rows)
    print(f"\n  누적   최소 {s['total_return'].min():.2%} / 최대 {s['total_return'].max():.2%}"
          f" / 중앙값 {s['total_return'].median():.2%}")
    print(f"  Sharpe 최소 {s['sharpe'].min():.3f} / 최대 {s['sharpe'].max():.3f}"
          f" / 중앙값 {s['sharpe'].median():.3f}")

    bh = summarize(buy_and_hold(prices[prices["date"] >= starts[0]]))
    print(f"\n  참고: 유니버스 동일가중 매수후보유 누적 {bh['total_return']:.2%}"
          f" Sharpe {bh['sharpe']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
