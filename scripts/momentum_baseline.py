#!/usr/bin/env python
"""모멘텀 + 시총가중 베이스라인 — 모델이 **이걸 이겨야** 의미가 있다.

왜 필요한가:
    목표가 "코스피200 초과수익"으로 바뀌자(2026-09-08), 실측에서 모델 없는 규칙 하나가
    지수를 크게 이겼다 — 12-1 모멘텀 상위 N 을 시총가중으로 담는 것이다.
    이걸 대조군으로 등록해두지 않으면 "모델이 지수를 이겼다"는 결과가 나와도
    **딥러닝이 기여했는지 알 수 없다.** GBDT 베이스라인(`gbdt_baseline.py`)이
    예측력에 대해 하는 일을, 이 스크립트는 매매 성과에 대해 한다.

    ⚠️ 이 규칙은 test 구간을 보고 고른 것이라 **선택편향이 있다.** 그래서 성과를
    자랑하는 용도가 아니라 **넘어야 할 문턱**으로만 쓴다. 달력 민감도까지 같이 낸다.

무엇을 고정하는가 (공정성의 전부):
    같은 패널 · 같은 유니버스 · 같은 리밸런싱 주기 · 같은 거래비용 ·
    같은 성과 함수(`evaluation.metrics.summarize`) · 같은 지수(`evaluation.benchmark`).

사용:
    python scripts/momentum_baseline.py
    python scripts/momentum_baseline.py --top 30 --lookback 252
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.data.storage import PROCESSED_DIR  # noqa: E402
from src.evaluation.benchmark import excess_return, index_returns  # noqa: E402
from src.evaluation.metrics import summarize  # noqa: E402
from src.trading.signal import one_way_cost  # noqa: E402
from src.utils.config import PROJECT_ROOT, load_config  # noqa: E402
from src.utils.logging import get_logger, setup_logging  # noqa: E402

log = get_logger("momentum_baseline")
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"


def _panel(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(종가 피벗, 시총 피벗). 둘 다 (date x code)."""
    sfx = cfg["data"].get("processed_suffix", "")
    df = pd.read_parquet(PROCESSED_DIR / f"panel{sfx}.parquet",
                         columns=["date", "code", "close", "mcap"])
    df["date"] = pd.to_datetime(df["date"])
    px = df.pivot_table(index="date", columns="code", values="close").sort_index().ffill()
    cap = df.pivot_table(index="date", columns="code", values="mcap").sort_index().ffill()
    return px, cap


def run(px: pd.DataFrame, cap: pd.DataFrame, dates: list, cfg: dict,
        *, top: int, lookback: int) -> pd.Series:
    """12-1 모멘텀 상위 `top` 종목을 시총가중. 리밸런싱 사이에는 그대로 둔다.

    비용은 백테스트와 같은 함수(`one_way_cost`)로 계산한다 — 여기서만 다른 비용을
    쓰면 "베이스라인이 더 좋다"가 비용 차이일 수 있다.
    """
    costs = cfg["trading"].get("costs", {})
    cost_rate = (one_way_cost(costs, selling=False) + one_way_cost(costs, selling=True)) / 2
    hold = int(cfg["backtest"]["rebalance_days"])

    rets = px.pct_change(fill_method=None).fillna(0.0)
    w = pd.Series(dtype=float)
    out = []
    for i, d in enumerate(dates):
        r = float((rets.loc[d].reindex(w.index) * w).sum()) if len(w) else 0.0
        paid = 0.0
        if i % hold == 0:
            hist = px.loc[:d]
            if len(hist) > lookback:
                mom = (hist.iloc[-1] / hist.iloc[-lookback] - 1).dropna()
                codes = list(mom.sort_values(ascending=False).head(top).index)
                c = cap.loc[d, codes].dropna()
                new = (c / c.sum()) if len(c) and c.sum() > 0 else pd.Series(
                    1.0 / len(codes), index=codes)
                keys = w.index.union(new.index)
                paid = float(
                    (new.reindex(keys).fillna(0.0) - w.reindex(keys).fillna(0.0)).abs().sum()
                ) * cost_rate
                w = new
        elif len(w):                    # 리밸런싱 사이에는 가격을 따라 비중이 흐른다
            w = w * (1 + rets.loc[d].reindex(w.index))
            w = w / w.sum()
        out.append(r - paid)
    return pd.Series(out, index=pd.Index(dates))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--lookback", type=int, default=252, help="모멘텀 관측일 (12개월)")
    ap.add_argument("--months", type=int, nargs="+", default=[24, 12, 6])
    args = ap.parse_args()

    setup_logging(run_name="momentum_baseline")
    cfg = load_config().raw
    px, cap = _panel(cfg)
    end = px.index.max()

    windows = {}
    for m in args.months:
        start = end - pd.DateOffset(months=m)
        dates = [d for d in px.index if start <= d <= end]
        r = run(px, cap, dates, cfg, top=args.top, lookback=args.lookback)
        metrics = summarize(r)

        idx = index_returns("201", pd.Index([d.date() for d in dates]))
        bench = summarize(idx) if idx is not None else None
        ex = (excess_return(metrics["total_return"], bench["total_return"])
              if bench else None)

        # 달력 민감도 — 시작일만 밀어 재실행. 한 칸짜리 성과를 인용하지 않기 위해서다.
        shifts = []
        hold = int(cfg["backtest"]["rebalance_days"])
        for k in range(hold):
            rk = run(px, cap, dates[k:], cfg, top=args.top, lookback=args.lookback)
            mk = summarize(rk)
            shifts.append({
                "start": str(dates[k].date()),
                "total_return": round(mk["total_return"], 5),
                "sharpe": round(mk["sharpe"], 4),
                "beats_index": bool(bench and mk["total_return"] > bench["total_return"]),
            })
        wins = sum(s["beats_index"] for s in shifts)

        windows[f"{m}개월"] = {
            "start": str(dates[0].date()), "end": str(dates[-1].date()),
            "n_days": len(dates), "strategy": metrics, "index": bench,
            "excess_vs_index": None if ex is None else round(ex, 5),
            "calendar_shifts": shifts, "calendar_wins": wins,
        }
        log.info(
            f"{m:2d}개월  모멘텀 {metrics['total_return']:+8.1%}  "
            f"Sharpe {metrics['sharpe']:5.2f}  MDD {metrics['max_drawdown']:7.1%}  |  "
            f"지수 {bench['total_return'] if bench else float('nan'):+8.1%}  "
            f"초과 {ex if ex is not None else float('nan'):+7.1%}  |  "
            f"달력 {wins}/{hold} 승"
        )

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "rule": {"top_n": args.top, "lookback": args.lookback,
                 "weighting": "market_cap",
                 "rebalance_days": int(cfg["backtest"]["rebalance_days"])},
        "note": "모델이 넘어야 할 문턱. test 구간을 보고 고른 규칙이라 선택편향이 있다.",
        "windows": windows,
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / f"momentum_baseline_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str),
                   encoding="utf-8")
    log.info("저장 %s", out.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
