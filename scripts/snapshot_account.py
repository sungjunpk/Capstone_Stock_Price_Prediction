#!/usr/bin/env python
"""계좌 스냅샷 기록 — 하루 한 번, 장 마감 후.

    python scripts/snapshot_account.py            # 오늘치 기록
    python scripts/snapshot_account.py --date 2026-08-26

**읽기 전용이다. 주문이 나가는 경로가 없다.**
거래가 없는 날에도 총자산을 찍어야 누적 수익률 곡선에 구멍이 안 생긴다.

같은 날 두 번 돌려도 줄이 하나다(날짜 키 upsert).
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.trading.broker import PaperBroker  # noqa: E402
from src.trading.record import (  # noqa: E402
    EQUITY_PATH,
    FILLS_PATH,
    HOLDINGS_PATH,
    compute_performance,
    record_equity,
    record_fills,
    record_holdings,
    save_baseline,
    save_performance,
)
from src.utils.config import PROJECT_ROOT  # noqa: E402
from src.utils.logging import get_logger, setup_logging  # noqa: E402

log = get_logger("snapshot")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", help="기록할 날짜 (YYYY-MM-DD, 기본 오늘). "
                                   "⚠️ 계좌 조회는 '지금' 상태다 — 과거 날짜를 주면 "
                                   "오늘 상태가 그 날짜로 기록된다. 재기록용으로만 쓸 것")
    ap.add_argument("--seed-baseline", type=float, metavar="원",
                    help="계좌 개시 잔고를 한 번 심는다. 첫 진입 수수료를 수익률에 "
                         "포함시키려면 필요하다. 이미 있으면 덮어쓰지 않는다")
    args = ap.parse_args()

    setup_logging(run_name="snapshot_account")
    on = pd.Timestamp(args.date).date() if args.date else date.today()

    if args.seed_baseline:
        # 심기만 하고 끝낸다. 계속 진행하면 **오늘 계좌 상태가 --date 날짜의
        # 스냅샷으로 기록된다** — 과거 계좌 상태는 알 수 없으므로 만들면 안 된다.
        save_baseline(args.seed_baseline, on)
        print(f"\n  기준선 {on} = {args.seed_baseline:,.0f}원")
        print("  스냅샷은 별도로 실행할 것: python scripts/snapshot_account.py")
        return 0

    with PaperBroker() as broker:
        account = broker.snapshot()
        diary = broker.fetch_trade_diary(on.strftime("%Y%m%d"))

    n_eq = record_equity(account, on)
    n_hold = record_holdings(account, on)
    n_fill = record_fills(diary, on)
    perf = compute_performance()
    save_performance(perf)

    print(f"\n  {on} 기록")
    print(f"    총자산      {account.equity:>15,.0f}원")
    print(f"    보유        {len(account.holdings):>15}종목")
    print(f"    체결 종목   {sum(1 for d in diary if d['buy_qty'] or d['sell_qty']):>15}개")
    print(f"\n  누적 ({perf['n_days']}일, 시작 {perf['start_equity']:,.0f}원)")
    if not perf["baseline_seeded"]:
        print("    ⚠️ 기준선 없음 — 첫날 진입 수수료가 수익률에서 빠져 있다 "
              "(--seed-baseline 으로 한 번 심을 것)")
    print(f"    누적수익률  {perf['total_return']:>15.2%}")
    print(f"    실현손익    {perf['realized_pnl']:>15,.0f}원")
    print(f"    수수료·세금 {perf['fee_tax']:>15,.0f}원")
    if not perf["reliable"]:
        print(f"\n  ⚠️ 관측 {perf['n_days']}일 — {perf['min_days_for_metrics']}일 미만이라 "
              "Sharpe·MDD 는 아직 해석하지 않는다")
    elif "strategy" in perf:
        s = perf["strategy"]
        print(f"    Sharpe      {s['sharpe']:>15.2f}")
        print(f"    최대낙폭    {s['max_drawdown']:>15.2%}")

    for p in (EQUITY_PATH, HOLDINGS_PATH, FILLS_PATH):
        print(f"\n저장: {p.relative_to(PROJECT_ROOT)}", end="")
    print(f"  ({n_eq}일 / 보유 {n_hold}행 / 체결 {n_fill}행)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
