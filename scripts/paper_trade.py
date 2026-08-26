#!/usr/bin/env python
"""모의투자 실행 — 학습된 모델로 오늘의 주문을 낸다.

하루 한 번 장중에 돌린다. 앞 두 단계는 데이터를 오늘까지 채우는 일이다:

    python scripts/collect.py --tr chart     # 어제까지의 일봉 증분 수집
    python scripts/build_features.py         # panel/macro/static 갱신
    python scripts/paper_trade.py            # 계획만 본다 (기본, 주문 안 나감)
    python scripts/paper_trade.py --execute  # 실제 모의투자 주문 전송

⚠️ **기본이 dry-run 이다.** `--execute` 를 붙여야 주문이 나간다.
   실전투자 경로는 존재하지 않는다 — config.py 와 PaperBroker 가 이중으로 막는다.

리밸런싱 주기(기본 5일)가 아니면 신규 진입을 하지 않고 손절/익절만 본다.
주기를 무시하고 강제로 리밸런싱하려면 `--force-rebalance`.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.models.inference import (  # noqa: E402
    latest_prices,
    load_features,
    load_model,
    predict_recent,
)
from src.trading.broker import BUY, PaperBroker  # noqa: E402
from src.trading.paper_trader import (  # noqa: E402
    TraderState,
    build_plan,
    execute_plan,
    is_rebalance_day,
    save_run,
)
from src.utils.config import PROJECT_ROOT, load_config  # noqa: E402
from src.utils.logging import get_logger, setup_logging  # noqa: E402

log = get_logger("paper_trade")
CKPT_DIR = PROJECT_ROOT / "outputs" / "checkpoints"

# 데이터가 이보다 오래되면 오늘 판단의 근거가 낡았다는 뜻이다.
_STALE_BUSINESS_DAYS = 3


def find_checkpoint(explicit: str | None) -> Path:
    """백테스트와 같은 규칙으로 고른다 — smoke 는 제외, 가장 최근 것."""
    if explicit:
        p = Path(explicit)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if not p.exists():
            raise SystemExit(f"체크포인트가 없다: {p}")
        return p

    cands = [p for p in CKPT_DIR.glob("phase1_*.pt") if "smoke" not in p.name]
    if not cands:
        raise SystemExit(
            "체크포인트가 없다. scripts/train.py 로 학습하거나 캐글에서 받은 .pt 를\n"
            "  outputs/checkpoints/ 에 두고 --checkpoint 로 지정할 것."
        )
    latest = max(cands, key=lambda p: p.stat().st_mtime)
    log.info("체크포인트 자동 선택: %s", latest.name)
    return latest


def _check_freshness(last_data_date: date, today: date) -> list[str]:
    """패널이 오늘 판단에 쓸 만큼 최신인가."""
    gap = int(np.busday_count(np.datetime64(last_data_date, "D"), np.datetime64(today, "D")))
    if gap <= _STALE_BUSINESS_DAYS:
        return []
    msg = (
        f"패널 마지막 거래일이 {last_data_date} 로 영업일 {gap}일 뒤처져 있다. "
        "오래된 데이터로 주문을 내지 말 것 — "
        "`python scripts/collect.py --tr chart && python scripts/build_features.py` 먼저."
    )
    log.warning(msg)
    return [msg]


def _print_plan(plan, account, dry_run: bool) -> None:
    print("\n" + "=" * 70)
    print(f"모의투자 계획 — 판단 기준일 {plan.decision_date}"
          + ("  [리밸런싱]" if plan.rebalancing else "  [손절/익절 점검만]"))
    print("=" * 70)
    print(f"  총자산      {plan.equity:>15,.0f}원")
    print(f"  주문가능    {plan.cash:>15,.0f}원")
    print(f"  보유종목    {len(account.holdings):>15}종목")

    s = plan.stats
    print("\n  판단 — 기권이 방향보다 먼저 온다")
    print(f"    후보 종목     {s['n_candidates']}개")
    print(f"    기권          {s['abstain']}개 ({s['abstain_rate']:.1%}) "
          f"— 임계 폭 {plan.abstain_threshold:.4f}")
    print(f"    매수 신호     {s['buy']}개   미선택(hold) {s['hold']}개")
    print(f"    목표 노출도   {s['target_gross']:.1%}")

    if plan.blocked_by_reason:
        print("\n  리스크 차단")
        for reason, cnt in sorted(plan.blocked_by_reason.items()):
            print(f"    {reason:<10} {cnt}건")
    if plan.forced_exits:
        print(f"    강제청산: {', '.join(plan.forced_exits)}")

    if not plan.orders:
        print("\n  주문 없음 — 오늘은 거래하지 않는다")
    else:
        print(f"\n  주문 {len(plan.orders)}건" + ("  (dry-run — 전송하지 않음)" if dry_run else ""))
        print(f"    {'종목':<8}{'구분':<6}{'수량':>8}{'단가':>12}{'금액':>14}"
              f"{'비중':>16}")
        for o in plan.orders:
            side = "매수" if o.side == BUY else "매도"
            print(f"    {o.code:<8}{side:<6}{o.quantity:>8,}{o.price:>12,.0f}"
                  f"{o.amount:>14,.0f}"
                  f"{o.weight_from:>8.1%} → {o.weight_to:>5.1%}")

    for note in plan.notes:
        print(f"\n  ⚠️ {note}")


def _print_holdings(account) -> None:
    if not account.holdings:
        return
    print("\n  현재 보유")
    print(f"    {'종목':<8}{'수량':>8}{'매입가':>12}{'현재가':>12}{'평가금액':>14}{'손익':>10}")
    for h in sorted(account.holdings.values(), key=lambda x: -x.eval_amount):
        print(f"    {h.code:<8}{h.quantity:>8,}{h.avg_price:>12,.0f}"
              f"{h.current_price:>12,.0f}{h.eval_amount:>14,.0f}{h.pnl_rate:>9.2f}%")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint")
    ap.add_argument("--execute", action="store_true",
                    help="실제 모의투자 주문 전송 (기본은 계획만 출력)")
    ap.add_argument("--force-rebalance", action="store_true",
                    help="리밸런싱 주기를 무시하고 이번에 리밸런싱한다")
    ap.add_argument("--recent-days", type=int, default=90,
                    help="기권 임계값을 잡을 예측 폭 분포의 관측 구간(일)")
    ap.add_argument("--ignore-stale", action="store_true",
                    help="데이터가 오래돼도 진행한다 (권장하지 않음)")
    args = ap.parse_args()

    setup_logging(run_name="paper_trade")
    cfg = load_config().raw
    dry_run = not args.execute
    today = date.today()

    # --- 1) 모델 + 피처
    loaded = load_model(find_checkpoint(args.checkpoint))
    bundle = load_features(cfg, loaded)
    notes = _check_freshness(bundle.last_date, today)
    if notes and not args.ignore_stale and not dry_run:
        raise SystemExit(
            "데이터가 오래돼 실주문을 막았다. 수집 후 다시 실행하거나 --ignore-stale 로 강행할 것."
        )

    # --- 2) 예측. 최신 하루가 아니라 최근 구간을 낸다(기권 임계값이 분포 기반이라서)
    recent = predict_recent(loaded, bundle, cfg, days=args.recent_days)

    # --- 3) 계좌
    with PaperBroker() as broker:
        account = broker.snapshot()
        # 미체결 조회는 계획 수립 **전에** 한다. 실패하면 예외로 멈춘다 —
        # '미체결 없음'으로 오해하고 주문을 얹는 것보다 안 내는 쪽이 낫다.
        unfilled = broker.fetch_unfilled()
        state = TraderState.load()
        state.sync_entries(set(account.holdings), today)

        rebalancing = args.force_rebalance or is_rebalance_day(
            state, today, int(cfg["backtest"].get("rebalance_days", 5))
        )

        # --- 4) 1차 계획: 종가 기준으로 '무엇을 건드릴지'만 정한다.
        #     기권·순위·사이징은 가격과 무관하므로 이 단계에서 확정된다.
        #     현재가는 그 다음, 실제로 건드릴 종목만 조회한다(호출 수 절약).
        closes = latest_prices(bundle)
        draft = build_plan(recent, account, closes, cfg, state=state,
                           today=today, rebalancing=rebalancing, unfilled=unfilled)

        touch = sorted({o.code for o in draft.orders} | set(account.holdings))
        quotes = broker.fetch_prices(touch) if touch else {}
        log.info("현재가 조회 %d종목 (계획 대상 + 보유분)", len(quotes))

        # --- 5) 최종 계획: 손절 판정과 주문 수량이 현재가 기준이 된다
        prices = {**closes, **quotes}
        plan = build_plan(recent, account, prices, cfg, state=state,
                          today=today, rebalancing=rebalancing, unfilled=unfilled)
        plan.notes.extend(notes)

        _print_holdings(account)
        _print_plan(plan, account, dry_run)

        # --- 6) 전송
        results = execute_plan(broker, plan, dry_run=dry_run)

        sent = [r for r in results if r.ok and not r.dry_run]
        failed = [r for r in results if not r.ok]
        if not dry_run:
            print(f"\n  전송 {len(sent)}건 / 실패 {len(failed)}건")
            for r in failed:
                print(f"    실패 {r.code} {r.side}: {r.error}")

        out = save_run(plan, results, dry_run=dry_run)

        # 상태는 **실제로 주문을 낸 경우에만** 갱신한다.
        # dry-run 이 리밸런싱 날짜를 소모하면 다음 실행이 조용히 건너뛴다.
        if not dry_run:
            if rebalancing:
                state.last_rebalance = today.isoformat()
            for r in sent:
                if r.side == BUY:
                    state.entry_dates.setdefault(r.code, today.isoformat())
            state.runs += 1
            state.save()

    print(f"\n저장: {out.relative_to(PROJECT_ROOT)}")
    if dry_run:
        print("주문은 나가지 않았다. 실제로 내려면 --execute 를 붙일 것.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
