#!/usr/bin/env python
"""모의투자 실행 — 학습된 모델로 오늘의 주문을 낸다.

하루 한 번 장중에 돌린다. 앞 두 단계는 데이터를 오늘까지 채우는 일이다:

    python scripts/collect.py --tr chart     # 어제까지의 일봉 증분 수집
    python scripts/build_features.py         # panel/macro/static 갱신
    python scripts/paper_trade.py            # 계획만 본다 (기본, 주문 안 나감)
    python scripts/paper_trade.py --execute  # 실제 모의투자 주문 전송

⚠️ **기본이 dry-run 이다.** `--execute` 를 붙여야 주문이 나간다.
   실전투자 경로는 존재하지 않는다 — config.py 와 PaperBroker 가 이중으로 막는다.

리밸런싱 주기(`backtest.rebalance_days`)가 아니면 신규 진입을 하지 않고 손절/익절만 본다.
주기를 무시하고 강제로 리밸런싱하려면 `--force-rebalance`.

**패널이 전 거래일보다 낡았으면 리밸런싱을 건너뛴다**(손절/익절은 그대로).
낡은 예측으로 새 베팅을 거는 것만 막는다 — 강행하려면 `--ignore-stale`.
"""

from __future__ import annotations

import argparse
import re
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
from src.trading.grid import build_ladder, should_relay  # noqa: E402
from src.trading.paper_trader import (  # noqa: E402
    TraderState,
    build_plan,
    data_fresh_for_rebalance,
    execute_plan,
    is_rebalance_day,
    save_run,
)
from src.trading.signal import (  # noqa: E402
    prediction_from_row,
)
from src.utils.config import PROJECT_ROOT, load_config  # noqa: E402
from src.utils.logging import get_logger, setup_logging  # noqa: E402

log = get_logger("paper_trade")
CKPT_DIR = PROJECT_ROOT / "outputs" / "checkpoints"

# 데이터가 이보다 오래되면 오늘 판단의 근거가 낡았다는 뜻이다.
_STALE_BUSINESS_DAYS = 3


def find_checkpoint(explicit: str | None) -> Path:
    """백테스트와 같은 규칙으로 고른다 — 일봉 트랙 것 중 가장 최근 것."""
    if explicit:
        p = Path(explicit)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if not p.exists():
            raise SystemExit(f"체크포인트가 없다: {p}")
        return p

    # ⚠️ 여기는 **실주문 경로**다. 60분봉 트랙 체크포인트(`phase1_*_60m.pt`)가
    #    섞여 들어오면 다른 트랙의 모델로 실제 주문이 나간다 — 에러가 아니라
    #    그럴듯하게 틀린 주문이라 알아채기 어렵다. 이름 형식으로 일봉만 남긴다.
    daily = re.compile(r"^phase1_[0-9a-f]{8}\.pt$")
    cands = [p for p in CKPT_DIR.glob("phase1_*.pt") if daily.match(p.name)]
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



def _busdays(start_iso: str, today: date) -> int:
    """사이클이 며칠째인가. 영업일 기준 — 주말에 사이클이 끝나면 안 된다."""
    return int(np.busday_count(np.datetime64(start_iso, "D"), np.datetime64(today, "D")))


def run_grid(args, cfg, recent, broker, state, today, dry_run: bool) -> int:
    """그리드 모드 — 목표 비중 대신 **지정가 사다리**를 건다.

    선별은 기존 경로를 그대로 쓴다(q50 상위 top_n). 이 함수가 하는 일은
    "고른 종목에 사다리를 어떻게 거느냐" 하나뿐이다 — 절대 규칙 7.

    **사이클 안에서도 폭은 모델이 다시 잡는다.** 매일 새 예측으로 간격을 계산하고,
    `grid.relay_band` 를 넘게 바뀌었으면 미체결 칸을 **취소하고 새 폭으로 다시 건다**.
    기준가는 사이클 시작값을 유지한다(`state.grids`) — 매일 옮기면 매일 재시작이다.
    밴드 안이면 이미 걸린 칸은 그대로 둔다. 같은 칸에 두 번 걸면 두 배를 산다.
    """
    gcfg = cfg["grid"]
    costs = cfg["trading"]["costs"]
    n_stocks = args.grid_stocks or int(cfg["trading"]["direction"]["top_n"])
    cycle_days = int(gcfg.get("cycle_days", 5))

    latest = recent[recent["date"] == recent["date"].max()]
    preds = []
    for r in latest.itertuples():
        try:
            preds.append(prediction_from_row(r))
        except ValueError as exc:
            log.warning("분위 교차 무시: %s", exc)
    # q50 상위 = 모델이 오를 것으로 본 종목
    picked = sorted(preds, key=lambda p: -p.q50)[:n_stocks]
    codes = [p.code for p in picked]
    log.info("그리드 대상 %d종목: %s", len(codes), ", ".join(codes))

    prices = broker.fetch_prices(codes)
    deposit = broker.fetch_deposit()
    budget_total = args.grid_budget or float(deposit.get("orderable", 0))
    per_stock = budget_total / max(len(codes), 1)

    # 무엇이 이미 걸려 있나 — 계좌가 진실이다
    live = broker.fetch_unfilled_detail()
    live_buys: dict[str, list[dict]] = {}
    for d in live:
        if d["side"] == BUY:
            live_buys.setdefault(d["code"], []).append(d)
    if live:
        log.warning("미체결 %d건", len(live))

    print("\n" + "=" * 74)
    print(f"그리드 플로우 — {len(codes)}종목 × {per_stock:,.0f}원"
          + ("   [dry-run — 전송하지 않음]" if dry_run else "   [실주문]"))
    print("=" * 74)
    new_grids: dict[str, dict] = {}
    print(f"  주문가능 {budget_total:,.0f}원 · 간격 = clip(q90−q10 × {gcfg['width_alpha']}, "
          f"왕복비용×{gcfg['floor_mult']}, {gcfg['max_spacing']:.0%})")

    orders, skipped, to_cancel = [], [], []
    for p in picked:
        px = prices.get(p.code)
        if not px:
            skipped.append((p.code, "현재가 조회 실패"))
            continue
        # 사이클 기준가: 진행 중이면 유지, 아니면 오늘 현재가로 새로 연다
        # 중심(조건 02)은 **사이클당 한 번만** 정한다. 진행 중이면 그때 값을 그대로 쓴다 —
        # 매일 다시 옮기면 사다리가 가격을 따라다니고, 그건 그리드가 아니라 매일 재시작이다.
        cyc = state.grids.get(p.code)
        if cyc and _busdays(cyc["date"], today) < cycle_days:
            center, old_spacing = float(cyc["center"]), float(cyc["spacing"])
            lcfg = gcfg | {"center_k": 0.0}
        else:
            center, old_spacing, lcfg = px, 0.0, gcfg
        lad = build_ladder(p, center, per_stock, costs, lcfg)
        if lad.skipped:
            skipped.append((p.code, lad.skipped))
            continue
        mine = live_buys.get(p.code, [])
        relay = should_relay(old_spacing, lad.spacing, gcfg)
        print(f"\n  {p.code}  현재가 {px:>10,.0f}원  기준가 {lad.center:>10,.0f}원  "
              f"폭 {p.interval_width*100:5.2f}% → 간격 {lad.spacing*100:4.2f}%"
              + (f"  (직전 {old_spacing*100:.2f}% → {'재배치' if relay else '유지'})"
                 if old_spacing else "  (새 사이클)"))
        for o in reversed(lad.sells):
            print(f"      매도 +{o.level}  {o.price:>10,}원 {o.quantity:>4}주")
        print(f"      ──기준──  {px:>10,.0f}원")
        for o in lad.buys:
            print(f"      매수 −{o.level}  {o.price:>10,}원 {o.quantity:>4}주")
        # 매수 칸만 건다. 매도 칸은 보유분이 생긴 뒤 짝으로 거는 것이 순서다.
        if mine and not relay:
            print(f"      ⏭  미체결 매수 {len(mine)}건 유지 (폭 변화가 밴드 안)")
            continue
        if mine:
            print(f"      ♻  미체결 매수 {len(mine)}건 취소 후 새 폭으로 재배치")
            to_cancel.extend(mine)
        orders.extend(lad.buys)
        new_grids[p.code] = {"center": lad.center, "spacing": lad.spacing,
                             "date": (cyc["date"] if old_spacing else today.isoformat())}

    if skipped:
        print("\n  제외:")
        for code, why in skipped:
            print(f"    {code}  {why}")

    print(f"\n  취소 {len(to_cancel)}건 · 걸 주문 {len(orders)}건"
          f" · 합계 {sum(o.price * o.quantity for o in orders):,}원")
    if dry_run:
        print("  (dry-run — --execute 를 붙여야 실제로 나간다)")
        return 0

    # 취소가 먼저다 — 새 칸을 먼저 걸면 같은 종목에 두 벌이 걸려 있는 순간이 생긴다
    for d in to_cancel:
        r = broker.cancel_order(d["code"], d["order_no"], d["quantity"], dry_run=False)
        if not r.ok:
            log.error("취소 실패 %s %s: %s — 이 종목 재배치를 접는다",
                      d["code"], d["order_no"], r.error)
            orders = [o for o in orders if o.code != d["code"]]
            new_grids.pop(d["code"], None)

    ok = 0
    for o in orders:
        r = broker.place_order(o.code, o.side, o.quantity,
                               price=o.price, order_type="limit", dry_run=False)
        ok += bool(r.ok)
        if not r.ok:
            log.error("주문 실패 %s %s %d주 @%s: %s", o.code, o.side, o.quantity, o.price, r.error)
    print(f"  전송 완료 {ok}/{len(orders)}건")
    state.grids.update(new_grids)
    state.save()
    return 0 if ok == len(orders) else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint")
    ap.add_argument("--execute", action="store_true",
                    help="실제 모의투자 주문 전송 (기본은 계획만 출력)")
    ap.add_argument("--force-rebalance", action="store_true",
                    help="리밸런싱 주기를 무시하고 이번에 리밸런싱한다")
    ap.add_argument("--liquidate", action="store_true",
                    help="보유 전 종목을 매도한다 (매수 없음). 전략 교체 시 초기화용")
    ap.add_argument("--recent-days", type=int,
                    help="기권 임계값을 잡을 예측 폭 분포의 관측 구간(일). 기본은 "
                         "trading.abstain.recent_days — 백테스트와 같은 값")
    ap.add_argument("--grid", action="store_true",
                    help="그리드 플로우 모드 — 목표 비중 대신 지정가 사다리를 건다")
    ap.add_argument("--grid-stocks", type=int,
                    help="사다리를 깔 종목 수 (기본: trading.direction.top_n)")
    ap.add_argument("--grid-budget", type=float,
                    help="그리드에 쓸 총 예산(원). 기본은 주문가능금액 전액")
    ap.add_argument("--ignore-stale", action="store_true",
                    help="데이터가 오래돼도 진행한다 — 리밸런싱 차단까지 푼다 (권장하지 않음)")
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
    recent_days = args.recent_days or int(cfg["trading"]["abstain"]["recent_days"])
    recent = predict_recent(loaded, bundle, cfg, days=recent_days)

    # --- 3) 계좌
    with PaperBroker() as broker:
        account = broker.snapshot()
        # 미체결 조회는 계획 수립 **전에** 한다. 실패하면 예외로 멈춘다 —
        # '미체결 없음'으로 오해하고 주문을 얹는 것보다 안 내는 쪽이 낫다.
        unfilled = broker.fetch_unfilled()
        state = TraderState.load()
        state.sync_entries(set(account.holdings), today)

        if args.grid:
            _print_holdings(account)
            return run_grid(args, cfg, recent, broker, state, today, dry_run)

        rebalancing = args.force_rebalance or is_rebalance_day(
            state, today, int(cfg["backtest"].get("rebalance_days", 5))
        )

        # 낡은 패널이면 신규 진입만 막는다. `--force-rebalance` 로도 안 뚫린다 —
        # 그 옵션은 '주기를 무시한다'는 뜻이지 '낡은 예측을 써도 좋다'가 아니다.
        skip_reason = "리밸런싱 주기가 아니다"
        if rebalancing and not args.ignore_stale and not data_fresh_for_rebalance(
            bundle.last_date, today
        ):
            rebalancing = False
            skip_reason = f"패널이 {bundle.last_date} 에 멈춰 있다(전 거래일보다 낡았다)"
            log.warning(
                "%s — 리밸런싱을 건너뛴다. 수집·피처를 갱신하고 다시 실행할 것 "
                "(강행: --ignore-stale)", skip_reason,
            )

        # --- 4) 1차 계획: 종가 기준으로 '무엇을 건드릴지'만 정한다.
        #     기권·순위·사이징은 가격과 무관하므로 이 단계에서 확정된다.
        #     현재가는 그 다음, 실제로 건드릴 종목만 조회한다(호출 수 절약).
        closes = latest_prices(bundle)
        # CVaR 한도(리스크 오버레이 6단계)용 수익률 행렬. 패널은 이미 과거만 담고 있다.
        # `risk.cvar_limit` 이 null 인 동안은 **측정만 되고 주문은 바뀌지 않는다.**
        rets = (bundle.raw_panel
                .pivot_table(index="date", columns="code", values="close")
                .sort_index().pct_change())
        draft = build_plan(recent, account, closes, cfg, state=state,
                           today=today, rebalancing=rebalancing,
                           rebalance_skip_reason=skip_reason,
                           liquidate_all=args.liquidate, unfilled=unfilled,
                           returns=rets)

        touch = sorted({o.code for o in draft.orders} | set(account.holdings))
        quotes = broker.fetch_prices(touch) if touch else {}
        log.info("현재가 조회 %d종목 (계획 대상 + 보유분)", len(quotes))

        # --- 5) 최종 계획: 손절 판정과 주문 수량이 현재가 기준이 된다
        prices = {**closes, **quotes}
        plan = build_plan(recent, account, prices, cfg, state=state,
                          today=today, rebalancing=rebalancing,
                          rebalance_skip_reason=skip_reason,
                          liquidate_all=args.liquidate, unfilled=unfilled,
                          returns=rets)
        plan.notes.extend(notes)

        _print_holdings(account)
        _print_plan(plan, account, dry_run)

        # --- 6) 전송
        results = execute_plan(broker, plan, dry_run=dry_run)

        # 전량 청산 뒤에는 **대금이 언제 쓸 수 있게 되는지**가 다음 행동을 가른다.
        # 추정하지 않고 바로 다시 조회해서 보여준다 (paper_trader 의 원칙과 같다).
        if args.liquidate and not dry_run:
            after = broker.fetch_deposit()
            log.info("청산 후 주문가능금액: %s원 (청산 전 %s원)",
                     f"{float(after.get('orderable', 0)):,.0f}", f"{account.orderable:,.0f}")

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
