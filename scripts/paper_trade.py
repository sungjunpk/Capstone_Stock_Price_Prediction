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
from src.trading.broker import BUY, SELL, PaperBroker  # noqa: E402
from src.trading.grid import (  # noqa: E402
    build_ladder,
    detect_fills,
    outside_ladder,
    reactions_to_fills,
    realized_vol,
    relay_spacing,
    sell_ladder_for_holding,
    should_relay,
    stop_loss_hit,
)
from src.trading.paper_trader import (  # noqa: E402
    TraderState,
    build_plan,
    data_fresh_for_rebalance,
    execute_plan,
    is_rebalance_day,
    save_run,
)
from src.trading.signal import (  # noqa: E402
    Action,
    QuantilePrediction,
    abstain_scores,
    generate_signals,
    prediction_from_row,
    resolve_abstain_threshold,
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


def _latest_preds(recent) -> dict[str, QuantilePrediction]:
    """판단 기준일의 예측만 꺼낸다."""
    latest = recent[recent["date"] == recent["date"].max()]
    out: dict[str, QuantilePrediction] = {}
    for r in latest.itertuples():
        try:
            p = prediction_from_row(r)
        except ValueError as exc:
            log.warning("분위 교차 무시: %s", exc)
            continue
        out[p.code] = p
    return out


def _pick_grid_codes(recent, cfg, held: set[str], n: int) -> list[QuantilePrediction]:
    """선별은 `signal.generate_signals` 하나만 쓴다 — **절대 규칙 7.**

    기권 → 시총풀(`mcap_pool`) → q50 순위 → 이력 버퍼가 전부 거기서 돈다.
    그리드가 자체 순위를 매기면 같은 판단이 두 벌이 되고, 실제로 그랬다
    (2026-09-27 발견: 기권률 설정과 시총풀이 그리드 경로에서 통째로 무시되고 있었다).
    """
    tcfg = cfg["trading"]
    by_code = _latest_preds(recent)
    max_width = resolve_abstain_threshold(
        abstain_scores(recent, tcfg["abstain"]), tcfg["abstain"]
    )
    signals = generate_signals(list(by_code.values()), tcfg,
                              max_width=max_width, held=held)
    chosen = [s.code for s in signals if s.action == Action.BUY]
    # generate_signals 는 입력 순서로 돌려준다 — 순위는 여기서 다시 세운다
    return sorted((by_code[c] for c in chosen), key=lambda p: -p.q50)[:n]


def _ladder_of(cyc: dict, pred: QuantilePrediction, gcfg: dict, costs: dict,
               spacing: float | None = None):
    """**진행 중인** 사이클의 사다리를 저장된 기준가·예산으로 재현한다.

    중심(조건 02)은 사이클당 한 번만 정하므로 `center_k` 를 끈다 — 매 회차 다시
    옮기면 사다리가 가격을 따라다니고, 그건 그리드가 아니라 매 회차 재시작이다.
    새 사이클은 이 함수를 쓰지 않는다: `build_ladder` 를 현재가로 직접 불러
    `center_k` 가 중심을 옮기게 한다.
    """
    # 배정액을 모르면 `build_ladder` 가 '칸당 예산 < 1주' 로 스스로 건너뛴다 —
    # 옛 형식 상태 파일에 KeyError 를 내는 것보다 사다리를 안 깔고 넘어가는 쪽이다.
    return build_ladder(pred, float(cyc["center"]), float(cyc.get("budget") or 0),
                        costs, gcfg | {"center_k": 0.0}, spacing)


def _open_cycle(pred: QuantilePrediction, px: float, budget: float,
                today, gcfg: dict, costs: dict):
    """새 사이클 — **현재가**에서 열고 `center_k` 가 중심을 옮긴다 (조건 02).

    진행 중인 사이클은 `_ladder_of` 를 쓴다. 차이가 여기 하나다: 중심을 정하는
    것은 사이클당 한 번이고, 그 한 번이 이 함수다.
    """
    lad = build_ladder(pred, px, budget, costs, gcfg)
    return lad, {"center": lad.center, "spacing": 0.0, "date": today.isoformat(),
                 "budget": budget, "rungs": {}, "stopped": False}


def _busy_sell_levels(rungs: dict) -> frozenset[int]:
    """**상단** 매도 칸 중 이미 걸려 있는 것. 같은 칸에 두 번 걸면 두 배를 판다.

    음수 level 은 매수 칸의 짝 매도라 중심 *아래*에 있다 — 다른 축이므로 센다면
    엉뚱한 상단 칸을 막는다(`grid.reactions_to_fills` 경고).
    """
    return frozenset(int(r["level"]) for r in rungs.values()
                     if r["side"] == SELL and int(r["level"]) > 0)


def _place_limits(broker, orders: list, state, dry_run: bool) -> int:
    """지정가 칸 전송 + **주문번호를 rungs 에 적는다.**

    적어두지 않으면 다음 회차가 체결을 못 본다 — 그게 이 함수의 본체다.
    매도를 먼저 보내고 매수를 나중에 보낸다(`execute_plan` 과 같은 순서).
    """
    ok = 0
    for o in sorted(orders, key=lambda o: o.side != SELL):
        r = broker.place_order(o.code, o.side, o.quantity,
                               price=o.price, order_type="limit", dry_run=dry_run)
        if not r.ok:
            log.error("주문 실패 %s %s %d주 @%s: %s",
                      o.code, o.side, o.quantity, o.price, r.error)
            continue
        ok += 1
        if not dry_run and r.order_no:
            state.grids[o.code].setdefault("rungs", {})[r.order_no] = {
                "side": o.side, "level": o.level, "price": o.price,
                "qty": o.quantity, "filled": 0,
            }
    return ok


def _print_ladder(lad, px: float, tag: str) -> None:
    print(f"\n  {lad.code}  현재가 {px:>10,.0f}원  기준가 {lad.center:>10,.0f}원  "
          f"간격 {lad.spacing*100:4.2f}%  {tag}")
    for o in reversed(lad.sells):
        print(f"      매도 +{o.level}  {o.price:>10,}원 {o.quantity:>4}주")
    print(f"      ──기준──  {lad.center:>10,.0f}원")
    for o in lad.buys:
        print(f"      매수 −{o.level}  {o.price:>10,}원 {o.quantity:>4}주")


def _grid_intraday(cfg, recent, broker, account, state, today, dry_run: bool) -> int:
    """장중 회차 — 체결에 반응한다. 선별은 없다.

    지정가가 살아 있는 동안이므로 `detect_fills` 가 옳다(사라짐 = 진짜 체결).
    매도 칸이 비어 있으면 보유분으로 채운다 — 아침에 낸 초기 시장가 매수가
    이 회차에 잡히는 자리다(체결 지연 8~10초, `docs/KIWOOM_VERIFY.md`).

    **사다리 이탈만 장중에 사이클을 다시 연다**(`outside_ladder`, 아침 회차와 같은
    조건·같은 처리). 벗어난 사다리는 체결될 칸이 0 이라 다음 아침까지 그 종목이
    통째로 멈춘다. 폭은 장중에 다시 잡지 않는다 — 폭의 입력(`q90−q10`)이 일봉
    예측이라 장중에 바뀌지 않고, 가격으로 바꾸면 결정권이 모델 밖으로 나간다
    (`docs/REFERENCES.md` 7.7 의 실현변동성 대조군).
    """
    gcfg, costs = cfg["grid"], cfg["trading"]["costs"]
    by_code = _latest_preds(recent)
    live = broker.fetch_unfilled_detail()
    # 이탈 판정에만 쓴다. 종목당 1회 호출이라 사이클이 걸린 종목만 본다
    prices = broker.fetch_prices([c for c, g in state.grids.items()
                                  if not g.get("stopped") and c in by_code])

    print("\n" + "=" * 74)
    print("그리드 플로우 — 장중 회차 (체결 반응)"
          + ("   [dry-run — 전송하지 않음]" if dry_run else "   [실주문]"))
    print("=" * 74)

    # 간격 재조정 — **모델이 아니라 가격**이 폭을 바꾼다(`grid.spacing_from_volatility`).
    # 켜 두는 동안 조건 01 의 결정권은 아침 사다리 한 번에만 남는다.
    rcfg = gcfg.get("intraday_relay") or {}
    relay_on = bool(rcfg.get("enabled"))
    orders, rebuild, seen_fills, relayed = [], [], 0, 0
    for code, cyc in state.grids.items():
        if cyc.get("stopped") or code not in by_code:
            continue
        rungs = cyc.setdefault("rungs", {})
        lad = _ladder_of(cyc, by_code[code], gcfg, costs)
        if lad.skipped:
            continue

        # 사다리 이탈 → 아래에서 사이클을 다시 연다. 체결 반응은 건너뛴다 —
        # 칸을 전부 거두므로 역산할 게 없고, 보유수량이 새 사다리를 덮는다.
        px = prices.get(code)
        if px and outside_ladder(lad, px):
            rebuild.append((code, px, None))
            continue

        # 폭이 밴드 밖으로 움직였나. 못 재면(종가 부족·아침 변동성 없음) 그대로 둔다
        if relay_on and cyc.get("vol_open"):
            now = realized_vol(
                broker.fetch_recent_closes(code, int(rcfg.get("vol_bars", 39)))
            )
            new_sp = relay_spacing(float(cyc.get("spacing_open") or 0),
                                   float(cyc["vol_open"]), now or 0.0, costs, gcfg)
            band = {"relay_band": float(rcfg.get("band", 0.20))}
            if new_sp and should_relay(float(cyc.get("spacing") or 0), new_sp, band):
                rebuild.append((code, None, new_sp))
                continue

        reactions = []
        for f in detect_fills(rungs, live):
            seen_fills += 1
            r = rungs[f.order_no]
            r["filled"] = int(r.get("filled", 0)) + f.quantity
            print(f"  {code}  {'매수' if f.side == BUY else '매도'} {f.level}칸 "
                  f"{f.price:>10,}원 {f.quantity:>4}주 체결")
            reactions.extend(reactions_to_fills([f], lad, costs))
            if int(r["filled"]) >= int(r["qty"]):
                rungs.pop(f.order_no, None)
        # ponytail: 반응 주문 전송이 실패하면 같은 회차에 재시도하지 않는다. 매수 체결
        # 쪽은 아래 '매도 칸 보충'이 다음 회차에 자기치유하지만, 매도 체결 → 재장전
        # 실패는 **아침 사다리 재구성까지** 그 칸이 빈다. 하루치 손실이라 감수한다 —
        # 같은 날 복구가 필요해지면 실패분을 state 에 남겨 다음 회차가 집어야 한다.
        orders.extend(reactions)

        # 보유분 중 매도 칸이 안 덮은 수량을 채운다 (비어 있으면 아무것도 안 나간다).
        # **이번 회차에 만든 짝 매도까지 세야 한다** — rungs 에는 아직 없으므로
        # 빼놓으면 같은 칸에 두 번 걸리고 매도가능수량을 넘긴다.
        pending = [o for o in reactions if o.side == SELL]
        busy = _busy_sell_levels(rungs) | {o.level for o in pending if o.level > 0}
        h = account.holdings.get(code)
        room = (h.sellable - sum(o.quantity for o in pending)) if h else 0
        if room > 0:
            fill_in = sell_ladder_for_holding(lad, room, busy)
            if fill_in:
                print(f"  {code}  매도 칸 보충 {len(fill_in)}건 "
                      f"(매도가능 {h.sellable}주 − 짝 매도 예약 "
                      f"{sum(o.quantity for o in pending)}주)")
                orders.extend(fill_in)

    for code, px, new_sp in rebuild:
        cyc = state.grids[code]
        old_sp = float(cyc.get("spacing") or 0)
        if px:                      # 이탈 — 기준가까지 다시 잡는다
            lad, new_cyc = _open_cycle(by_code[code], px, float(cyc.get("budget") or 0),
                                       today, gcfg, costs)
            why = (f"사다리 이탈(현재가 {px:,.0f}원) — "
                   f"기준가 {lad.center:,.0f}원으로 사이클을 다시 연다")
        else:                       # 간격만 — 기준가는 그대로 둔다
            lad = _ladder_of(cyc, by_code[code], gcfg, costs, new_sp)
            new_cyc = {**cyc, "rungs": {}}
            relayed += 1
            why = (f"간격 {old_sp*100:.2f}% → {new_sp*100:.2f}% "
                   f"(밴드 ±{float(rcfg.get('band', 0.20)):.0%}) — 사다리를 다시 건다")
        if lad.skipped:
            print(f"  {code}  {why} — 실패: {lad.skipped}")
            continue
        print(f"  ↻ {code} {why}")
        if not dry_run:
            for d in (d for d in live if d["code"] == code):
                r = broker.cancel_order(code, d["order_no"], d["unfilled_qty"],
                                        dry_run=False)
                if not r.ok:
                    log.error("취소 실패 %s %s: %s", code, d["order_no"], r.error)
        state.grids[code] = {**new_cyc, "spacing": lad.spacing}
        orders.extend(lad.buys)
        h = account.holdings.get(code)
        if h and h.sellable > 0:
            orders.extend(sell_ladder_for_holding(lad, h.sellable))
        # ponytail: 초기 시장가 매수는 안 낸다 — 장중 회차는 지정가만 내고, 아래로
        # 이탈한 경우엔 예산이 이미 사다리에 들어가 있어 중복 지출이 된다. 위로
        # 이탈해 보유가 0 이면 다음 아침까지 매수 칸만 사는 대가를 진다.

    if not orders:
        # 반응할 주문이 없어도 **체결·재개를 봤으면 저장한다** — 안 그러면 다음
        # 회차가 같은 체결을 다시 잡거나 거둔 사다리로 되돌아간다.
        if (seen_fills or rebuild) and not dry_run:
            state.save()
        print(f"  반응 주문 없음 (체결 {seen_fills}건) — 걸어둔 칸을 그대로 둔다")
        return 0

    print(f"\n  반응 주문 {len(orders)}건 · 체결 {seen_fills}건 · "
          f"재배치 {len(rebuild)}건(간격 {relayed})")
    for o in orders:
        print(f"    {o.code}  {'매수' if o.side == BUY else '매도'} {o.level}칸 "
              f"{o.price:>10,}원 {o.quantity:>4}주")
    if dry_run:
        print("  (dry-run — --execute 를 붙여야 실제로 나간다)")
        return 0

    ok = _place_limits(broker, orders, state, dry_run=False)
    print(f"  전송 완료 {ok}/{len(orders)}건")
    state.save()
    return 0 if ok == len(orders) else 1


def _grid_morning(args, cfg, recent, broker, account, state, today, dry_run: bool) -> int:
    """아침 첫 회차 — 사이클을 정리하고 사다리를 **새로** 깐다.

    전날 지정가는 장 마감에 소멸했으므로 `rungs` 를 믿지 않는다. 순서가 중요하다:
    취소·청산을 먼저 보내고 예수금을 **다시 조회한** 뒤 새 사이클 예산을 정한다 —
    매도 대금이 당일 주문가능금액에 언제 반영되는지는 계좌 설정에 따라 다르다.
    """
    gcfg, costs = cfg["grid"], cfg["trading"]["costs"]
    cycle_days = int(gcfg.get("cycle_days", 5))
    n_stocks = args.grid_stocks or int(gcfg["stocks"])

    picked = _pick_grid_codes(recent, cfg, set(account.holdings), n_stocks)
    codes = [p.code for p in picked]
    log.info("그리드 대상 %d종목: %s", len(codes), ", ".join(codes))

    live = broker.fetch_unfilled_detail()
    print("\n" + "=" * 74)
    print(f"그리드 플로우 — 아침 첫 회차 · {len(codes)}종목"
          + ("   [dry-run — 전송하지 않음]" if dry_run else "   [실주문]"))
    print("=" * 74)

    # --- 1) 사이클 정리. **`state.grids` 에 있는 종목만 건드린다** —
    #        계좌에 섞인 구 전략 보유분을 그리드가 팔면 안 된다.
    liquidate, reopen = [], []
    for code, cyc in list(state.grids.items()):
        expired = _busdays(str(cyc["date"]), today) >= cycle_days
        if cyc.get("stopped"):
            if expired:
                state.grids.pop(code)      # 사이클 끝 — 다시 뽑힐 수 있다
            continue
        h = account.holdings.get(code)
        budget = float(cyc.get("budget") or 0)
        if h and stop_loss_hit(h.pnl_amount, budget, gcfg):
            print(f"  ⛔ {code} 손절 — 미실현 {h.pnl_amount:,.0f}원 / 배정 {budget:,.0f}원 "
                  f"(한도 {gcfg['stop_loss_pct']:.0%})")
            liquidate.append(code)
            cyc["stopped"] = True
        elif expired and code not in codes:
            print(f"  ✂ {code} 사이클 만료 + 선별 밖 — 청산")
            liquidate.append(code)
        elif expired:
            print(f"  ↻ {code} 사이클 만료 — 기준가를 다시 잡는다 (보유 유지)")
            reopen.append(code)

    # 어제 칸이 남아 있을 이유는 없지만(장 마감 소멸) 같은 날 두 번 돌면 남는다
    to_cancel = [d for d in live if d["code"] in state.grids]
    sell_all = [(c, account.holdings[c]) for c in liquidate if c in account.holdings]
    if to_cancel or sell_all:
        print(f"\n  취소 {len(to_cancel)}건 · 시장가 청산 {len(sell_all)}종목")

    if not dry_run:
        for d in to_cancel:
            r = broker.cancel_order(d["code"], d["order_no"], d["unfilled_qty"],
                                   dry_run=False)
            if not r.ok:
                log.error("취소 실패 %s %s: %s", d["code"], d["order_no"], r.error)
            else:
                (state.grids[d["code"]].get("rungs") or {}).pop(d["order_no"], None)
        for code, h in sell_all:
            r = broker.place_order(code, SELL, h.sellable, order_type="market",
                                   dry_run=False)
            if not r.ok:
                log.error("청산 실패 %s %d주: %s", code, h.sellable, r.error)
    for code in liquidate:
        if state.grids.get(code, {}).get("stopped"):
            state.grids[code]["rungs"] = {}     # 손절은 사이클 끝까지 재진입 금지
        else:
            state.grids.pop(code, None)

    # --- 2) 예수금 재조회 → 새 사이클 예산. 추정하지 않는다.
    #
    # 예산은 **처음 여는 종목**에만 나눈다. 사이클을 다시 여는 종목은 이미 배정액이
    # 있고 보유분도 그 돈으로 산 것이다 — 새로 나누면 손절 분모(미실현손익 / 배정액)가
    # 어긋나 정상 보유를 손절로 오판정한다.
    brand_new = [p for p in picked
                 if p.code not in state.grids and p.code not in account.holdings]
    deposit = broker.fetch_deposit()
    budget_total = args.grid_budget or float(deposit.get("orderable", 0))
    per_new = budget_total / max(len(brand_new), 1)
    print(f"\n  주문가능 {budget_total:,.0f}원 · 처음 여는 {len(brand_new)}종목 "
          f"× {per_new:,.0f}원 · 간격 = clip(q90−q10 × {gcfg['width_alpha']}, "
          f"왕복비용×{gcfg['floor_mult']}, {gcfg['max_spacing']:.0%})")

    prices = broker.fetch_prices(codes)
    # 장중 재조정을 켠 경우에만 아침 변동성을 잰다 (종목당 1콜)
    rcfg = gcfg.get("intraday_relay") or {}
    vol_open = (
        {c: realized_vol(broker.fetch_recent_closes(c, int(rcfg.get("vol_bars", 39))))
         for c in codes}
        if rcfg.get("enabled") else {}
    )

    # --- 3) 사다리. 진행 중인 사이클은 기준가·예산을 유지하고 폭만 다시 잡는다.
    orders, init_buys, skipped = [], [], []
    for p in picked:
        px = prices.get(p.code)
        if not px:
            skipped.append((p.code, "현재가 조회 실패"))
            continue
        cyc = state.grids.get(p.code)
        if cyc and cyc.get("stopped"):
            skipped.append((p.code, "손절된 종목 — 사이클 끝까지 재진입 금지"))
            continue
        # **그리드가 안 산 보유분이 있으면 사다리를 깔지 않는다.**
        # 깔면 그 수량이 매도 칸으로 편입돼 그리드와 무관한 가격에 팔려나가고,
        # 손절 분모(미실현손익 / 배정액)에도 섞여 정상 보유가 손절로 오판정된다
        # (2026-09-27 감사에서 드러났다 — 그때 계좌에 구 전략 11종목이 있었다).
        if cyc is None and p.code in account.holdings:
            skipped.append((p.code, f"그리드 밖 보유 "
                                    f"{account.holdings[p.code].quantity}주 — 편입하지 않는다"))
            continue
        fresh = cyc is None or p.code in reopen
        if fresh:
            # 사이클을 다시 여는 종목은 **기존 배정액을 유지한다**
            lad, cyc = _open_cycle(
                p, px, per_new if cyc is None else float(cyc["budget"]),
                today, gcfg, costs,
            )
        else:
            lad = _ladder_of(cyc, p, gcfg, costs)
            # 예외 하나 — 가격이 사다리 바깥이면 더 체결될 칸이 없다. 현재가로 다시 연다
            if not lad.skipped and outside_ladder(lad, px):
                log.info("%s 사다리 이탈(현재가 %.0f원) — 사이클을 새로 연다", p.code, px)
                fresh = True
                lad, cyc = _open_cycle(p, px, float(cyc["budget"]), today, gcfg, costs)
        if lad.skipped:
            skipped.append((p.code, lad.skipped))
            continue

        old_spacing = float(cyc.get("spacing") or 0)
        tag = ("(새 사이클)" if fresh else
               f"(직전 {old_spacing*100:.2f}% → "
               f"{'재배치' if should_relay(old_spacing, lad.spacing, gcfg) else '유지'})")
        _print_ladder(lad, px, tag)

        # 장중 재조정의 분모 — 아침 폭과 그때의 변동성. 비율만 장중에 곱한다
        base = {"spacing": lad.spacing, "spacing_open": lad.spacing,
                "vol_open": vol_open.get(p.code)}
        state.grids[p.code] = {**cyc, **base}
        orders.extend(lad.buys)
        h = account.holdings.get(p.code)
        if fresh and lad.init_quantity > 0:
            # 초기 매수는 **시장가**다 — 재고가 생겨야 매도 칸이 성립한다.
            # 체결 지연 8~10초라 매도 칸은 다음 회차가 보충한다.
            init_buys.append((p.code, lad.init_quantity, px))
            print(f"      초기매수  시장가 {lad.init_quantity:>4}주 "
                  f"(약 {lad.init_quantity * px:,.0f}원)")
        if h and h.sellable > 0:
            fill_in = sell_ladder_for_holding(lad, h.sellable)
            orders.extend(fill_in)
            if fill_in:
                print(f"      매도 칸 {len(fill_in)}건 ← 보유 {h.sellable}주")

    if skipped:
        print("\n  제외:")
        for code, why in skipped:
            print(f"    {code}  {why}")

    print(f"\n  초기매수 {len(init_buys)}건 · 걸 주문 {len(orders)}건"
          f" · 지정가 합계 {sum(o.price * o.quantity for o in orders):,}원")
    if dry_run:
        print("  (dry-run — --execute 를 붙여야 실제로 나간다)")
        return 0

    for code, qty, _ in init_buys:
        r = broker.place_order(code, BUY, qty, order_type="market", dry_run=False)
        if not r.ok:
            log.error("초기매수 실패 %s %d주: %s", code, qty, r.error)
    ok = _place_limits(broker, orders, state, dry_run=False)
    print(f"  전송 완료 {ok}/{len(orders)}건")
    state.grid_session = today.isoformat()
    state.save()
    return 0 if ok == len(orders) else 1


def run_grid(args, cfg, recent, broker, account, state, today, dry_run: bool) -> int:
    """그리드 플로우 — 목표 비중 대신 **지정가 사다리**를 건다.

    선별은 `signal.generate_signals` 하나만 쓴다(절대 규칙 7). 이 함수가 정하는 것은
    "고른 종목에 사다리를 어떻게 거느냐" 하나뿐이다.

    **미체결 지정가는 장 마감에 소멸한다**(`docs/KIWOOM_VERIFY.md` 지정가 검증).
    그래서 장중 30분 간격으로 돌고 회차가 둘로 갈린다:

        아침 첫 회차   사이클 정리 → 청산 → 예수금 재조회 → 사다리를 새로 깐다
        장중 회차      체결에 반응 (짝 매도 / 재장전 / 매도 칸 보충)
                       + 사다리 이탈이면 그 종목만 사이클을 다시 연다

    구분은 `state.grid_session` 이 오늘인가로 한다. 날을 넘겨 `detect_fills` 를
    부르면 소멸을 체결로 오인해 **사지도 않은 수량에 매도를 건다.**
    """
    if state.grid_session == today.isoformat():
        return _grid_intraday(cfg, recent, broker, account, state, today, dry_run)
    return _grid_morning(args, cfg, recent, broker, account, state, today, dry_run)


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
                    help="사다리를 깔 종목 수 (기본: grid.stocks)")
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
            return run_grid(args, cfg, recent, broker, account, state, today, dry_run)

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
