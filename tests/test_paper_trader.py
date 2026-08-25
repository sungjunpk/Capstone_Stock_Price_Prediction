"""모의투자 실행기 테스트 — API 없이 판단·수량 환산만 검증한다.

여기서 지키려는 것:
  1) 목표 비중 → 정수 주 환산이 총자산 기준으로 맞는가
  2) 전량 청산이 최소 거래폭 밴드에 막히지 않는가 (막히면 손절이 무력화된다)
  3) 리밸런싱 날이 아니면 신규 진입을 하지 않는가
  4) 손절 기준가가 **브로커 매입단가**인가 (로컬 상태가 아니라)
  5) 매수가 주문가능금액을 못 넘는가
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from src.trading.broker import BUY, SELL, AccountSnapshot, Holding, OrderResult
from src.trading.paper_trader import (
    TraderState,
    build_plan,
    execute_plan,
    is_rebalance_day,
)

TODAY = date(2026, 8, 25)


def make_cfg(**over) -> dict:
    cfg = {
        "trading": {
            "abstain": {"percentile": 100},          # 전원 통과 — 순위만 본다
            "direction": {"mode": "cross_sectional", "top_n": 2, "exit_rank": 2,
                          "min_candidates": 0, "long_threshold": 0.004,
                          "short_threshold": -0.004},
            "sizing": {"method": "rank_normalized", "exposure_scaling": False,
                       "max_position_pct": 0.5, "min_trade_weight": 0.01},
            "risk": {"stop_loss_pct": -0.05, "take_profit_pct": 0.10,
                     "max_trades_per_day": 20, "max_gross_exposure": 1.0},
            "costs": {"commission_bps": 1.5, "tax_bps": 18.0, "slippage_bps": 5.0},
        },
        "backtest": {"rebalance_days": 5},
    }
    cfg["trading"].update(over)
    return cfg


def make_preds(rows: list[tuple[str, float, float, float]], n_days: int = 3) -> pd.DataFrame:
    """최근 구간 예측. 마지막 날짜가 판단 대상이 된다."""
    frames = []
    for i in range(n_days):
        d = date(2026, 8, 21 + i)
        frames.append(pd.DataFrame(
            [{"code": c, "date": d, "q10": lo, "q50": mid, "q90": hi}
             for c, lo, mid, hi in rows]
        ))
    return pd.concat(frames, ignore_index=True)


def account(cash: float, holdings: list[Holding] = (), deposit=None) -> AccountSnapshot:
    hs = {h.code: h for h in holdings}
    return AccountSnapshot(
        cash=cash,
        deposit=cash if deposit is None else deposit,
        holdings=hs,
        total_eval=sum(h.eval_amount for h in hs.values()),
    )


class TestWeightToQuantity:
    def test_buys_integer_shares_from_equity(self):
        """비중 × 총자산 ÷ 가격 → 내림. 1주 미만이면 사지 않는다."""
        cfg = make_cfg()
        preds = make_preds([("AAA", -0.01, 0.02, 0.03), ("BBB", -0.01, 0.01, 0.03)])
        acct = account(1_000_000)
        plan = build_plan(preds, acct, {"AAA": 30_000, "BBB": 7_000},
                          cfg, state=TraderState(), today=TODAY)

        by_code = {o.code: o for o in plan.orders}
        assert set(by_code) == {"AAA", "BBB"}
        assert all(o.side == BUY for o in plan.orders)
        # max_position_pct 0.5 → 종목당 50만원. 30,000원짜리는 16주.
        assert by_code["AAA"].quantity == 16
        assert by_code["BBB"].quantity == 71
        assert by_code["AAA"].amount <= 0.5 * acct.equity + 1

    def test_price_too_high_yields_no_order(self):
        cfg = make_cfg()
        preds = make_preds([("AAA", -0.01, 0.02, 0.03)])
        plan = build_plan(preds, account(100_000), {"AAA": 900_000},
                          cfg, state=TraderState(), today=TODAY)
        assert plan.orders == []

    def test_sell_capped_by_sellable(self):
        """미결제분은 팔 수 없다 — 브로커가 거부하기 전에 우리가 줄인다."""
        cfg = make_cfg()
        held = Holding("OLD", "구", quantity=100, sellable=40, avg_price=10_000,
                       current_price=10_000, eval_amount=1_000_000)
        preds = make_preds([("AAA", -0.01, 0.02, 0.03)])
        plan = build_plan(preds, account(0, [held]), {"AAA": 10_000, "OLD": 10_000},
                          cfg, state=TraderState(), today=TODAY)

        sells = [o for o in plan.orders if o.side == SELL]
        assert len(sells) == 1
        assert sells[0].code == "OLD"
        assert sells[0].quantity == 40      # 100주 보유지만 40주만


class TestExitAlwaysPasses:
    def test_liquidation_ignores_min_trade_band(self):
        """비중 변화가 밴드보다 작아도 **전량 청산은 나간다.**

        밴드가 청산을 막으면 손절이 무력화되고 포지션이 영원히 남는다.
        """
        cfg = make_cfg()
        cfg["trading"]["sizing"]["min_trade_weight"] = 0.9   # 사실상 모든 조정을 막는다
        tiny = Holding("OLD", "구", quantity=1, sellable=1, avg_price=1_000,
                       current_price=1_000, eval_amount=1_000)
        preds = make_preds([("AAA", -0.01, 0.02, 0.03)])
        plan = build_plan(preds, account(999_000, [tiny]),
                          {"AAA": 10_000, "OLD": 1_000},
                          cfg, state=TraderState(), today=TODAY)

        assert [o.code for o in plan.orders if o.side == SELL] == ["OLD"]

    def test_stop_loss_uses_broker_average_price(self):
        """손절 기준가는 브로커 매입단가다. 로컬 상태 파일이 비어도 동작해야 한다."""
        cfg = make_cfg()
        loser = Holding("BAD", "손실", quantity=10, sellable=10, avg_price=10_000,
                        current_price=9_000, eval_amount=90_000)
        preds = make_preds([("BAD", -0.01, 0.05, 0.06)])   # 신호는 매수(1위)
        plan = build_plan(preds, account(0, [loser]), {"BAD": 9_000},
                          cfg, state=TraderState(), today=TODAY)

        assert "BAD" in plan.forced_exits          # -10% → 손절이 신호를 이긴다
        assert "손절" in plan.blocked_by_reason
        assert [o.side for o in plan.orders] == [SELL]


class TestRebalanceCadence:
    def test_non_rebalance_day_makes_no_new_entries(self):
        cfg = make_cfg()
        preds = make_preds([("AAA", -0.01, 0.02, 0.03)])
        plan = build_plan(preds, account(1_000_000), {"AAA": 10_000},
                          cfg, state=TraderState(), today=TODAY, rebalancing=False)

        assert plan.orders == []
        assert plan.rebalancing is False

    def test_non_rebalance_day_still_stops_out(self):
        """주기가 아니어도 손절은 나간다."""
        cfg = make_cfg()
        loser = Holding("BAD", "손실", quantity=10, sellable=10, avg_price=10_000,
                        current_price=9_000, eval_amount=90_000)
        preds = make_preds([("BAD", -0.01, 0.05, 0.06)])
        plan = build_plan(preds, account(0, [loser]), {"BAD": 9_000},
                          cfg, state=TraderState(), today=TODAY, rebalancing=False)

        assert plan.forced_exits == ["BAD"]
        assert [o.side for o in plan.orders] == [SELL]

    def test_non_rebalance_day_keeps_unsignaled_holdings(self):
        """신호가 없다는 이유로 보유분을 팔지 않는다 — 청산은 리밸런싱 때만."""
        cfg = make_cfg()
        keep = Holding("KEEP", "유지", quantity=10, sellable=10, avg_price=10_000,
                       current_price=10_200, eval_amount=102_000)
        preds = make_preds([("AAA", -0.01, 0.02, 0.03)])
        plan = build_plan(preds, account(0, [keep]), {"AAA": 10_000, "KEEP": 10_200},
                          cfg, state=TraderState(), today=TODAY, rebalancing=False)
        assert plan.orders == []

    @pytest.mark.parametrize(
        "last,expected", [(None, True), ("2026-08-24", False), ("2026-08-18", True)]
    )
    def test_is_rebalance_day(self, last, expected):
        state = TraderState(last_rebalance=last)
        assert is_rebalance_day(state, TODAY, 5) is expected


class TestState:
    def test_days_held_counts_business_days(self):
        state = TraderState(entry_dates={"AAA": "2026-08-18"})
        assert state.days_held("AAA", TODAY) == 5      # 월→월
        assert state.days_held("없음", TODAY) == 0

    def test_sync_entries_adds_and_drops(self):
        state = TraderState(entry_dates={"OLD": "2026-01-01"})
        state.sync_entries({"NEW"}, TODAY)
        assert state.entry_dates == {"NEW": TODAY.isoformat()}


class FakeBroker:
    """주문을 기록만 하는 가짜 브로커. 현금 재조회 값을 주입한다."""

    def __init__(self, orderable: float = 0.0):
        self.sent: list[tuple] = []
        self.orderable = orderable

    def place_order(self, code, side, quantity, *, price=0.0,
                    order_type="market", dry_run=True):
        self.sent.append((code, side, quantity))
        return OrderResult(code, side, quantity, price, order_type, dry_run,
                           order_no="X")

    def fetch_deposit(self):
        return {"orderable": self.orderable}


class TestExecute:
    def _plan_with_orders(self):
        cfg = make_cfg()
        preds = make_preds([("AAA", -0.01, 0.02, 0.03), ("BBB", -0.01, 0.01, 0.03)])
        return build_plan(preds, account(1_000_000), {"AAA": 100_000, "BBB": 100_000},
                          cfg, state=TraderState(), today=TODAY)

    def test_buys_are_capped_by_cash(self):
        """계획이 5주여도 현금이 2주치면 2주만 나간다."""
        plan = self._plan_with_orders()
        plan.cash = 250_000                     # 100,000원짜리 2주까지
        broker = FakeBroker()
        results = execute_plan(broker, plan, dry_run=True)

        bought = sum(q for _, side, q in broker.sent if side == BUY)
        assert bought == 2
        assert any(r.error == "주문가능금액 부족" for r in results)

    def test_sells_go_before_buys(self):
        cfg = make_cfg()
        old = Holding("OLD", "구", quantity=10, sellable=10, avg_price=10_000,
                      current_price=10_000, eval_amount=100_000)
        preds = make_preds([("AAA", -0.01, 0.02, 0.03)])
        plan = build_plan(preds, account(100_000, [old]),
                          {"AAA": 10_000, "OLD": 10_000},
                          cfg, state=TraderState(), today=TODAY)
        broker = FakeBroker()
        execute_plan(broker, plan, dry_run=True)

        sides = [side for _, side, _ in broker.sent]
        assert sides.index(SELL) < sides.index(BUY)
