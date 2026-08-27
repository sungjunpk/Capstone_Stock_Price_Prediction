"""보유 비중 기록 테스트 — 발표용 시계열이 믿을 만한가.

여기서 지키려는 것:
  1) 비중 합이 1 인가 (Σw + c = 1) — 현금을 잔차로 잡아야 성립한다
  2) 현금 비중이 **주문가능금액이 아니라 잔차**인가 (D+2 결제 때문에 다르다)
  3) 같은 날 두 번 기록하면 그날 행이 통째로 교체되는가
     — 종목 단위 upsert 면 그 사이 매도된 종목의 행이 남아 합이 1 을 넘는다
"""

from __future__ import annotations

from datetime import date

import pytest

from src.trading import record
from src.trading.broker import AccountSnapshot, Holding

TODAY = date(2026, 8, 27)


def make_account(holdings: dict[str, float], *, equity: float,
                 orderable: float) -> AccountSnapshot:
    """holdings: 종목코드 → 평가금액."""
    hs = {
        code: Holding(code, f"이름{code}", quantity=10, sellable=10,
                      avg_price=amount / 10, current_price=amount / 10,
                      eval_amount=amount)
        for code, amount in holdings.items()
    }
    return AccountSnapshot(
        cash=orderable,
        deposit=orderable,
        holdings=hs,
        total_eval=sum(holdings.values()),
        estimated_assets=equity,
    )


@pytest.fixture(autouse=True)
def tmp_records(tmp_path, monkeypatch):
    """실제 기록 파일을 건드리지 않는다."""
    monkeypatch.setattr(record, "HOLDINGS_PATH", tmp_path / "holdings.jsonl")


class TestWeightConvention:
    def test_weights_sum_to_one(self):
        """Σw + c = 1 — Boyd et al.(2024) §2.1 규약."""
        acct = make_account({"A": 60_000_000.0, "B": 30_000_000.0},
                            equity=100_000_000.0, orderable=9_672_457.0)
        rows = record.holdings_rows(acct, TODAY)

        assert sum(r["weight"] for r in rows) == pytest.approx(1.0, abs=1e-6)

    def test_cash_is_residual_not_orderable(self):
        """현금은 총자산 - 주식평가합. 주문가능금액을 쓰면 합이 1 을 넘는다."""
        acct = make_account({"A": 90_494_910.0},
                            equity=99_669_687.0, orderable=9_672_457.0)
        rows = record.holdings_rows(acct, TODAY)
        cash = next(r for r in rows if r["code"] == record.CASH_CODE)

        assert cash["eval_amount"] == pytest.approx(99_669_687.0 - 90_494_910.0)
        assert cash["eval_amount"] != pytest.approx(9_672_457.0)
        assert sum(r["weight"] for r in rows) == pytest.approx(1.0, abs=1e-6)

    def test_weight_is_share_of_total_assets(self):
        acct = make_account({"A": 10_000_000.0},
                            equity=100_000_000.0, orderable=90_000_000.0)
        rows = record.holdings_rows(acct, TODAY)
        a = next(r for r in rows if r["code"] == "A")

        assert a["weight"] == pytest.approx(0.10)

    def test_zero_equity_is_skipped_not_crashed(self):
        acct = make_account({}, equity=0.0, orderable=0.0)
        assert record.holdings_rows(acct, TODAY) == []


class TestRerunSameDay:
    def test_sold_holding_does_not_linger(self):
        """오전에 2종목, 오후에 1종목으로 다시 기록 → 판 종목 행이 남으면 안 된다."""
        morning = make_account({"A": 50_000_000.0, "B": 40_000_000.0},
                               equity=100_000_000.0, orderable=10_000_000.0)
        record.record_holdings(morning, TODAY)

        afternoon = make_account({"A": 50_000_000.0},
                                 equity=100_000_000.0, orderable=50_000_000.0)
        record.record_holdings(afternoon, TODAY)

        rows = record.read_jsonl(record.HOLDINGS_PATH)
        codes = {r["code"] for r in rows}

        assert "B" not in codes
        assert sum(r["weight"] for r in rows) == pytest.approx(1.0, abs=1e-6)

    def test_other_days_are_kept(self):
        acct = make_account({"A": 90_000_000.0},
                            equity=100_000_000.0, orderable=10_000_000.0)
        record.record_holdings(acct, date(2026, 8, 26))
        record.record_holdings(acct, TODAY)

        rows = record.read_jsonl(record.HOLDINGS_PATH)
        assert {r["date"] for r in rows} == {"2026-08-26", "2026-08-27"}


# ------------------------------------------------- 전량 청산 (전략 교체용)
def test_liquidate_all_sells_every_holding_and_buys_nothing():
    """전량 청산은 예측을 보지 않고 보유분만 0으로 만든다."""
    import pandas as pd

    from src.trading.broker import SELL, AccountSnapshot, Holding
    from src.trading.paper_trader import TraderState, build_plan

    cfg = {
        "features": {"return_horizon": 5},
        "backtest": {"rebalance_days": 10},
        "trading": {
            "abstain": {"max_interval_width": 0.05},
            "direction": {"mode": "cross_sectional", "top_n": 2, "exit_rank": 4,
                          "min_candidates": 1, "long_threshold": 0.004,
                          "short_threshold": -0.004},
            "sizing": {"method": "rank_normalized", "exposure_scaling": False,
                       "max_position_pct": 0.5, "min_trade_weight": 0.01},
            "risk": {"stop_loss_pct": -0.05, "take_profit_pct": 0.10,
                     "max_trades_per_day": 10, "max_gross_exposure": 1.0},
            "costs": {"commission_bps": 1.5, "tax_bps": 18.0, "slippage_bps": 5.0},
        },
    }
    d = date(2026, 8, 27)
    preds = pd.DataFrame({"code": ["A", "B", "C"], "date": [d] * 3,
                          "q10": [-0.01] * 3, "q50": [0.05, 0.04, 0.03],
                          "q90": [0.02] * 3})
    hold = {c: Holding(code=c, name=c, quantity=10, sellable=10,
                       avg_price=50_000, current_price=50_000,
                       eval_amount=500_000)
            for c in ("A", "Z")}
    account = AccountSnapshot(cash=0.0, deposit=0.0, holdings=hold,
                              total_eval=1_000_000, estimated_assets=1_000_000)
    prices = {"A": 50_000, "B": 50_000, "C": 50_000, "Z": 50_000}
    plan = build_plan(preds, account, prices, cfg, state=TraderState(),
                      today=d, liquidate_all=True)

    # A 는 예측 1위지만 판다 — 예측을 보지 않는다는 뜻이다
    assert {o.code for o in plan.orders} == {"A", "Z"}
    assert all(o.side == SELL for o in plan.orders)
