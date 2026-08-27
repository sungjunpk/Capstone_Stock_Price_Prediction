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
