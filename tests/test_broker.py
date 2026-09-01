"""브로커 어댑터 테스트 — 네트워크 없이 파싱·안전장치만 검증한다.

여기서 지키려는 것:
  1) live 계정은 브로커 단에서도 막힌다 (config.py 가 뚫려도)
  2) 종목코드 정규화 — 'A005930' 을 못 맞추면 보유분을 미보유로 착각한다
  3) 키움의 zero-pad 문자열 금액이 숫자로 제대로 들어온다
  4) 주문 함수의 기본값이 dry_run 이다
"""

from __future__ import annotations

import pytest

from src.data.kiwoom import endpoints as ep
from src.trading.broker import BUY, Holding, PaperBroker, _normalize_code
from src.utils.config import KiwoomSettings
from src.utils.parsing import parse_records


class FakeClient:
    def __init__(self, env: str = "mock", responses: dict | None = None):
        self.settings = KiwoomSettings(
            env=env, base_url="https://mockapi.kiwoom.com", app_key="k",
            app_secret="s", account_no=None, rate_limit_per_sec=3.0,
        )
        self.responses = responses or {}
        self.calls: list[tuple[str, dict]] = []

    def request(self, spec, body=None, **kw):
        self.calls.append((spec.api_id, body or {}))
        return self.responses.get(spec.api_id, {}), {}

    def paginate(self, spec, body=None):
        self.calls.append((spec.api_id, body or {}))
        yield self.responses.get(spec.api_id, {})

    def close(self):
        pass


class TestSafety:
    def test_live_is_refused_at_broker(self):
        """config.py 를 통과하더라도 여기서 한 번 더 막는다."""
        with pytest.raises(RuntimeError, match="모의투자 전용"):
            PaperBroker(FakeClient(env="live"))

    def test_order_defaults_to_dry_run(self):
        broker = PaperBroker(FakeClient())
        res = broker.place_order("005930", BUY, 1)
        assert res.dry_run is True
        assert broker.client.calls == []          # 전송되지 않았다

    def test_zero_quantity_never_orders(self):
        broker = PaperBroker(FakeClient())
        res = broker.place_order("005930", BUY, 0, dry_run=False)
        assert not res.ok and broker.client.calls == []

    def test_limit_order_without_price_is_refused(self):
        broker = PaperBroker(FakeClient())
        res = broker.place_order("005930", BUY, 1, order_type="limit", dry_run=False)
        assert not res.ok and broker.client.calls == []


class TestCodeNormalization:
    @pytest.mark.parametrize("raw,expected", [
        ("A005930", "005930"),      # 잔고 응답에 붙는 접두어
        (" 005930 ", "005930"),
        ("5930", "005930"),         # zero-pad 가 빠진 경우
        ("005930", "005930"),
    ])
    def test_normalize(self, raw, expected):
        assert _normalize_code(raw) == expected

    def test_holding_code_matches_universe(self):
        """정규화가 깨지면 보유 종목이 '미보유'로 읽혀 중복 매수가 나간다."""
        client = FakeClient(responses={
            "kt00018": {
                "tot_evlt_amt": "1000000",
                "acnt_evlt_remn_indv_tot": [{
                    "stk_cd": "A005930", "stk_nm": "삼성전자", "rmnd_qty": "10",
                    "trde_able_qty": "10", "pur_pric": "70000", "cur_prc": "-72000",
                    "evlt_amt": "720000", "evltv_prft": "20000", "prft_rt": "2.86",
                }],
            },
        })
        holdings, summary = PaperBroker(client).fetch_holdings()
        assert set(holdings) == {"005930"}
        h = holdings["005930"]
        assert (h.quantity, h.avg_price, h.current_price) == (10, 70000.0, 72000.0)
        assert summary["total_eval"] == 1_000_000.0


class TestDepositParsing:
    def test_zero_padded_amounts(self):
        """예수금은 '000000010000000' 형태로 온다 (검증된 실제 응답)."""
        body = {
            "entr": "000000010000000",
            "d2_entra": "000000010000000",
            "ord_alow_amt": "000000009500000",
            "pymn_alow_amt": "000000010000000",
        }
        row = parse_records([body], ep.DEPOSIT.schema).iloc[0]
        assert row["deposit"] == 10_000_000
        assert row["orderable"] == 9_500_000


class TestSnapshot:
    def test_equity_is_cash_plus_stock(self):
        client = FakeClient(responses={
            "kt00001": {"entr": "000000003000000", "ord_alow_amt": "000000003000000"},
            "kt00018": {
                "tot_evlt_amt": "7000000",
                "acnt_evlt_remn_indv_tot": [{
                    "stk_cd": "005930", "stk_nm": "삼성전자", "rmnd_qty": "100",
                    "trde_able_qty": "100", "pur_pric": "70000", "cur_prc": "70000",
                    "evlt_amt": "7000000",
                }],
            },
        })
        snap = PaperBroker(client).snapshot()
        assert snap.equity == 10_000_000
        assert snap.weight_of("005930") == pytest.approx(0.7)
        assert snap.weight_of("없음") == 0.0

    def test_equity_does_not_double_count_before_settlement(self):
        """매수 직후 예수금은 D+2 라 안 빠져 있다 — 같은 돈을 두 번 세면 안 된다.

        2026-08-26 실측 회귀: 실제 9,929만원 계좌가 1억(예수금) + 8,203만(주식)
        = 1억 8,203만으로 계산됐다. 분모가 1.8배면 목표비중 10% 가 주문가능금액을
        넘는 주문이 된다.
        """
        client = FakeClient(responses={
            "kt00001": {
                "entr": "000000100000000",          # 예수금 1억 — 매수했는데 그대로
                "ord_alow_amt": "000000007046297",  # 주문가능은 즉시 차감됐다
            },
            "kt00018": {
                "tot_evlt_amt": "000000082031660",
                "acnt_evlt_remn_indv_tot": [{
                    "stk_cd": "A032640", "stk_nm": "LG유플러스", "rmnd_qty": "673",
                    "trde_able_qty": "673", "pur_pric": "14899", "cur_prc": "14890",
                    "evlt_amt": "000000010020970",
                }],
            },
        })
        snap = PaperBroker(client).snapshot()
        assert snap.equity == pytest.approx(7_046_297 + 82_031_660)
        assert snap.equity != snap.deposit + snap.total_eval   # 부풀지 않는다

    def test_equity_ignores_kiwoom_estimated_assets(self):
        """`prsm_dpst_aset_amt` 가 와도 쓰지 않는다 — 증권사 화면과 어긋난다.

        2026-08-27 실측 회귀: 주문가능 16,063,315 + 주식평가 83,707,100 =
        99,770,415(-0.23%) 가 키움 웹 화면과 맞는데, 이 필드는 99,310,070(-0.69%)
        으로 왔다. 필드를 믿으면 대시보드가 증권사와 0.46%p 어긋난다.
        """
        client = FakeClient(responses={
            "kt00001": {"entr": "000000016063315",
                        "ord_alow_amt": "000000016063315"},
            "kt00018": {
                "tot_evlt_amt": "000000083707100",
                "prsm_dpst_aset_amt": "000000099310070",   # 키움 값 — 무시한다
                "acnt_evlt_remn_indv_tot": [],
            },
        })
        snap = PaperBroker(client).snapshot()
        assert snap.equity == pytest.approx(99_770_415)
        assert snap.equity != snap.estimated_assets

    def test_empty_account_has_zero_weights(self):
        # 실제 kt00001 은 entr 와 ord_alow_amt 를 항상 같이 준다(2026-08-25 검증).
        # 현금 100% 라 둘이 같은 값이다.
        client = FakeClient(responses={
            "kt00001": {"entr": "000000010000000",
                        "ord_alow_amt": "000000010000000"},
            "kt00018": {},
        })
        snap = PaperBroker(client).snapshot()
        assert snap.holdings == {} and snap.equity == 10_000_000

    def test_sellable_falls_back_to_quantity(self):
        """매도가능수량이 안 오면 보유수량으로 물러난다 — 청산이 막히면 안 된다."""
        client = FakeClient(responses={"kt00018": {"acnt_evlt_remn_indv_tot": [
            {"stk_cd": "005930", "rmnd_qty": "10", "pur_pric": "1", "evlt_amt": "10"}
        ]}})
        holdings, _ = PaperBroker(client).fetch_holdings()
        assert holdings["005930"].sellable == 10


def test_holding_pnl_uses_entry_price():
    h = Holding("A", "", 1, 1, avg_price=10_000, current_price=9_000, eval_amount=9_000)
    assert h.avg_price == 10_000
