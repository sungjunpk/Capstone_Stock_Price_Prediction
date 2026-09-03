"""리스크 오버레이 — 매매 판단 4단계.

총 익스포저 계산은 조용히 틀리기 쉽다. 신호가 원한 비중이 그대로 나오는지,
막아야 할 때만 막는지를 양쪽에서 잡는다.
"""

import pandas as pd
import pytest

from src.trading.risk import (
    Position,
    apply_risk_overlay,
    market_cvar,
    portfolio_cvar,
)
from src.trading.signal import Action, Signal

CFG = {
    "sizing": {"max_position_pct": 0.10},
    "risk": {
        "max_gross_exposure": 0.90,
        "max_trades_per_day": 20,
        "stop_loss_pct": -0.05,
        "take_profit_pct": 0.10,
    },
}


def _buys(n: int, w: float = 0.09) -> list[Signal]:
    return [Signal(f"NEW{i}", Action.BUY, w, 0.5, "") for i in range(n)]


def _held(n: int, w: float = 0.09, entry: float = 100.0) -> dict[str, Position]:
    return {f"OLD{i}": Position(f"OLD{i}", w, entry, 5) for i in range(n)}


def _prices(*groups, px: float = 100.0) -> dict[str, float]:
    return {c: px for g in groups for c in g}


def _gross(decision) -> float:
    return sum(s.target_weight for s in decision.signals)


def test_rotation_does_not_shrink_new_positions():
    """보유 9종목을 전부 다른 9종목으로 교체 — 신규 비중이 깎이면 안 된다.

    회귀 테스트다. 곧 청산될 보유분을 '보유 중'으로 세는 바람에
    목표 0.81 이 0.09 로 잘린 적이 있다(9배 축소). 매 회차 종목을 갈아타는
    횡단면 순위 방식에서는 전략이 사실상 현금만 들고 있게 된다.
    """
    positions, sigs = _held(9), _buys(9)
    d = apply_risk_overlay(sigs, positions, _prices(positions, [s.code for s in sigs]), CFG)
    assert _gross(d) == pytest.approx(0.81)


def test_held_counts_when_caller_keeps_unsignaled():
    """반대로 호출자가 보유분을 유지한다면 그건 세어야 한다."""
    positions, sigs = _held(9), _buys(9)
    d = apply_risk_overlay(
        sigs, positions, _prices(positions, [s.code for s in sigs]), CFG,
        liquidate_unsignaled=False,
    )
    assert _gross(d) == pytest.approx(0.09)      # 0.90 - 0.81 만 남는다


def test_gross_cap_still_binds():
    """상한 자체는 살아 있어야 한다 — 12종목 x 9% = 1.08 > 0.90."""
    sigs = _buys(12)
    d = apply_risk_overlay(sigs, {}, _prices([s.code for s in sigs]), CFG)
    assert _gross(d) == pytest.approx(0.90)


def test_position_cap_applied():
    sigs = [Signal("A", Action.BUY, 0.50, 0.9, "")]
    d = apply_risk_overlay(sigs, {}, {"A": 100.0}, CFG)
    assert d.signals[0].target_weight == pytest.approx(0.10)


def test_stop_loss_forces_exit():
    positions = {"A": Position("A", 0.09, 100.0, 3)}
    d = apply_risk_overlay([], positions, {"A": 94.0}, CFG)      # -6% < -5%
    assert d.forced_exits == ["A"]
    assert "손절" in d.blocked["A"]


def test_take_profit_forces_exit():
    positions = {"A": Position("A", 0.09, 100.0, 3)}
    d = apply_risk_overlay([], positions, {"A": 111.0}, CFG)     # +11% > +10%
    assert d.forced_exits == ["A"]
    assert "익절" in d.blocked["A"]


def test_trade_cap_keeps_highest_confidence():
    cfg = {**CFG, "risk": {**CFG["risk"], "max_trades_per_day": 3}}
    sigs = [Signal(f"S{i}", Action.BUY, 0.05, i / 10, "") for i in range(10)]
    d = apply_risk_overlay(sigs, {}, _prices([s.code for s in sigs]), cfg)

    entered = {s.code for s in d.signals if s.target_weight > 0}
    assert entered == {"S9", "S8", "S7"}


def test_short_signal_becomes_exit_when_shorting_disallowed():
    positions = {"A": Position("A", 0.09, 100.0, 3)}
    sigs = [Signal("A", Action.SELL, 0.09, 0.5, ""), Signal("B", Action.SELL, 0.09, 0.5, "")]
    d = apply_risk_overlay(sigs, positions, {"A": 100.0, "B": 100.0}, CFG)

    assert [s.code for s in d.signals] == ["A"]
    assert d.signals[0].target_weight == 0.0     # 보유분은 청산
    assert "공매도 불가" in d.blocked["B"]        # 미보유분은 무시


def test_abstain_and_hold_never_reach_orders():
    sigs = [Signal("A", Action.ABSTAIN, 0.0, 0.0, ""), Signal("B", Action.HOLD, 0.0, 0.3, "")]
    d = apply_risk_overlay(sigs, {}, {"A": 100.0, "B": 100.0}, CFG)
    assert d.signals == []


# ------------------------------------------------------------ 차단 사유 집계
#
# 발생 지점에서 센다. 문자열을 파싱하면 메시지 문구를 바꾸는 순간 조용히 깨진다.


def test_reasons_counted_by_category():
    positions = {
        "LOSS": Position("LOSS", 0.09, 100.0, 3),
        "GAIN": Position("GAIN", 0.09, 100.0, 3),
    }
    sigs = [Signal("NOHOLD", Action.SELL, 0.09, 0.5, "")]
    prices = {"LOSS": 94.0, "GAIN": 111.0, "NOHOLD": 100.0}

    d = apply_risk_overlay(sigs, positions, prices, CFG)
    assert d.blocked_by_reason == {"손절": 1, "익절": 1, "공매도불가": 1}


def test_trade_cap_counted():
    cfg = {**CFG, "risk": {**CFG["risk"], "max_trades_per_day": 3}}
    sigs = [Signal(f"S{i}", Action.BUY, 0.05, i / 10, "") for i in range(10)]
    d = apply_risk_overlay(sigs, {}, _prices([s.code for s in sigs]), cfg)
    assert d.blocked_by_reason == {"거래한도": 7}


def test_reasons_empty_when_nothing_blocked():
    sigs = _buys(3)
    d = apply_risk_overlay(sigs, {}, _prices([s.code for s in sigs]), CFG)
    assert d.blocked_by_reason == {}


def test_reason_counts_match_blocked_dict():
    """카테고리 합계는 항상 blocked 항목 수와 같아야 한다."""
    cfg = {**CFG, "risk": {**CFG["risk"], "max_trades_per_day": 2}}
    positions = {"LOSS": Position("LOSS", 0.09, 100.0, 3)}
    sigs = _buys(6) + [Signal("X", Action.SELL, 0.09, 0.5, "")]
    prices = _prices(positions, [s.code for s in sigs])
    prices["LOSS"] = 90.0

    d = apply_risk_overlay(sigs, positions, prices, cfg)
    assert sum(d.blocked_by_reason.values()) == len(d.blocked)


class TestMaxHolding:
    """지평 만료 청산 — 타점 탐지 트랙에서만 켠다."""

    def _cfg(self, **risk):
        return {
            "sizing": {"max_position_pct": 0.1},
            "risk": {"stop_loss_pct": -0.05, "take_profit_pct": 0.10, **risk},
        }

    def test_disabled_by_default_keeps_daily_track_behaviour(self):
        pos = {"005930": Position("005930", 0.1, 100.0, days_held=999)}
        out = apply_risk_overlay([], pos, {"005930": 100.0}, self._cfg())
        assert out.forced_exits == []

    def test_expired_position_is_liquidated(self):
        pos = {"005930": Position("005930", 0.1, 100.0, days_held=7)}
        out = apply_risk_overlay([], pos, {"005930": 100.0},
                                 self._cfg(max_holding_bars=7))
        assert out.forced_exits == ["005930"]
        assert out.blocked_by_reason == {"보유만료": 1}

    def test_stop_loss_wins_over_expiry_for_reason_label(self):
        """같은 청산이라도 사유는 손익이 더 정보량이 많다."""
        pos = {"005930": Position("005930", 0.1, 100.0, days_held=99)}
        out = apply_risk_overlay([], pos, {"005930": 90.0},
                                 self._cfg(max_holding_bars=7))
        assert out.forced_exits == ["005930"]
        assert "손절" in out.blocked_by_reason

    def test_expiry_fires_even_without_a_price(self):
        pos = {"005930": Position("005930", 0.1, 100.0, days_held=10)}
        out = apply_risk_overlay([], pos, {}, self._cfg(max_holding_bars=7))
        assert out.forced_exits == ["005930"]


class TestPortfolioCVaR:
    """꼬리손실 측정 — 이 값이 틀리면 6단계 축소가 통째로 틀린다."""

    @staticmethod
    def _returns(**series) -> pd.DataFrame:
        return pd.DataFrame(series, index=pd.RangeIndex(len(next(iter(series.values())))))

    @staticmethod
    def _bad_then_flat(n_bad: int = 10, n_flat: int = 190) -> list[float]:
        """최악 5% 가 정확히 -10% 인 200일. 손으로 계산되는 표본이다."""
        return [-0.10] * n_bad + [0.01] * n_flat

    def test_exact_value_on_a_hand_computable_series(self):
        r = self._returns(A=self._bad_then_flat())
        assert portfolio_cvar({"A": 1.0}, r) == pytest.approx(0.10)

    def test_positively_homogeneous(self):
        """CVaR(2w) == 2·CVaR(w). 닫힌 형태 축소(scale = 한도/CVaR)가 딛고 선 성질이다."""
        r = self._returns(A=self._bad_then_flat())
        assert portfolio_cvar({"A": 0.4}, r) == pytest.approx(
            2 * portfolio_cvar({"A": 0.2}, r)
        )

    def test_correlated_book_is_riskier_than_diversified_one(self):
        """이 게이트의 존재 이유 — 같은 개별 변동성이라도 같이 무너지면 꼬리가 두껍다."""
        bad = self._bad_then_flat()
        together = self._returns(A=bad, B=bad)                    # 완전 상관
        apart = self._returns(A=bad, B=bad[10:20] + bad[:10] + bad[20:])  # 나쁜 날이 어긋난다
        w = {"A": 0.5, "B": 0.5}
        assert portfolio_cvar(w, together) == pytest.approx(0.10)
        assert portfolio_cvar(w, apart) < portfolio_cvar(w, together)

    def test_returns_none_when_history_is_too_short(self):
        """꼬리를 못 재는데 한도를 걸면 표본 부족이 곧 축소가 된다 — 모르면 개입하지 않는다."""
        r = self._returns(A=[0.01] * 50)
        assert portfolio_cvar({"A": 1.0}, r) is None


class TestCVaRGate:
    """리스크 오버레이 6단계. **기본은 꺼져 있어야 한다.**"""

    RETURNS = pd.DataFrame({"A": [-0.10] * 10 + [0.01] * 190})

    def _cfg(self, **risk):
        return {
            "sizing": {"max_position_pct": 0.10},
            "risk": {"max_gross_exposure": 1.00, "max_trades_per_day": 20,
                     "stop_loss_pct": -0.05, "take_profit_pct": 0.10, **risk},
        }

    def _run(self, cfg, returns):
        return apply_risk_overlay(
            [Signal("A", Action.BUY, 0.10, 0.5, "")], {}, {"A": 100.0}, cfg,
            returns=returns,
        )

    @pytest.mark.parametrize("cfg_kw,returns", [
        ({}, RETURNS),                      # 한도가 없다 → 측정도 안 한다
        ({"cvar_limit": 0.005}, None),      # 수익률이 없다 → 잴 수가 없다
    ])
    def test_inactive_paths_leave_orders_untouched(self, cfg_kw, returns):
        """회귀 가드. A 단계에서 실거래 주문이 한 건도 바뀌면 안 된다."""
        out = self._run(self._cfg(**cfg_kw), returns)
        assert out.signals[0].target_weight == pytest.approx(0.10)
        assert out.cvar is None and out.cvar_scale == 1.0
        assert "CVaR한도" not in out.blocked_by_reason

    def test_limit_scales_the_book_to_exactly_the_limit(self):
        # 비중 0.10 x 꼬리 -0.10 → CVaR 0.010. 한도 0.005 면 절반으로 줄어야 한다.
        out = self._run(self._cfg(cvar_limit=0.005), self.RETURNS)
        assert out.cvar == pytest.approx(0.010)
        assert out.cvar_scale == pytest.approx(0.5)
        assert out.blocked_by_reason["CVaR한도"] == 1
        after = {s.code: s.target_weight for s in out.signals}
        assert portfolio_cvar(after, self.RETURNS) == pytest.approx(0.005)

    def test_limit_above_the_book_does_nothing(self):
        """0% 도 100% 도 아니어야 게이트다 — 넉넉한 한도에서는 안 물어야 한다."""
        out = self._run(self._cfg(cvar_limit=0.05), self.RETURNS)
        assert out.cvar == pytest.approx(0.010)
        assert out.cvar_scale == 1.0
        assert out.signals[0].target_weight == pytest.approx(0.10)


class TestMarketCVaR:
    """상대 한도의 기준자 — 전 종목 균등배분의 꼬리."""

    def test_matches_equal_weight_portfolio_on_a_complete_matrix(self):
        r = pd.DataFrame({"A": [-0.10] * 10 + [0.01] * 190,
                          "B": [0.01] * 10 + [-0.10] * 10 + [0.01] * 180})
        assert market_cvar(r) == pytest.approx(
            portfolio_cvar({"A": 0.5, "B": 0.5}, r)
        )

    def test_survives_gaps_that_would_wipe_the_joint_sample(self):
        """유니버스 전체에 dropna(how="any") 를 걸면 종목 하나가 비는 날마다
        행이 통째로 날아간다. 146종목이면 남는 행이 거의 없다 — 그래서 날짜별 평균을 쓴다."""
        r = pd.DataFrame({"A": [-0.10] * 10 + [0.01] * 190,
                          "B": [float("nan")] * 190 + [0.01] * 10})
        assert portfolio_cvar({"A": 0.5, "B": 0.5}, r) is None   # 완전한 행이 10개뿐
        assert market_cvar(r) == pytest.approx(0.10)             # A 만으로도 잰다


class TestRelativeCVaRLimit:
    """시장 대비 상대 한도 — 절대 한도가 국면 간에 이식되지 않아 나온 재설계."""

    # A 는 0~9일, B 는 10~19일에 -10%. 시장(균등배분)의 꼬리는 -4.5% 로 얇아진다.
    RETURNS = pd.DataFrame({"A": [-0.10] * 10 + [0.01] * 190,
                            "B": [0.01] * 10 + [-0.10] * 10 + [0.01] * 180})

    def _run(self, **risk):
        cfg = {"sizing": {"max_position_pct": 0.10},
               "risk": {"max_gross_exposure": 1.00, "max_trades_per_day": 20,
                        "stop_loss_pct": -0.05, "take_profit_pct": 0.10, **risk}}
        return apply_risk_overlay(
            [Signal("A", Action.BUY, 0.10, 0.5, "")], {}, {"A": 100.0}, cfg,
            returns=self.RETURNS,
        )

    def test_limit_is_resolved_against_the_market_tail(self):
        # 우리 책: 0.10 x A → 꼬리 0.010. 시장 균등배분 꼬리 0.045.
        # k=0.1 이면 한도 0.0045 → 절반 아래로 줄어야 한다.
        out = self._run(cvar_limit_ratio=0.1)
        assert out.cvar_market == pytest.approx(0.045)
        assert out.cvar == pytest.approx(0.010)
        assert out.cvar_scale == pytest.approx(0.45)
        after = {s.code: s.target_weight for s in out.signals}
        assert portfolio_cvar(after, self.RETURNS) == pytest.approx(0.1 * 0.045)

    def test_generous_ratio_does_not_bind(self):
        out = self._run(cvar_limit_ratio=1.0)
        assert out.cvar_scale == 1.0
        assert out.signals[0].target_weight == pytest.approx(0.10)

    def test_zero_limit_is_the_strictest_not_the_gate_being_off(self):
        """0.0 은 falsy다. truthy 로 검사하면 '가장 엄격한 한도'가 '꺼짐'으로 뒤집힌다."""
        out = self._run(cvar_limit=0.0)
        assert out.cvar_scale == 0.0
        assert out.signals[0].target_weight == 0.0
        assert out.blocked_by_reason["CVaR한도"] == 1
