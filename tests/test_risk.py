"""리스크 오버레이 — 매매 판단 4단계.

총 익스포저 계산은 조용히 틀리기 쉽다. 신호가 원한 비중이 그대로 나오는지,
막아야 할 때만 막는지를 양쪽에서 잡는다.
"""

import pytest

from src.trading.risk import Position, apply_risk_overlay
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
