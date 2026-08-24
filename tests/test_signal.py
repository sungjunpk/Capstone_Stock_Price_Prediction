"""매매 신호 규칙 — 기권 로직이 이 프로젝트의 핵심 차별점이라 반드시 지킨다."""

import pytest

from src.trading.signal import Action, QuantilePrediction, generate_signal, generate_signals

CFG = {
    "abstain": {"max_interval_width": 0.05},
    "direction": {"long_threshold": 0.004, "short_threshold": -0.004},
    "sizing": {"method": "inverse_width", "kelly_fraction": 0.25, "max_position_pct": 0.10},
    "risk": {"max_gross_exposure": 0.9},
    "costs": {"commission_bps": 1.5, "tax_bps": 18.0, "slippage_bps": 5.0},
}


def test_wide_interval_abstains_even_with_strong_signal():
    p = QuantilePrediction("005930", q10=-0.05, q50=0.08, q90=0.12)  # 폭 0.17
    s = generate_signal(p, CFG)
    assert s.action is Action.ABSTAIN
    assert s.target_weight == 0.0


def test_narrow_interval_buys():
    p = QuantilePrediction("005930", q10=0.005, q50=0.02, q90=0.035)
    s = generate_signal(p, CFG)
    assert s.action is Action.BUY
    assert 0 < s.target_weight <= CFG["sizing"]["max_position_pct"]


def test_threshold_never_below_transaction_cost():
    # 왕복비용 ≈ 0.0031. q50 이 그 아래면 진입 금지
    p = QuantilePrediction("005930", q10=-0.001, q50=0.002, q90=0.005)
    assert generate_signal(p, CFG).action is Action.HOLD


def test_tighter_interval_gets_bigger_position():
    tight = generate_signal(QuantilePrediction("A", 0.015, 0.02, 0.025), CFG)
    loose = generate_signal(QuantilePrediction("B", -0.005, 0.02, 0.045), CFG)
    assert tight.target_weight > loose.target_weight


def test_crossed_quantiles_rejected():
    with pytest.raises(ValueError, match="분위 교차"):
        QuantilePrediction("005930", q10=0.05, q50=0.01, q90=0.09)


def test_gross_exposure_capped():
    preds = [QuantilePrediction(f"{i:06d}", 0.015, 0.02, 0.025) for i in range(20)]
    total = sum(s.target_weight for s in generate_signals(preds, CFG))
    assert total <= CFG["risk"]["max_gross_exposure"] + 1e-9
