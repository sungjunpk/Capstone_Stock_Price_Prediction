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


def test_percentile_threshold_adapts_to_prediction_spread():
    """절대 임계값을 추측하면 기권률이 0% 나 100% 로 튄다.

    실측: 추측값 0.05 로 기권률 95.8%, 거래 0건이 나왔다
    (5일 수익률의 자연 폭은 0.124).
    """
    from src.trading.signal import resolve_abstain_threshold

    widths = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]

    th = resolve_abstain_threshold(widths, {"percentile": 30})
    narrow = sum(w <= th for w in widths)
    assert 2 <= narrow <= 4, f"30분위인데 {narrow}/10 만 통과"

    # 절대값 설정이면 그대로 쓴다
    assert resolve_abstain_threshold(widths, {"max_interval_width": 0.07}) == 0.07


def test_percentile_threshold_makes_trades_happen():
    """백분위 방식이면 예측이 아무리 넓어도 상위 N% 는 반드시 거래된다."""
    from src.trading.signal import resolve_abstain_threshold

    cfg = {**CFG, "abstain": {"percentile": 30}}
    preds = [
        QuantilePrediction(f"{i:06d}", 0.02 - w / 2, 0.02, 0.02 + w / 2)
        for i, w in enumerate([0.06, 0.09, 0.12, 0.18, 0.25, 0.30, 0.35, 0.40, 0.5, 0.6])
    ]
    widths = [p.interval_width for p in preds]
    th = resolve_abstain_threshold(widths, cfg["abstain"])

    sigs = generate_signals(preds, cfg, max_width=th)
    traded = [s for s in sigs if s.action is Action.BUY]
    assert traded, "백분위 방식인데 거래가 하나도 없다"
