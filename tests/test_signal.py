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


# ------------------------------------------------------------ 횡단면 순위 모드
#
# 절대 임계값에서 순위로 바꾼 이유는 하나다: **공통 수준 편차에 안 흔들리게 하려고.**
# 그 성질을 test_level_shift_does_not_change_selection 이 지킨다.

XS_CFG = {
    "abstain": {"max_interval_width": 0.05, "percentile": 30},
    "direction": {
        "mode": "cross_sectional", "top_n": 5, "min_candidates": 10,
        "long_threshold": 0.004, "short_threshold": -0.004,
    },
    "sizing": {"method": "rank_normalized", "exposure_scaling": False,
               "max_position_pct": 0.10},
    "risk": {"max_gross_exposure": 0.90},
    "costs": {"commission_bps": 1.5, "tax_bps": 18.0, "slippage_bps": 5.0},
}


def _narrow(code: str, q50: float, width: float = 0.02) -> QuantilePrediction:
    """폭을 직접 지정한 예측. q10/q90 을 q50 중심으로 대칭 배치한다."""
    return QuantilePrediction(code, q50 - width / 2, q50, q50 + width / 2)


def _universe(n: int = 30, width: float = 0.02) -> list[QuantilePrediction]:
    # q50 을 -0.01 ~ +0.01 로 고르게 깔아 순위가 명확하게 갈리도록 한다
    return [_narrow(f"{i:03d}", -0.01 + 0.02 * i / (n - 1), width) for i in range(n)]


def _chosen(sigs) -> set[str]:
    return {s.code for s in sigs if s.action is Action.BUY}


def test_cross_sectional_picks_exactly_top_n():
    sigs = generate_signals(_universe(30), XS_CFG, max_width=0.05)
    buys = [s for s in sigs if s.action is Action.BUY]
    assert len(buys) == XS_CFG["direction"]["top_n"]
    # q50 이 가장 큰 5개여야 한다 (029 가 최상위)
    assert _chosen(sigs) == {"029", "028", "027", "026", "025"}


def test_level_shift_does_not_change_selection():
    """모든 q50 에 같은 값을 더해도 선택 종목이 같아야 한다.

    이 성질 하나 때문에 절대 임계값을 버렸다. 실측에서 모델의 예측 수준이
    실제보다 약 1%p 낮았는데, 절대 임계값 방식은 그 편차를 그대로 얻어맞아
    거래가 사실상 멈췄다.
    """
    base = _universe(30)
    shifted = [QuantilePrediction(p.code, p.q10 + 0.05, p.q50 + 0.05, p.q90 + 0.05)
               for p in base]

    assert _chosen(generate_signals(base, XS_CFG, max_width=0.05)) == \
           _chosen(generate_signals(shifted, XS_CFG, max_width=0.05))


def test_level_shift_would_break_absolute_mode():
    """대조: absolute 모드는 같은 이동에 결과가 뒤집힌다 — 순위 방식의 존재 이유."""
    cfg = {**XS_CFG, "direction": {**XS_CFG["direction"], "mode": "absolute"}}
    base = _universe(30)
    down = [QuantilePrediction(p.code, p.q10 - 0.02, p.q50 - 0.02, p.q90 - 0.02)
            for p in base]

    assert len(_chosen(generate_signals(base, cfg, max_width=0.05))) > 0
    assert len(_chosen(generate_signals(down, cfg, max_width=0.05))) == 0


def test_abstain_still_applied_before_ranking():
    """폭이 넓으면 q50 이 1등이어도 기권한다 — 판단 순서가 안 바뀌었는지 확인."""
    preds = _universe(30)
    preds[-1] = _narrow("029", 0.05, width=0.20)      # q50 최고, 그러나 폭이 넓다

    sigs = {s.code: s for s in generate_signals(preds, XS_CFG, max_width=0.05)}
    assert sigs["029"].action is Action.ABSTAIN
    assert "029" not in _chosen(sigs.values())


def test_too_few_candidates_all_abstain():
    preds = _universe(30)
    # 25개를 넓은 폭으로 만들어 생존자를 5개(< min_candidates 10)로 줄인다
    preds = [_narrow(p.code, p.q50, width=0.20) if i < 25 else p
             for i, p in enumerate(preds)]

    sigs = generate_signals(preds, XS_CFG, max_width=0.05)
    assert all(s.action is Action.ABSTAIN for s in sigs)
    assert sum(s.target_weight for s in sigs) == 0.0


def test_cross_sectional_respects_gross_and_position_caps():
    sigs = generate_signals(_universe(30), XS_CFG, max_width=0.05)
    assert sum(s.target_weight for s in sigs) <= XS_CFG["risk"]["max_gross_exposure"] + 1e-9
    assert all(s.target_weight <= XS_CFG["sizing"]["max_position_pct"] + 1e-9 for s in sigs)


def test_normalized_sizing_actually_invests():
    """inverse_width 의 실패 지점: 상위 종목을 골라도 총 노출이 미미했다."""
    # 폭이 임계값 바로 아래라 conf 가 0 에 가까운 상황
    preds = [_narrow(f"{i:03d}", -0.01 + 0.02 * i / 29, width=0.0495) for i in range(30)]
    sigs = generate_signals(preds, XS_CFG, max_width=0.05)

    gross = sum(s.target_weight for s in sigs)
    assert gross > 0.4, f"정규화가 안 먹었다 (gross={gross:.3f})"


def test_exposure_scales_with_survivor_count():
    """생존 후보가 줄면 총 노출도 줄어야 한다 — 기권이 '얼마나 쉬는가'에도 반영."""
    # top_n 은 10으로 둔다. 5로 두면 5 x 종목상한 10% = 0.50 이 노출 상한 0.90 보다
    # 낮아서, 스케일링이 아니라 종목상한이 총 노출을 결정해버려 검증이 무의미해진다.
    cfg = {
        **XS_CFG,
        "direction": {**XS_CFG["direction"], "top_n": 10},
        "sizing": {**XS_CFG["sizing"], "exposure_scaling": True},
    }

    # 유니버스 100개 중 30개 생존(=percentile 30, 정상) vs 15개 생존(불확실)
    def gross_with(n_survivors: int) -> float:
        preds = [_narrow(f"{i:03d}", -0.01 + 0.02 * i / 99,
                         width=0.02 if i >= 100 - n_survivors else 0.20)
                 for i in range(100)]
        return sum(s.target_weight for s in generate_signals(preds, cfg, max_width=0.05))

    full, half = gross_with(30), gross_with(15)
    assert full > half > 0
    assert half == pytest.approx(full / 2, rel=0.15)


def test_unknown_mode_is_rejected():
    cfg = {**XS_CFG, "direction": {**XS_CFG["direction"], "mode": "몰라요"}}
    with pytest.raises(ValueError, match="direction.mode"):
        generate_signals(_universe(30), cfg, max_width=0.05)


# ------------------------------------------------------------ 이력(hysteresis) 버퍼
#
# 살 때는 top_n, 팔 때는 exit_rank. 11등이 된 종목을 파는 건 정보가 아니라
# 노이즈에 반응하는 것이고, 그 노이즈에 연 7% 의 거래비용을 냈다.

BUF_CFG = {
    **XS_CFG,
    "direction": {**XS_CFG["direction"], "top_n": 5, "exit_rank": 10},
}


def test_held_name_survives_outside_top_n():
    """8등으로 밀린 보유 종목이 exit_rank 10 안이면 계속 들고 간다."""
    preds = _universe(30)                      # q50 오름차순 → 029 가 1등
    held = {"022"}                             # 1등부터 세면 8등

    chosen = _chosen(generate_signals(preds, BUF_CFG, max_width=0.05, held=held))
    assert "022" in chosen
    assert len(chosen) == 5


def test_held_name_sold_beyond_exit_rank():
    """exit_rank 밖(15등)이면 보유 중이어도 판다."""
    held = {"015"}                             # 1등부터 세면 15등
    chosen = _chosen(generate_signals(_universe(30), BUF_CFG, max_width=0.05, held=held))
    assert "015" not in chosen


def test_buffer_never_exceeds_top_n():
    """보유가 exit_rank 를 가득 채워도 선택은 top_n 이하다."""
    held = {f"{i:03d}" for i in range(20, 30)}     # 상위 10개 전부 보유
    sigs = generate_signals(_universe(30), BUF_CFG, max_width=0.05, held=held)
    assert len(_chosen(sigs)) == BUF_CFG["direction"]["top_n"]


def test_buffer_prefers_incumbents_over_newcomers():
    """자리가 모자라면 보유분이 우선이다 — 그래야 회전율이 준다."""
    held = {"026", "025", "024"}               # 4,5,6등
    chosen = _chosen(generate_signals(_universe(30), BUF_CFG, max_width=0.05, held=held))
    assert {"026", "025", "024"} <= chosen


def test_abstain_beats_buffer():
    """보유 중이어도 신뢰구간이 넓어지면 판다 — 기권이 항상 먼저다."""
    preds = _universe(30)
    preds[22] = _narrow("022", preds[22].q50, width=0.20)      # 폭이 넓어졌다

    sigs = {s.code: s for s in
            generate_signals(preds, BUF_CFG, max_width=0.05, held={"022"})}
    assert sigs["022"].action is Action.ABSTAIN


def test_no_held_means_old_behavior():
    """held 를 안 주면 버퍼 이전과 똑같이 상위 top_n 을 고른다."""
    preds = _universe(30)
    assert _chosen(generate_signals(preds, BUF_CFG, max_width=0.05)) == \
           _chosen(generate_signals(preds, BUF_CFG, max_width=0.05, held=set()))


def test_exit_rank_equal_to_top_n_disables_buffer():
    cfg = {**XS_CFG, "direction": {**XS_CFG["direction"], "top_n": 5, "exit_rank": 5}}
    preds = _universe(30)
    assert _chosen(generate_signals(preds, cfg, max_width=0.05, held={"022"})) == \
           _chosen(generate_signals(preds, cfg, max_width=0.05))


def test_buffer_reduces_turnover():
    """순위가 매 회차 흔들릴 때, 버퍼가 있으면 교체가 확실히 줄어든다."""
    import random

    def rotations(cfg) -> int:
        rnd = random.Random(0)
        held, swaps = set(), 0
        for _ in range(60):
            # 진짜 순위 + 노이즈 — 경계 근처 종목들이 매번 자리를 바꾼다
            preds = [_narrow(f"{i:03d}", -0.01 + 0.02*i/29 + rnd.gauss(0, 0.002))
                     for i in range(30)]
            new = _chosen(generate_signals(preds, cfg, max_width=0.05, held=held))
            swaps += len(new - held)
            held = new
        return swaps

    no_buffer = {**XS_CFG, "direction": {**XS_CFG["direction"], "top_n": 5, "exit_rank": 5}}
    assert rotations(BUF_CFG) < rotations(no_buffer)


# --------------------------------------------------------- 시총가중 · 기권 기준
# 목표가 "코스피200 초과수익"이 되면서 들어온 두 축이다 (2026-09-08).
# 지수가 시총가중이라 동일·확신도 가중으로는 구조적으로 못 이긴다 — 그런데 비중
# 계산이 조용히 틀려도 성과가 조금 나빠질 뿐 에러가 안 난다. 그래서 테스트로 못박는다.

XS_CAP = {
    "abstain": {"percentile": 50},
    "direction": {"mode": "cross_sectional", "top_n": 3, "exit_rank": 3,
                  "min_candidates": 0, "long_threshold": 0.004,
                  "short_threshold": -0.004},
    "sizing": {"method": "cap_weighted", "cap_alpha": 1.0, "conf_beta": 0.0,
               "max_position_pct": 1.0, "exposure_scaling": False},
    "risk": {"max_gross_exposure": 0.90},
    "costs": {"commission_bps": 1.5, "tax_bps": 18.0, "slippage_bps": 5.0},
}


def _capped(code: str, q50: float, mcap: float, vol: float | None = None):
    return QuantilePrediction(code, q50 - 0.01, q50, q50 + 0.01, mcap=mcap, vol=vol)


def test_cap_weighted_gives_bigger_weight_to_bigger_cap():
    preds = [_capped("A", 0.03, 900.0), _capped("B", 0.02, 300.0),
             _capped("C", 0.01, 100.0)]
    w = {s.code: s.target_weight for s in generate_signals(preds, XS_CAP, max_width=0.05)}
    assert w["A"] > w["B"] > w["C"] > 0
    # 몫이 시총 비율(9:3:1)을 그대로 따라야 한다
    assert w["A"] / w["C"] == pytest.approx(9.0, rel=1e-6)


def test_cap_alpha_zero_falls_back_to_confidence_weighting():
    cfg = {**XS_CAP, "sizing": {**XS_CAP["sizing"], "cap_alpha": 0.0, "conf_beta": 1.0}}
    preds = [_capped("A", 0.03, 900.0), _capped("B", 0.02, 300.0),
             _capped("C", 0.01, 100.0)]
    w = {s.code: s.target_weight for s in generate_signals(preds, cfg, max_width=0.05)}
    assert w["A"] == pytest.approx(w["C"])      # 폭이 같으니 확신도도 같다


def test_missing_cap_uses_median_not_zero():
    """시총 미상 종목이 비중 0 이 되거나 독차지하면 안 된다 — 척도가 섞이는 사고다."""
    preds = [_capped("A", 0.03, 100.0), _capped("B", 0.02, None),
             _capped("C", 0.01, 100.0)]
    w = {s.code: s.target_weight for s in generate_signals(preds, XS_CAP, max_width=0.05)}
    assert w["B"] > 0
    assert w["B"] == pytest.approx(w["A"])      # 중앙값(=100)으로 채운다


def test_width_over_vol_lets_high_vol_names_survive():
    """절대 폭 기준이면 고변동 종목이 항상 배제된다 — 그게 지수 승자를 걸러냈다."""
    wide_but_volatile = QuantilePrediction("HI", -0.05, 0.02, 0.05, mcap=100.0, vol=0.05)
    narrow_calm = QuantilePrediction("LO", 0.005, 0.01, 0.025, mcap=100.0, vol=0.005)

    by_width = {"abstain": {"basis": "width"}}
    by_vol = {"abstain": {"basis": "width_over_vol"}}
    from src.trading.signal import abstain_score

    assert abstain_score(wide_but_volatile, "width") > abstain_score(narrow_calm, "width")
    # 변동성으로 나누면 순서가 뒤집힌다 — 고변동주가 자기 기준에서는 좁다
    assert (abstain_score(wide_but_volatile, "width_over_vol")
            < abstain_score(narrow_calm, "width_over_vol"))
    assert by_width and by_vol      # 설정 키 이름이 바뀌면 위 basis 문자열도 바뀌어야 한다


def test_unknown_abstain_basis_is_rejected():
    from src.trading.signal import abstain_basis

    with pytest.raises(ValueError):
        abstain_basis({"basis": "제멋대로"})


def test_prediction_from_row_carries_context_and_survives_nan():
    """백테스트와 모의투자가 이 함수 하나로 예측을 만든다 (절대 규칙 7)."""
    import pandas as pd

    from src.trading.signal import prediction_from_row

    df = pd.DataFrame([
        {"code": "A", "q10": 0.0, "q50": 0.01, "q90": 0.02, "mcap": 500.0, "rvol_20": 0.03},
        {"code": "B", "q10": 0.0, "q50": 0.01, "q90": 0.02, "mcap": float("nan"),
         "rvol_20": float("nan")},
    ])
    a, b = (prediction_from_row(r) for r in df.itertuples())
    assert (a.mcap, a.vol) == (500.0, 0.03)
    assert b.mcap is None and b.vol is None     # NaN 은 '모름'이지 0 이 아니다


def test_mcap_pool_cuts_to_the_largest_before_abstain():
    """시총 풀은 판단이 아니라 **유니버스 정의**라 기권보다 먼저 걸려야 한다.

    벤치마크가 시총가중 지수라, 중소형주를 아무리 잘 골라도 못 따라간다 —
    실측(2026-09-10): 유니버스 전부를 동일가중으로 사도 베타 0.49 였다.
    """
    cfg = {**XS_CAP, "direction": {**XS_CAP["direction"], "mcap_pool": 2, "top_n": 2,
                                   "exit_rank": 2}}
    preds = [_capped("BIG1", 0.01, 900.0), _capped("BIG2", 0.02, 800.0),
             _capped("SMALL", 0.09, 10.0)]          # q50 은 SMALL 이 압도적으로 높다
    sigs = {s.code: s for s in generate_signals(preds, cfg, max_width=0.05)}
    assert sigs["BIG1"].action is Action.BUY
    assert sigs["BIG2"].action is Action.BUY
    # 풀 밖이면 q50 이 아무리 높아도 안 산다
    assert sigs["SMALL"].action is not Action.BUY


def test_mcap_pool_keeps_unknown_cap_names():
    """시총을 '모른다'를 '작다'로 읽으면 안 된다 — 신규상장이 조용히 배제된다."""
    cfg = {**XS_CAP, "direction": {**XS_CAP["direction"], "mcap_pool": 1, "top_n": 2,
                                   "exit_rank": 2}}
    preds = [_capped("BIG", 0.01, 900.0), _capped("UNKNOWN", 0.05, None)]
    sigs = {s.code: s for s in generate_signals(preds, cfg, max_width=0.05)}
    assert sigs["UNKNOWN"].action is Action.BUY


def test_mcap_pool_none_keeps_everyone():
    """기본값(null)에서는 아무도 안 자른다 — 기존 동작이 그대로여야 한다."""
    preds = [_capped("A", 0.03, 900.0), _capped("B", 0.02, 1.0)]
    sigs = generate_signals(preds, XS_CAP, max_width=0.05)
    assert sum(1 for s in sigs if s.action is Action.BUY) == 2
