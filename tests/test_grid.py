"""그리드 실행 층 테스트 — 호가단위·간격 하한·예산·짝 매도.

조용히 틀리면 주문이 거부되거나(호가) 체결될 때마다 손해가 되는(간격 하한)
자리들이라 전부 잠근다.
"""

import pytest

from src.trading.grid import (
    build_ladder,
    center_from_quantiles,
    grid_offsets,
    outside_ladder,
    paired_sell_price,
    round_to_tick,
    should_relay,
    spacing_from_quantiles,
    spans_from_quantiles,
    tick_size,
)
from src.trading.signal import QuantilePrediction, round_trip_cost

COSTS = {"commission_bps": 1.88, "tax_bps": 20.0, "slippage_bps": 5.0}
CFG = {"levels": 5, "ratio": 1.0, "width_alpha": 0.20, "floor_mult": 2.0, "max_spacing": 0.10}


def pred(width: float, code: str = "005930") -> QuantilePrediction:
    return QuantilePrediction(code=code, q10=-width / 2, q50=0.0, q90=width / 2)


class TestTick:
    @pytest.mark.parametrize("price,expected", [
        (1_999, 1), (2_000, 5), (4_999, 5), (5_000, 10), (19_999, 10),
        (20_000, 50), (49_999, 50), (50_000, 100), (199_999, 100),
        (200_000, 500), (499_999, 500), (500_000, 1_000), (3_000_000, 1_000),
    ])
    def test_거래소_호가단위(self, price, expected):
        assert tick_size(price) == expected

    def test_매수는_내림_매도는_올림(self):
        # 71,234 원대의 호가단위는 100 원. 반올림이면 71,200 으로 같아지지만
        # 매도는 올려야 **의도보다 싸게 파는 일**이 없다.
        assert round_to_tick(71_234) == 71_200
        assert round_to_tick(71_234, up=True) == 71_300

    def test_결과는_항상_호가의_배수(self):
        for p in (1_234, 7_777, 33_333, 123_456, 777_777):
            for up in (False, True):
                r = round_to_tick(p, up=up)
                assert r % tick_size(r) == 0, f"{p} -> {r}"


class TestSpacing:
    def test_모델_폭에_비례한다(self):
        narrow = spacing_from_quantiles(pred(0.05), COSTS, CFG)
        wide = spacing_from_quantiles(pred(0.20), COSTS, CFG)
        assert wide > narrow

    def test_왕복비용_아래로는_안_내려간다(self):
        """폭이 0 이어도 하한이 물어야 한다 — 비용보다 좁은 칸은 구조적 손해다."""
        floor = round_trip_cost(COSTS) * CFG["floor_mult"]
        assert spacing_from_quantiles(pred(0.0), COSTS, CFG) == pytest.approx(floor)

    def test_상한이_있다(self):
        assert spacing_from_quantiles(pred(5.0), COSTS, CFG) == CFG["max_spacing"]


class TestCenter:
    """조건 02 — 사다리 중심을 모델의 q50 이 옮긴다."""

    def at(self, q50: float) -> QuantilePrediction:
        return QuantilePrediction(code="005930", q10=q50 - 0.06, q50=q50, q90=q50 + 0.06)

    def test_기본은_현재가다(self):
        assert center_from_quantiles(self.at(0.03), 70_000, CFG) == 70_000

    def test_오를_것으로_보면_위로_올린다(self):
        cfg = CFG | {"center_k": 1.0}
        assert center_from_quantiles(self.at(0.02), 70_000, cfg) == pytest.approx(71_400)

    def test_내릴_것으로_보면_아래로_내린다(self):
        cfg = CFG | {"center_k": 1.0}
        assert center_from_quantiles(self.at(-0.02), 70_000, cfg) == pytest.approx(68_600)

    def test_상한이_사다리를_붙잡는다(self):
        """예측이 튀는 날 사다리가 통째로 떨어지면 한쪽이 전부 즉시 체결된다."""
        cfg = CFG | {"center_k": 5.0, "center_cap": 0.05}
        assert center_from_quantiles(self.at(0.30), 70_000, cfg) == pytest.approx(73_500)

    def test_초기매수는_현재가로_센다(self):
        """중심은 사다리를 놓는 자리일 뿐 — 시장가 매수는 현재가에 나간다."""
        cfg = CFG | {"center_k": 1.0}
        lad = build_ladder(self.at(0.02), 70_000, 10_000_000, COSTS, cfg)
        assert lad.init_quantity == int(5_000_000 // 70_000)
        assert lad.center == pytest.approx(71_400)


class TestSkew:
    """모델의 분위 비대칭 -> 사다리 위아래 배분."""

    def pred_skew(self, lo: float, hi: float) -> QuantilePrediction:
        return QuantilePrediction(code="005930", q10=-lo, q50=0.0, q90=hi)

    def test_기본은_대칭이다(self):
        up, dn = spans_from_quantiles(pred(0.12), COSTS, CFG)
        assert up == pytest.approx(dn)

    def test_위쪽_꼬리가_길면_매도칸이_멀어진다(self):
        cfg = CFG | {"skew": "quantile"}
        up, dn = spans_from_quantiles(self.pred_skew(0.04, 0.12), COSTS, cfg)
        assert up > dn

    def test_아래쪽_꼬리가_길면_매수칸이_멀어진다(self):
        cfg = CFG | {"skew": "quantile"}
        up, dn = spans_from_quantiles(self.pred_skew(0.12, 0.04), COSTS, cfg)
        assert dn > up

    def test_합은_대칭일_때와_같다(self):
        """전체 범위는 그대로 두고 배분만 바꾼다 — 그래야 비대칭 효과만 비교된다."""
        p = self.pred_skew(0.04, 0.12)
        sym = sum(spans_from_quantiles(p, COSTS, CFG))
        skew = sum(spans_from_quantiles(p, COSTS, CFG | {"skew": "quantile"}))
        assert skew == pytest.approx(sym)

    def test_한쪽이_눌려도_비용_하한은_지킨다(self):
        cfg = CFG | {"skew": "quantile"}
        up, dn = spans_from_quantiles(self.pred_skew(0.001, 0.30), COSTS, cfg)
        floor = round_trip_cost(COSTS) * CFG["floor_mult"] * CFG["levels"]
        assert dn >= floor

    def test_사다리에_실제로_반영된다(self):
        cfg = CFG | {"skew": "quantile"}
        p = self.pred_skew(0.04, 0.12)
        lad = build_ladder(p, 70_000, 10_000_000, COSTS, cfg)
        up_gap = lad.sells[-1].price - 70_000
        dn_gap = 70_000 - lad.buys[-1].price
        assert up_gap > dn_gap


class TestOffsets:
    def test_등간격은_균등하다(self):
        off = grid_offsets(5, 0.015, 1.0)
        gaps = [off[0]] + [off[i] - off[i - 1] for i in range(1, 5)]
        assert all(g == pytest.approx(0.015) for g in gaps)

    def test_가변은_바깥으로_갈수록_촘촘하다(self):
        off = grid_offsets(5, 0.015, 0.6)
        gaps = [off[0]] + [off[i] - off[i - 1] for i in range(1, 5)]
        assert all(gaps[i] > gaps[i + 1] for i in range(4))

    def test_총_범위는_모양과_무관하게_같다(self):
        """그래야 등간격 vs 가변이 '분포' 차이만으로 비교된다."""
        assert grid_offsets(5, 0.015, 1.0)[-1] == pytest.approx(0.075)
        assert grid_offsets(5, 0.015, 0.6)[-1] == pytest.approx(0.075)

    @pytest.mark.parametrize("bad", [0.0, -0.5, 1.5])
    def test_ratio_범위를_지킨다(self, bad):
        with pytest.raises(ValueError):
            grid_offsets(5, 0.015, bad)


class TestLadder:
    def test_예산을_넘지_않는다(self):
        lad = build_ladder(pred(0.12), 70_000, 10_000_000, COSTS, CFG)
        spent = sum(o.price * o.quantity for o in lad.buys)
        assert spent + lad.init_quantity * 70_000 <= 10_000_000

    def test_모든_지정가가_호가에_맞는다(self):
        lad = build_ladder(pred(0.12), 71_234, 10_000_000, COSTS, CFG)
        for o in lad.buys + lad.sells:
            assert o.price % tick_size(o.price) == 0

    def test_매수는_아래_매도는_위(self):
        lad = build_ladder(pred(0.12), 70_000, 10_000_000, COSTS, CFG)
        assert all(o.price < 70_000 for o in lad.buys)
        assert all(o.price > 70_000 for o in lad.sells)

    def test_고가주는_사다리를_안_깐다(self):
        """칸당 예산 100만원인데 1주가 144만원이면 성립하지 않는다(삼성전기 사례)."""
        lad = build_ladder(pred(0.12), 1_440_000, 10_000_000, COSTS, CFG)
        assert lad.skipped is not None
        assert lad.buys == [] and lad.sells == []

    def test_현재가가_없으면_건너뛴다(self):
        assert build_ladder(pred(0.12), 0, 10_000_000, COSTS, CFG).skipped is not None


class TestRelay:
    """사이클 중간의 폭 재조정 — 매일 새 예측으로 다시 계산한다."""

    def test_밴드_안이면_그냥_둔다(self):
        assert not should_relay(0.030, 0.032, CFG)      # +6.7%

    def test_밴드를_넘으면_다시_깐다(self):
        assert should_relay(0.030, 0.035, CFG)          # +16.7%
        assert should_relay(0.030, 0.026, CFG)          # −13.3%

    def test_처음이면_무조건_깐다(self):
        assert should_relay(0.0, 0.030, CFG)

    def test_밴드는_설정이_정한다(self):
        assert should_relay(0.030, 0.032, CFG | {"relay_band": 0.02})


class TestOutsideLadder:
    """사이클 중간에 기준가를 다시 잡는 유일한 조건 — 사다리 이탈."""

    def test_범위_안이면_기준가를_안_옮긴다(self):
        lad = build_ladder(pred(0.12), 70_000, 10_000_000, COSTS, CFG)
        top, bot = max(o.price for o in lad.sells), min(o.price for o in lad.buys)
        assert not outside_ladder(lad, 70_000)
        assert not outside_ladder(lad, top) and not outside_ladder(lad, bot)

    def test_위아래로_벗어나면_다시_잡는다(self):
        lad = build_ladder(pred(0.12), 70_000, 10_000_000, COSTS, CFG)
        top, bot = max(o.price for o in lad.sells), min(o.price for o in lad.buys)
        assert outside_ladder(lad, top + 100) and outside_ladder(lad, bot - 100)


class TestPairedSell:
    def test_한_칸_위_레벨을_쓴다(self):
        """px x (1+spacing) 근사가 아니라 실제 윗칸이어야 한다 — 가변에서 어긋난다."""
        cfg = CFG | {"ratio": 0.6}
        lad = build_ladder(pred(0.12), 70_000, 10_000_000, COSTS, cfg)
        assert paired_sell_price(lad, 3) == next(o.price for o in lad.buys if o.level == 2)

    def test_첫_칸의_짝은_기준가다(self):
        lad = build_ladder(pred(0.12), 70_000, 10_000_000, COSTS, CFG)
        assert paired_sell_price(lad, 1) == round_to_tick(70_000, up=True)

    def test_매입가_아래로는_안_내려간다(self):
        """폭이 좁아지면 윗칸이 매입가 밑으로 온다 — 그 자리는 확정 손해다."""
        narrow = build_ladder(pred(0.02), 70_000, 10_000_000, COSTS, CFG)
        fill = 69_000                      # 넓던 시절 −2칸에서 체결된 가격
        p = paired_sell_price(narrow, 3, fill_price=fill, costs=COSTS)
        assert p >= fill * (1 + round_trip_cost(COSTS))

    def test_바닥이_필요없으면_윗칸_그대로다(self):
        """칸 간격이 왕복비용보다 넓으면(하한이 강제한다) 바닥은 안 문다."""
        lad = build_ladder(pred(0.12), 70_000, 10_000_000, COSTS, CFG)
        fill = next(o.price for o in lad.buys if o.level == 3)     # 3번 칸은 3번 가격에 체결된다
        upper = next(o.price for o in lad.buys if o.level == 2)
        assert paired_sell_price(lad, 3, fill_price=fill, costs=COSTS) == upper
