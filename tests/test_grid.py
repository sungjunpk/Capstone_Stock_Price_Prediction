"""그리드 실행 층 테스트 — 호가단위·간격 하한·예산·짝 매도.

조용히 틀리면 주문이 거부되거나(호가) 체결될 때마다 손해가 되는(간격 하한)
자리들이라 전부 잠근다.
"""

import pytest

from src.trading.grid import (
    build_ladder,
    grid_offsets,
    paired_sell_price,
    round_to_tick,
    spacing_from_quantiles,
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


class TestPairedSell:
    def test_한_칸_위_레벨을_쓴다(self):
        """px x (1+spacing) 근사가 아니라 실제 윗칸이어야 한다 — 가변에서 어긋난다."""
        cfg = CFG | {"ratio": 0.6}
        lad = build_ladder(pred(0.12), 70_000, 10_000_000, COSTS, cfg)
        assert paired_sell_price(lad, 3) == next(o.price for o in lad.buys if o.level == 2)

    def test_첫_칸의_짝은_기준가다(self):
        lad = build_ladder(pred(0.12), 70_000, 10_000_000, COSTS, CFG)
        assert paired_sell_price(lad, 1) == round_to_tick(70_000, up=True)
