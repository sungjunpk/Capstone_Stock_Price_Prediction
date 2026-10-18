"""장중 회차의 사다리 이탈 재개 — 체결될 칸이 0 인 채로 하루가 죽는 것을 막는다.

가격이 사다리 밖으로 나가면 그 종목은 다음 아침까지 아무 칸도 안 걸린다.
조용히 멈추는 실패라 테스트로 잠근다. 폭(`spacing`)은 장중에 안 바뀐다는 것도
같이 확인한다 — 입력이 일봉 예측이라 바뀔 수가 없다.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import paper_trade as pt  # noqa: E402
from src.trading.broker import AccountSnapshot, Holding, OrderResult  # noqa: E402
from src.trading.grid import realized_vol, relay_spacing  # noqa: E402

TODAY = date(2026, 10, 1)
CODE = "005930"
CFG = {
    "grid": {"levels": 5, "ratio": 1.0, "width_alpha": 0.20, "floor_mult": 2.0,
             "max_spacing": 0.10, "center_k": 0.0, "stop_loss_pct": -0.15},
    "trading": {"costs": {"commission_bps": 1.88, "tax_bps": 20.0, "slippage_bps": 5.0}},
}


class FakeBroker:
    def __init__(self, price: float, closes: list[float] | None = None):
        self.price = price
        self.closes = closes or []
        self.cancelled: list[str] = []
        self.placed: list[tuple[str, int, int]] = []   # (side, price, qty)
        self._no = 0

    def fetch_unfilled_detail(self):
        return [{"code": CODE, "order_no": "A1", "unfilled_qty": 10,
                 "filled_qty": 0, "side": "buy", "name": "삼성전자"}]

    def fetch_prices(self, codes):
        return {c: self.price for c in codes}

    def fetch_recent_closes(self, code, bars, tic="10"):
        return self.closes[-bars:]

    def cancel_order(self, code, order_no, quantity=0, *, dry_run=True):
        self.cancelled.append(order_no)
        return OrderResult(code, "cancel", quantity, 0.0, "cancel", dry_run)

    def place_order(self, code, side, quantity, *, price=0.0,
                    order_type="limit", dry_run=True):
        self._no += 1
        self.placed.append((side, int(price), quantity))
        return OrderResult(code, side, quantity, price, order_type, dry_run,
                           order_no=f"N{self._no}")


def setup(price: float, closes=None, spacing: float = 0.024, cfg=CFG):
    """기준가 70,000원 · 폭 2.4%(q90−q10=12%) 사다리가 걸려 있는 상태."""
    recent = pd.DataFrame([{"date": pd.Timestamp("2026-09-30"), "code": CODE,
                            "q10": -0.06, "q50": 0.0, "q90": 0.06}])
    broker = FakeBroker(price, closes)
    account = AccountSnapshot(cash=0.0, deposit=0.0, holdings={
        CODE: Holding(CODE, "삼성전자", 100, 100, 70_000, price, 100 * price),
    })
    state = pt.TraderState(grids={CODE: {
        "center": 70_000.0, "spacing": spacing, "date": "2026-09-29",
        "spacing_open": 0.024, "vol_open": 0.02,
        "budget": 10_000_000.0, "stopped": False,
        "rungs": {"A1": {"side": "buy", "level": 1, "price": 68_300, "qty": 10, "filled": 0}},
    }})
    state.save = lambda *a, **k: None          # 상태 파일을 건드리지 않는다
    return recent, broker, account, state


def run(price: float, closes=None, spacing: float = 0.024, cfg=CFG):
    recent, broker, account, state = setup(price, closes, spacing, cfg)
    rc = pt._grid_intraday(cfg, recent, broker, account, state, TODAY, dry_run=False)
    return rc, broker, state.grids[CODE]


# 변동성 거의 0 → 간격이 바닥(왕복비용×2)까지 좁아진다
FLAT = [70_000 + (i % 2) for i in range(39)]
RELAY_CFG = CFG | {"grid": CFG["grid"] | {
    "intraday_relay": {"enabled": True, "band": 0.20, "vol_bars": 39},
}}
VOL_FLAT = realized_vol(FLAT)


class TestOutsideLadderIntraday:
    def test_아래로_이탈하면_현재가로_사이클을_다시_연다(self):
        _, broker, cyc = run(60_000)
        assert broker.cancelled == ["A1"]              # 죽은 칸을 거둔다
        assert cyc["center"] == 60_000                 # 기준가가 현재가로 내려왔다
        assert "A1" not in cyc["rungs"]                # 거둔 칸이 체결로 오인되면 안 된다
        assert broker.placed, "새 사다리를 걸어야 한다"

    def test_위로_이탈해도_같다(self):
        _, broker, cyc = run(80_000)
        assert broker.cancelled == ["A1"]
        assert cyc["center"] == 80_000

    def test_범위_안이면_아무것도_거두지_않는다(self):
        _, broker, cyc = run(70_500)
        assert broker.cancelled == []
        assert cyc["center"] == 70_000.0               # 기준가는 사이클 안에서 안 움직인다
        assert "A1" in cyc["rungs"]

    def test_장중에는_폭이_안_바뀐다(self):
        """같은 일봉 예측에서 나오므로 재개해도 간격은 아침과 같다."""
        _, _, cyc = run(60_000)
        assert round(cyc["spacing"], 6) == 0.024

    def test_장중_재개는_시장가를_안_낸다(self):
        """지정가만 — 아래로 이탈한 경우 예산은 이미 사다리에 들어가 있다."""
        _, broker, _ = run(60_000)
        assert broker.placed and all(p > 0 for _, p, _ in broker.placed)


class TestIntradayRelay:
    """장중 간격 재조정 — 폭을 **모델이 아니라 가격**이 바꾼다. 설정으로만 켜진다."""

    def test_안_켜면_분봉을_안_부른다(self):
        _, broker, cyc = run(70_500, FLAT)
        assert broker.cancelled == [] and cyc["spacing"] == 0.024

    def test_밴드_밖이면_기준가는_두고_사다리만_다시_건다(self):
        _, broker, cyc = run(70_500, FLAT, cfg=RELAY_CFG)
        assert broker.cancelled == ["A1"]
        assert cyc["center"] == 70_000.0                   # 기준가는 안 옮긴다
        assert cyc["spacing"] < 0.024 and "A1" not in cyc["rungs"]
        assert broker.placed

    def test_밴드_안이면_그대로_둔다(self):
        sp = relay_spacing(0.024, 0.02, VOL_FLAT, RELAY_CFG["trading"]["costs"], RELAY_CFG["grid"])
        _, broker, cyc = run(70_500, FLAT, spacing=sp, cfg=RELAY_CFG)
        assert broker.cancelled == [] and cyc["spacing"] == sp

    def test_종가가_모자라면_아무것도_안_한다(self):
        """못 재는 것과 '좁아졌다'는 다르다 — 분봉 조회가 실패해도 사다리를 안 건드린다."""
        _, broker, cyc = run(70_500, [], cfg=RELAY_CFG)
        assert broker.cancelled == [] and cyc["spacing"] == 0.024

    def test_이탈이_간격보다_먼저다(self):
        """둘 다 해당하면 기준가까지 다시 잡는 쪽을 쓴다 — 체결될 칸이 0인 게 더 급하다."""
        _, broker, cyc = run(60_000, FLAT, cfg=RELAY_CFG)
        assert cyc["center"] == 60_000
