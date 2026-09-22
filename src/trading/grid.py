"""그리드 플로우 — 목표 비중을 **지정가 사다리**로 바꾸는 실행 층.

> 루트 [`CLAUDE.md`](../../CLAUDE.md)의 절대 규칙이 우선한다.

## 왜 signal.py 를 건드리지 않나 (절대 규칙 7)

절대 규칙 7의 핵심은 "신호는 한 곳"이다. 그리드는 **신호가 아니라 실행 방식**이다 —
*무엇을 얼마나 살지*는 `signal.generate_signals` 가 정하고, 이 모듈은 *그 목표에
어떻게 도달할지*만 답한다. 그래서 종목 선별은 여기 없다. `trading.direction.mode =
cross_sectional` 과 `top_n` 이 이미 q50 순위로 고르고 있고, 그리드는 그 결과를 받는다.

## 모델이 정하는 것 — 간격

`q90 - q10` 은 모델이 본 **불확실성 폭**이고, 그게 곧 변동성 예측이다.
폭이 넓다고 본 종목·시점에는 사다리를 넓게, 좁으면 촘촘하게 깐다.

    간격 = clip(폭 x alpha, 하한 = 왕복비용 x 2, 상한)

**하한이 있는 이유:** 왕복비용보다 좁은 칸은 체결될 때마다 손해다. 모델이 아무리
좁은 폭을 예측해도 이 바닥 아래로는 못 내려간다 (`signal.round_trip_cost` 재사용).

alpha 의 실측 근거는 `docs/REFERENCES.md` 7.5 절. 일봉 198종목 x 90사이클에서
alpha=0.20(평균 간격 3.1%)이 고정 간격 대비 test 고유분 +0.17 -> +0.30%p 였다.

## 사다리 모양 — ratio

`ratio=1.0` 이면 등간격, `0<ratio<1` 이면 중심에서 멀어질수록 칸이 촘촘해진다
(Yeh et al., arXiv:2211.12839 의 flexible grid). 실측에서는 등간격이 근소하게
나았으므로 기본값은 1.0 이고, 가변은 비교 실험용으로 열어둔다.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.trading.signal import QuantilePrediction, round_trip_cost

# 한국거래소 호가단위 (2023-01 개편). (미만가격, 호가단위)
# 지정가는 호가에 맞지 않으면 **주문이 거부된다** — 계산이 아니라 제약이다.
_TICKS: tuple[tuple[float, int], ...] = (
    (2_000, 1), (5_000, 5), (20_000, 10), (50_000, 50),
    (200_000, 100), (500_000, 500),
)
_TICK_ABOVE = 1_000


def tick_size(price: float) -> int:
    """그 가격대의 호가단위."""
    for limit, tick in _TICKS:
        if price < limit:
            return tick
    return _TICK_ABOVE


def round_to_tick(price: float, *, up: bool = False) -> int:
    """호가단위로 맞춘다. `up=True` 면 올림 — 매도 호가에 쓴다.

    매수는 내림, 매도는 올림으로 두어 **의도한 것보다 비싸게 사거나 싸게 파는 일**이
    없게 한다. 반올림이면 한 틱만큼 불리한 쪽으로 넘어갈 수 있다.
    """
    t = tick_size(price)
    q = price / t
    return int((-(-q // 1) if up else q // 1) * t)


def spacing_from_quantiles(
    pred: QuantilePrediction, costs: dict, grid_cfg: dict,
) -> float:
    """모델의 분위 폭 -> 칸 간격(비율).

    하한은 왕복비용의 배수로 강제한다. `floor_mult=2.0` 이면 왕복 한 번에
    비용만큼은 남아야 한다는 뜻이다.
    """
    alpha = float(grid_cfg.get("width_alpha", 0.20))
    floor = round_trip_cost(costs) * float(grid_cfg.get("floor_mult", 2.0))
    cap = float(grid_cfg.get("max_spacing", 0.10))
    return min(max(pred.interval_width * alpha, floor), cap)


def grid_offsets(n_levels: int, spacing: float, ratio: float = 1.0) -> list[float]:
    """중심에서의 누적 오프셋(비율). 길이 `n_levels`, 마지막이 전체 범위.

    ratio=1.0  등간격 — 칸마다 `spacing`
    0<ratio<1  중심에서 멀어질수록 촘촘 (첫 칸이 넓고 뒤로 갈수록 좁아진다)

    총 범위는 어느 쪽이든 `spacing x n_levels` 로 같다 — 그래야 모양만 비교된다.
    """
    if n_levels <= 0:
        raise ValueError(f"칸 수는 1 이상이어야 한다: {n_levels}")
    span = spacing * n_levels
    if abs(ratio - 1.0) < 1e-9:
        gaps = [span / n_levels] * n_levels
    elif 0.0 < ratio < 1.0:
        first = span * (1 - ratio) / (1 - ratio ** n_levels)
        gaps = [first * ratio ** i for i in range(n_levels)]
    else:
        raise ValueError(f"ratio 는 (0, 1] 이어야 한다: {ratio}")
    out, acc = [], 0.0
    for g in gaps:
        acc += g
        out.append(acc)
    return out


@dataclass(frozen=True)
class GridOrder:
    """사다리 한 칸에 걸 지정가 주문."""

    code: str
    side: str          # signal.BUY / SELL 과 같은 문자열을 쓴다
    price: int         # 호가단위에 맞춘 값
    quantity: int
    level: int         # 중심에서 몇 칸 떨어졌나 (1부터)


@dataclass(frozen=True)
class Ladder:
    """한 종목의 사다리 전체."""

    code: str
    center: float
    spacing: float
    budget: float
    per_level: float
    init_quantity: int          # 기준가에서 먼저 사는 수량 (예산 절반)
    buys: list[GridOrder]
    sells: list[GridOrder]
    skipped: str | None = None  # 사다리를 못 깐 이유. None 이면 정상


def build_ladder(
    pred: QuantilePrediction,
    price: float,
    budget: float,
    costs: dict,
    grid_cfg: dict,
) -> Ladder:
    """목표 예산과 모델 예측 -> 지정가 사다리. **API 를 호출하지 않는다.**

    순수 함수로 두는 이유는 `paper_trader.build_plan` 과 같다 — 백테스트와
    모의투자가 같은 함수를 쓰고, 주문 없이 판단만 테스트할 수 있어야 한다.

    price: 사다리 중심. 호출자가 **그 시점까지의** 값을 넘긴다(종가 또는 VWAP).
    budget: 이 종목에 배정된 금액. 절반은 기준가 매수, 절반은 아래 칸에 나뉜다.
    """
    n = int(grid_cfg.get("levels", 5))
    ratio = float(grid_cfg.get("ratio", 1.0))
    spacing = spacing_from_quantiles(pred, costs, grid_cfg)
    per_level = (budget / 2) / n

    def empty(reason: str) -> Ladder:
        return Ladder(pred.code, price, spacing, budget, per_level, 0, [], [], reason)

    if price <= 0:
        return empty("현재가 없음")
    # 칸당 예산으로 1주도 못 사면 사다리가 성립하지 않는다 (고가주)
    if per_level < price:
        return empty(f"칸당 예산 {per_level:,.0f}원 < 1주 {price:,.0f}원")

    offsets = grid_offsets(n, spacing, ratio)
    buys, sells = [], []
    for i, off in enumerate(offsets, start=1):
        bp = round_to_tick(price * (1 - off))            # 매수는 내림
        sp = round_to_tick(price * (1 + off), up=True)   # 매도는 올림
        if bp > 0 and (q := int(per_level // bp)) > 0:
            buys.append(GridOrder(pred.code, "buy", bp, q, i))
        if sp > 0 and (q := int(per_level // sp)) > 0:
            sells.append(GridOrder(pred.code, "sell", sp, q, i))

    init_qty = int((budget / 2) // price)
    return Ladder(pred.code, price, spacing, budget, per_level, init_qty, buys, sells)


def paired_sell_price(ladder: Ladder, level: int) -> int:
    """`level` 번 매수 칸이 체결됐을 때 **한 칸 위**에 걸 매도 가격.

    `price x (1 + spacing)` 이 아니라 실제 윗칸을 쓴다 — 가변 사다리(ratio<1)에서는
    칸마다 간격이 달라 그 근사가 어긋나고, 등간격에서도 호가 반올림 때문에 미세하게 틀린다.
    """
    if level <= 1:
        return round_to_tick(ladder.center, up=True)
    for o in ladder.buys:
        if o.level == level - 1:
            return o.price
    raise ValueError(f"{ladder.code}: {level - 1}번 매수 칸이 없다")
