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

## 모델이 정하는 것 2 — 위아래 비대칭 (skew)

`skew="quantile"` 이면 **사다리가 모델이 본 분포 모양을 따라간다.**
`q90-q50` 이 `q50-q10` 보다 크면 위쪽 꼬리가 길다는 뜻이고, 그만큼 매도 칸을 위로 민다.
두 쪽 합은 대칭일 때와 같다 — 전체 범위는 그대로고 **배분만** 바뀐다.

⚠️ **기본값은 `none` 이다 — 재보니 효과가 없었다.** 일봉 198종목에서 대칭과
사이클마다 짝지어 비교했을 때 5거래일 test 는 절반도 못 이겼다(+0.008%p, p=0.47).
모델의 분위가 거의 대칭이라(평균 기울기 +0.083) 사다리가 실질적으로 안 기운다.
표는 `docs/REFERENCES.md` 7.6.

## 모델이 정하는 것 3 — 사이클 중간의 폭 재조정

폭을 사이클 시작에 한 번 정하고 끝내면, 그 사이 시장이 바뀌어도 사다리가 안 따라간다.
모델은 **매일** 새 예측을 내므로 매일 폭을 다시 계산하고, 변화가 `relay_band` 를
넘을 때만 **미체결 칸을 취소하고 다시 건다**. 기준가는 안 옮긴다 — 옮기면 그리드가
아니라 매일 재시작이다.

실측(일봉 198종목, 5거래일 사이클, 사이클마다 짝지어 비교): 고정 대비
val **+0.049%p (p=0.000)** · test **+0.037%p (p=0.001)**. `docs/REFERENCES.md` 7.7.

## 사다리 모양 — ratio

`ratio=1.0` 이면 등간격, `0<ratio<1` 이면 중심에서 멀어질수록 칸이 촘촘해진다
(Yeh et al., arXiv:2211.12839 의 flexible grid). 실측에서는 등간격이 근소하게
나았으므로 실측 기준 최적은 1.0 이지만, 기본값은 사용자 결정으로 0.6(가변)이다.
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


def spans_from_quantiles(
    pred: QuantilePrediction, costs: dict, grid_cfg: dict,
) -> tuple[float, float]:
    """(위쪽 총범위, 아래쪽 총범위). `skew` 가 정한다.

    `skew="none"`      위아래 같다 — 폭만 모델이 정한다
    `skew="quantile"`  **모델이 본 분포 모양을 사다리가 따라간다.**
        q90-q50 이 q50-q10 보다 크면 위쪽 꼬리가 길다는 뜻이고,
        그만큼 매도 칸을 위로 민다. 두 쪽 합은 대칭일 때와 같다 —
        전체 범위는 그대로 두고 **배분만** 바꾼다.

    하한은 **쪽마다** 건다. 한쪽이 극단적으로 눌려 칸이 비용보다 좁아지면
    그쪽 체결은 볼 때마다 손해이기 때문이다.
    """
    n = int(grid_cfg.get("levels", 5))
    base = spacing_from_quantiles(pred, costs, grid_cfg) * n   # 대칭일 때 한쪽 범위
    if str(grid_cfg.get("skew", "none")) != "quantile":
        return base, base

    up_raw = max(pred.q90 - pred.q50, 0.0)
    dn_raw = max(pred.q50 - pred.q10, 0.0)
    total = up_raw + dn_raw
    if total <= 0:
        return base, base
    # 합을 2*base 로 고정하고 분포 모양대로 나눈다
    up, dn = 2 * base * up_raw / total, 2 * base * dn_raw / total
    floor = round_trip_cost(costs) * float(grid_cfg.get("floor_mult", 2.0)) * n
    cap = float(grid_cfg.get("max_spacing", 0.10)) * n
    return min(max(up, floor), cap), min(max(dn, floor), cap)


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


def should_relay(old_spacing: float, new_spacing: float, grid_cfg: dict) -> bool:
    """폭이 충분히 바뀌었나 — 사다리를 다시 깔지 판단한다.

    밴드가 없으면 예측의 잡음만큼 매일 취소·재주문이 나간다. 실측에서 밴드 10% 는
    무밴드와 성과가 같은데 재배치는 4분의 1이었다(사이클당 4.0회 -> 1.1회).
    """
    if old_spacing <= 0:
        return True
    return abs(new_spacing / old_spacing - 1) > float(grid_cfg.get("relay_band", 0.10))


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
    up_span, dn_span = spans_from_quantiles(pred, costs, grid_cfg)
    per_level = (budget / 2) / n

    def empty(reason: str) -> Ladder:
        return Ladder(pred.code, price, spacing, budget, per_level, 0, [], [], reason)

    if price <= 0:
        return empty("현재가 없음")
    # 칸당 예산으로 1주도 못 사면 사다리가 성립하지 않는다 (고가주)
    if per_level < price:
        return empty(f"칸당 예산 {per_level:,.0f}원 < 1주 {price:,.0f}원")

    up_off = grid_offsets(n, up_span / n, ratio)
    dn_off = grid_offsets(n, dn_span / n, ratio)
    buys, sells = [], []
    for i, (uo, do) in enumerate(zip(up_off, dn_off, strict=True), start=1):
        bp = round_to_tick(price * (1 - do))            # 매수는 내림
        sp = round_to_tick(price * (1 + uo), up=True)   # 매도는 올림
        if bp > 0 and (q := int(per_level // bp)) > 0:
            buys.append(GridOrder(pred.code, "buy", bp, q, i))
        if sp > 0 and (q := int(per_level // sp)) > 0:
            sells.append(GridOrder(pred.code, "sell", sp, q, i))

    init_qty = int((budget / 2) // price)
    return Ladder(pred.code, price, spacing, budget, per_level, init_qty, buys, sells)


def paired_sell_price(
    ladder: Ladder, level: int,
    fill_price: float | None = None, costs: dict | None = None,
) -> int:
    """`level` 번 매수 칸이 체결됐을 때 **한 칸 위**에 걸 매도 가격.

    `price x (1 + spacing)` 이 아니라 실제 윗칸을 쓴다 — 가변 사다리(ratio<1)에서는
    칸마다 간격이 달라 그 근사가 어긋나고, 등간격에서도 호가 반올림 때문에 미세하게 틀린다.

    `fill_price`+`costs` 를 주면 **매입가 + 왕복비용** 아래로는 안 내려간다.
    폭 재조정(`should_relay`)으로 사다리가 좁아지면 윗칸이 매입가 밑으로 내려올 수
    있는데, 그 자리에 매도를 걸면 체결될 때마다 확정 손해다.
    """
    if level <= 1:
        base = round_to_tick(ladder.center, up=True)
    else:
        rung = next((o.price for o in ladder.buys if o.level == level - 1), None)
        if rung is None:
            raise ValueError(f"{ladder.code}: {level - 1}번 매수 칸이 없다")
        base = rung
    if fill_price and costs:
        base = max(base, round_to_tick(fill_price * (1 + round_trip_cost(costs)), up=True))
    return base
