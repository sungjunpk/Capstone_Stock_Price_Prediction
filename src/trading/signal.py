"""분위 예측 → 매매 신호 변환.

⚠️ CLAUDE.md 절대 규칙: 백테스트와 모의투자는 **이 파일의 함수를 그대로 공유**한다.
   실행 경로별 분기를 만들지 말 것. 여기서만 고치면 양쪽이 같이 바뀌어야 한다.

판단 순서 (프로젝트 핵심 차별점):
  1) 기권  — (q90-q10) 신뢰구간이 넓으면 "지금은 판단하지 않는다"
  2) 방향  — q50 이 거래비용을 넘는 임계값을 초과할 때만 진입
  3) 사이징— 확신할수록(구간이 좁을수록) 크게. 균등배분 아님
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Action(StrEnum):
    BUY = "buy"
    SELL = "sell"
    ABSTAIN = "abstain"   # 신뢰구간이 넓어 판단 보류
    HOLD = "hold"         # 판단은 했으나 임계값 미달


@dataclass(frozen=True)
class QuantilePrediction:
    code: str
    q10: float
    q50: float
    q90: float

    @property
    def interval_width(self) -> float:
        return self.q90 - self.q10

    def __post_init__(self) -> None:
        if not (self.q10 <= self.q50 <= self.q90):
            raise ValueError(
                f"{self.code}: 분위 교차 발생 "
                f"(q10={self.q10:.4f}, q50={self.q50:.4f}, q90={self.q90:.4f}) — "
                "모델 출력에 단조성 제약이 필요하다"
            )


@dataclass(frozen=True)
class Signal:
    code: str
    action: Action
    target_weight: float   # 총자산 대비 목표 비중 (0~max_position_pct)
    confidence: float      # 0~1, 사이징 근거
    reason: str


def round_trip_cost(costs: dict) -> float:
    """왕복 거래비용(비율). 임계값은 반드시 이보다 커야 의미가 있다."""
    bps = (
        2 * float(costs.get("commission_bps", 0.0))
        + float(costs.get("tax_bps", 0.0))
        + 2 * float(costs.get("slippage_bps", 0.0))
    )
    return bps / 10_000.0


def _confidence(width: float, max_width: float) -> float:
    """구간이 좁을수록 1 에 가깝게. max_width 에서 0."""
    if max_width <= 0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - width / max_width))


def generate_signal(pred: QuantilePrediction, trading_cfg: dict) -> Signal:
    """단일 종목 신호. 백테스트/모의투자 공용 진입점."""
    abstain_cfg = trading_cfg["abstain"]
    dir_cfg = trading_cfg["direction"]
    sizing_cfg = trading_cfg["sizing"]
    cost = round_trip_cost(trading_cfg.get("costs", {}))

    max_width = float(abstain_cfg["max_interval_width"])
    max_pos = float(sizing_cfg["max_position_pct"])

    # 1) 기권 판단 — 방향보다 먼저 온다
    if pred.interval_width > max_width:
        return Signal(
            pred.code, Action.ABSTAIN, 0.0, 0.0,
            f"신뢰구간 {pred.interval_width:.4f} > 임계 {max_width:.4f}",
        )

    # 2) 방향 판단 — 임계값은 거래비용 위로 강제
    long_th = max(float(dir_cfg["long_threshold"]), cost)
    short_th = min(float(dir_cfg["short_threshold"]), -cost)

    conf = _confidence(pred.interval_width, max_width)

    if pred.q50 >= long_th:
        action, edge = Action.BUY, pred.q50
    elif pred.q50 <= short_th:
        action, edge = Action.SELL, pred.q50
    else:
        return Signal(
            pred.code, Action.HOLD, 0.0, conf,
            f"q50 {pred.q50:.4f} 가 임계 [{short_th:.4f}, {long_th:.4f}] 안 — 진입 없음",
        )

    # 3) 포지션 사이징
    weight = _size(sizing_cfg, conf, edge, pred.interval_width, max_pos)
    return Signal(
        pred.code, action, weight, conf,
        f"q50={pred.q50:.4f}, 폭={pred.interval_width:.4f}, conf={conf:.2f}",
    )


def _size(
    sizing_cfg: dict, conf: float, edge: float, width: float, max_pos: float
) -> float:
    method = sizing_cfg.get("method", "inverse_width")

    if method == "inverse_width":
        w = conf * max_pos
    elif method == "kelly":
        # 분위 폭을 표준편차 대용으로 쓴 단순 Kelly (f* ≈ μ/σ²).
        # 풀 Kelly 는 과베팅이라 fraction 으로 줄인다.
        sigma = max(width / 2.563, 1e-6)  # q90-q10 ≈ 2.563σ (정규 가정)
        f = abs(edge) / (sigma**2)
        w = min(f * float(sizing_cfg.get("kelly_fraction", 0.25)), max_pos)
    else:
        raise ValueError(f"알 수 없는 sizing method: {method}")

    return float(min(max(w, 0.0), max_pos))


def generate_signals(
    preds: list[QuantilePrediction], trading_cfg: dict
) -> list[Signal]:
    """유니버스 전체. 총 익스포저 상한을 넘으면 비중을 비례 축소한다."""
    signals = [generate_signal(p, trading_cfg) for p in preds]

    max_gross = float(trading_cfg.get("risk", {}).get("max_gross_exposure", 1.0))
    gross = sum(s.target_weight for s in signals)
    if gross > max_gross > 0:
        scale = max_gross / gross
        signals = [
            Signal(s.code, s.action, s.target_weight * scale, s.confidence,
                   s.reason + f" [gross scale ×{scale:.2f}]")
            for s in signals
        ]
    return signals
