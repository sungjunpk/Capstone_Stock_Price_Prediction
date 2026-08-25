"""분위 예측 → 매매 신호 변환.

⚠️ CLAUDE.md 절대 규칙: 백테스트와 모의투자는 **이 파일의 함수를 그대로 공유**한다.
   실행 경로별 분기를 만들지 말 것. 여기서만 고치면 양쪽이 같이 바뀌어야 한다.

판단 순서 (프로젝트 핵심 차별점):
  1) 기권  — (q90-q10) 신뢰구간이 넓으면 "지금은 판단하지 않는다"
  2) 방향  — `direction.mode` 가 정한다
       absolute        q50 > 고정 임계값
       cross_sectional 기권 통과분끼리 q50 순위를 매겨 상위 N개 (기본값)
  3) 사이징— 확신할수록(구간이 좁을수록) 크게. 균등배분 아님

`cross_sectional` 이 기본인 이유는 `generate_signals` 의 docstring 에 적어두었다.
한 줄 요약: 절대 임계값은 예측 수준이 국면과 어긋나면 그대로 무너진다.
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


def resolve_abstain_threshold(widths, abstain_cfg: dict) -> float:
    """기권 임계값을 절대값으로 확정한다.

    설정이 숫자면 그대로 쓰고, `percentile: 30` 이면 **관측된 예측 폭의 30분위**를
    임계값으로 삼는다(= 가장 확신하는 30%만 거래).

    절대값을 미리 추측하면 거의 항상 틀린다. 실측: 5일 수익률의 자연 폭이 0.124 인데
    초기 추측값은 0.05 여서 기권률이 95.8% 가 나왔다(거래 0건).

    백테스트와 모의투자가 **같은 함수**를 쓴다 — 모의투자에서는 최근 예측 폭들을
    넣어 같은 방식으로 임계값을 구한다.
    """
    if "percentile" in abstain_cfg:
        import numpy as np

        w = np.asarray([float(x) for x in widths])
        if w.size == 0:
            raise ValueError("percentile 방식은 관측된 폭이 필요하다")
        return float(np.percentile(w, float(abstain_cfg["percentile"])))
    return float(abstain_cfg["max_interval_width"])


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


def generate_signal(
    pred: QuantilePrediction, trading_cfg: dict, *, max_width: float | None = None
) -> Signal:
    """단일 종목 신호. 백테스트/모의투자 공용 진입점.

    max_width: 기권 임계값을 밖에서 확정해 넘길 때 사용(percentile 방식).
        None 이면 설정의 절대값을 쓴다.
    """
    dir_cfg = trading_cfg["direction"]
    sizing_cfg = trading_cfg["sizing"]
    cost = round_trip_cost(trading_cfg.get("costs", {}))

    if max_width is None:
        max_width = float(trading_cfg["abstain"]["max_interval_width"])
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

    # rank_normalized 는 횡단면 모드 전용이다(비중을 종목들 사이에서 정규화하므로
    # 단일 종목만 보고는 계산할 수 없다). absolute 모드에서 이 설정을 만나면
    # 같은 취지의 per-stock 방식인 inverse_width 로 물러난다 — 대조군 실행을 위해서다.
    if method in ("inverse_width", "rank_normalized"):
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
    preds: list[QuantilePrediction], trading_cfg: dict,
    *, max_width: float | None = None,
) -> list[Signal]:
    """유니버스 전체 신호. **백테스트/모의투자 공용 진입점이다.**

    방향 판단 방식이 두 가지다 (`direction.mode`):

      absolute        q50 이 고정 임계값을 넘으면 매수. 단일 종목만 봐도 판단 가능
      cross_sectional 기권을 통과한 종목끼리 q50 순위를 매겨 상위 N개 매수

    cross_sectional 을 쓰는 이유: absolute 는 **예측 수준(level)이 국면과 어긋나면
    그대로 무너진다.** 실측(2026-08-25 test 구간)에서 모델의 5일 예측 중앙값이
    -0.15% 인데 실제 평균은 +0.80% 였고, 매수 임계값 0.40% 는 예측의 90분위(0.27%)
    보다도 높아서 2.1년 동안 체결이 64건에 그쳤다.
    순위 방식은 모든 종목에 공통으로 낀 편차가 상쇄되므로 이 문제를 받지 않는다.
    """
    mode = str(trading_cfg["direction"].get("mode", "absolute"))

    if mode == "absolute":
        signals = [generate_signal(p, trading_cfg, max_width=max_width) for p in preds]
    elif mode == "cross_sectional":
        signals = _cross_sectional_signals(preds, trading_cfg, max_width=max_width)
    else:
        raise ValueError(f"알 수 없는 direction.mode: {mode}")

    max_gross = float(trading_cfg.get("risk", {}).get("max_gross_exposure", 1.0))
    gross = sum(s.target_weight for s in signals)
    if gross > max_gross > 0:
        scale = max_gross / gross
        signals = [
            Signal(s.code, s.action, s.target_weight * scale, s.confidence,
                   s.reason + f" [gross scale \u00d7{scale:.2f}]")
            for s in signals
        ]
    return signals


def _target_exposure(n_survivors: int, universe_size: int, trading_cfg: dict) -> float:
    """이번 판단에서 총 몇 %를 투자할 것인가.

    `exposure_scaling` 이 켜져 있으면 **살아남은 후보 수에 비례**해 노출도를 줄인다.
    이게 없으면 정규화 때문에 항상 만기 투자가 되어, 기권 로직이 '무엇을 사는가'에만
    영향을 주고 '얼마나 쉬는가'에는 영향을 못 준다 — 이 프로젝트의 차별점이 반쪽이 된다.

    평소(기권 임계값을 percentile 로 잡았으므로 생존율 ≈ 그 값)엔 만기 투자가 되고,
    시장 전체 불확실성이 커져 통과 종목이 줄면 자동으로 현금 비중이 는다.
    """
    risk_cfg = trading_cfg.get("risk", {})
    max_gross = float(risk_cfg.get("max_gross_exposure", 1.0))

    if not bool(trading_cfg["sizing"].get("exposure_scaling", False)):
        return max_gross
    if universe_size <= 0:
        return 0.0

    abstain_cfg = trading_cfg["abstain"]
    # percentile 방식이면 그 값이 곧 '정상적인 생존율'이다.
    target_ratio = float(abstain_cfg.get("percentile", 100.0)) / 100.0
    if target_ratio <= 0:
        return max_gross

    survivor_ratio = n_survivors / universe_size
    return max_gross * min(survivor_ratio / target_ratio, 1.0)


def _normalize_weights(
    confs: list[float], exposure: float, max_pos: float
) -> list[float]:
    """확신도 비율대로 나눠 담되, 종목당 상한을 넘으면 나머지에 재분배한다.

    단순히 conf x max_pos 로 하면 안 된다. 기권을 통과한 종목은 정의상 폭이 임계값
    아래라 conf 가 0 근처에 몰려 있어서, 상위 종목을 골라도 총 노출이 20%를 못 넘는다
    (실측에서 이것이 두 번째 병목이었다).
    """
    n = len(confs)
    if n == 0 or exposure <= 0:
        return [0.0] * n

    # conf 가 전부 0이면(폭이 모두 임계값에 붙어 있으면) 균등배분으로 물러난다
    total = sum(confs)
    share = [c / total for c in confs] if total > 1e-12 else [1.0 / n] * n

    weights = [s * exposure for s in share]
    for _ in range(n):                       # 상한에 걸린 만큼만 재분배, 최대 n회
        excess = sum(max(w - max_pos, 0.0) for w in weights)
        if excess <= 1e-12:
            break
        free = [i for i, w in enumerate(weights) if w < max_pos - 1e-12]
        if not free:
            break
        weights = [min(w, max_pos) for w in weights]
        room = sum(max_pos - weights[i] for i in free)
        if room <= 1e-12:
            break
        add = min(excess, room)
        for i in free:
            weights[i] += add * (max_pos - weights[i]) / room
    return [min(w, max_pos) for w in weights]


def _cross_sectional_signals(
    preds: list[QuantilePrediction], trading_cfg: dict,
    *, max_width: float | None = None,
) -> list[Signal]:
    """기권 -> 순위 -> 사이징. 판단 순서에서 **기권이 여전히 맨 앞이다.**"""
    dir_cfg = trading_cfg["direction"]
    sizing_cfg = trading_cfg["sizing"]

    if max_width is None:
        max_width = float(trading_cfg["abstain"]["max_interval_width"])
    max_pos = float(sizing_cfg["max_position_pct"])
    top_n = int(dir_cfg.get("top_n", 10))
    min_candidates = int(dir_cfg.get("min_candidates", 0))

    # 1) 기권 — 신뢰구간이 넓으면 순위 경쟁에 아예 참여시키지 않는다
    survivors, out = [], {}
    for p in preds:
        if p.interval_width > max_width:
            out[p.code] = Signal(
                p.code, Action.ABSTAIN, 0.0, 0.0,
                f"신뢰구간 {p.interval_width:.4f} > 임계 {max_width:.4f}",
            )
        else:
            survivors.append(p)

    # 2) 후보가 너무 적으면 순위 자체가 의미 없다 — 전원 관망
    if len(survivors) < min_candidates:
        for p in survivors:
            out[p.code] = Signal(
                p.code, Action.ABSTAIN, 0.0, 0.0,
                f"후보 {len(survivors)}개 < 최소 {min_candidates}개 — 순위 무의미",
            )
        return [out[p.code] for p in preds]

    # 3) q50 내림차순 상위 top_n 만 매수. 공통 편차는 여기서 상쇄된다
    ranked = sorted(survivors, key=lambda p: -p.q50)
    chosen, rest = ranked[:top_n], ranked[top_n:]

    for rank, p in enumerate(rest, start=len(chosen) + 1):
        out[p.code] = Signal(
            p.code, Action.HOLD, 0.0, _confidence(p.interval_width, max_width),
            f"q50 순위 {rank}/{len(ranked)} — 상위 {top_n} 밖",
        )

    # 4) 사이징 — 확신도 비율대로 나눠 담는다
    exposure = _target_exposure(len(survivors), len(preds), trading_cfg)
    confs = [_confidence(p.interval_width, max_width) for p in chosen]
    weights = _normalize_weights(confs, exposure, max_pos)

    for rank, (p, conf, w) in enumerate(zip(chosen, confs, weights, strict=True), 1):
        out[p.code] = Signal(
            p.code, Action.BUY, w, conf,
            f"q50 순위 {rank}/{len(ranked)} (q50={p.q50:.4f}, "
            f"\ud3ed={p.interval_width:.4f}, \ub178\ucd9c={exposure:.0%})",
        )

    return [out[p.code] for p in preds]
