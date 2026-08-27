"""리스크 오버레이 — 매매 판단 4단계.

signal.py 가 낸 신호를 **주문 직전에** 걸러낸다.
백테스트와 모의투자가 같은 코드를 쓴다(CLAUDE.md 절대 규칙).

여기서 막는 것:
  - 개별 종목 비중 상한
  - 총 익스포저 상한
  - 손절/익절/보유만료 (보유 중인 포지션)
  - 일일 최대 거래횟수 — 과적합 신호로 인한 과도거래 방지
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from src.trading.signal import Action, Signal


@dataclass(frozen=True)
class Position:
    code: str
    weight: float          # 현재 총자산 대비 비중
    entry_price: float
    days_held: int = 0     # 리밸런싱 단위 경과 수. 분봉 트랙에서는 '봉' 이다

    def pnl_pct(self, current_price: float) -> float:
        if self.entry_price <= 0:
            return 0.0
        return current_price / self.entry_price - 1.0


# 차단 사유 카테고리. **발생 지점에서 세고, 문자열을 파싱하지 않는다.**
# (파싱은 메시지 문구를 바꾸는 순간 조용히 깨진다)
STOP_LOSS = "손절"
TAKE_PROFIT = "익절"
MAX_HOLDING = "보유만료"
TRADE_CAP = "거래한도"
NO_SHORT = "공매도불가"


@dataclass(frozen=True)
class RiskDecision:
    """오버레이를 통과한 최종 주문."""

    signals: list[Signal]
    forced_exits: list[str]        # 손절/익절로 강제 청산되는 종목
    blocked: dict[str, str]        # 종목 → 차단 사유 (리포트용)
    blocked_by_reason: dict[str, int] = field(default_factory=dict)


def apply_risk_overlay(
    signals: list[Signal],
    positions: dict[str, Position],
    prices: dict[str, float],
    trading_cfg: dict,
    *,
    allow_short: bool = False,
    liquidate_unsignaled: bool = True,
) -> RiskDecision:
    """신호 목록 → 리스크 규칙을 적용한 최종 주문.

    allow_short: 국내 개인 공매도는 사실상 제한적이고 모의투자도 매수 위주라
        기본은 False. SELL 신호는 '보유 중이면 청산, 아니면 무시'로 처리한다.

    liquidate_unsignaled: 호출자가 **신호 없는 보유분을 청산**하는지 여부.
        백테스트가 그렇게 동작하므로 기본 True 다.
        이걸 틀리게 잡으면 총 익스포저 계산이 무너진다 — 곧 팔 종목을 '보유 중'으로
        세어 신규 진입 비중을 깎아버린다. 실측(2026-08-25): 9종목을 교체 매매하는
        상황에서 목표 0.81 이 0.09 로 잘렸다(9배 축소).
        매 회차 종목을 갈아타는 횡단면 순위 방식에서는 특히 치명적이다.
    """
    risk = trading_cfg.get("risk", {})
    max_pos = float(trading_cfg["sizing"]["max_position_pct"])
    max_gross = float(risk.get("max_gross_exposure", 1.0))
    max_trades = int(risk.get("max_trades_per_day", 10**9))
    stop_loss = float(risk.get("stop_loss_pct", -1.0))
    take_profit = float(risk.get("take_profit_pct", 10.0))
    # 예측 지평이 지나면 그 예측은 만료다. 0/미설정이면 비활성(일봉 트랙의 기존 동작).
    # 타점 탐지 트랙에서는 이게 없으면 손절/익절에 안 걸린 포지션이 영원히 남는다.
    max_holding = int(risk.get("max_holding_bars", 0))

    blocked: dict[str, str] = {}
    reasons: dict[str, int] = {}

    def _block(code: str, category: str, detail: str) -> None:
        blocked[code] = detail
        reasons[category] = reasons.get(category, 0) + 1

    # --- 1) 손절/익절/보유만료: 신호와 무관하게 먼저 강제 청산
    forced_exits = []
    for code, pos in positions.items():
        px = prices.get(code)
        pnl = pos.pnl_pct(px) if px is not None else None
        if pnl is not None and pnl <= stop_loss:
            forced_exits.append(code)
            _block(code, STOP_LOSS, f"손절 {pnl:+.1%}")
        elif pnl is not None and pnl >= take_profit:
            forced_exits.append(code)
            _block(code, TAKE_PROFIT, f"익절 {pnl:+.1%}")
        elif max_holding and pos.days_held >= max_holding:
            forced_exits.append(code)
            _block(code, MAX_HOLDING, f"보유 {pos.days_held} ≥ 만료 {max_holding}")

    # --- 2) 공매도 불가면 SELL 은 '보유 시 청산'으로만 해석
    kept: list[Signal] = []
    for s in signals:
        if s.code in forced_exits:
            continue
        if s.action is Action.SELL and not allow_short:
            if s.code in positions:
                kept.append(replace(s, target_weight=0.0,
                                    reason=s.reason + " [청산]"))
            else:
                _block(s.code, NO_SHORT, "공매도 불가 — 무시")
            continue
        if s.action in (Action.ABSTAIN, Action.HOLD):
            continue
        kept.append(s)

    # --- 3) 개별 종목 상한
    kept = [
        replace(s, target_weight=min(s.target_weight, max_pos)) for s in kept
    ]

    # --- 4) 일일 최대 거래횟수 — 확신도가 높은 순으로 남긴다
    new_entries = [s for s in kept if s.target_weight > 0]
    if len(new_entries) > max_trades:
        new_entries.sort(key=lambda s: -s.confidence)
        for s in new_entries[max_trades:]:
            _block(s.code, TRADE_CAP, f"일일 거래한도 {max_trades} 초과")
        keep_codes = {s.code for s in new_entries[:max_trades]}
        kept = [s for s in kept if s.target_weight == 0 or s.code in keep_codes]

    # --- 5) 총 익스포저 상한 (**계속 들고 갈** 보유분만 포함)
    #
    # 여기에 셀 것은 "이번 주문 뒤에도 남아 있을 비중"이다.
    #   - forced_exits    : 손절/익절로 청산 → 안 남는다
    #   - kept 에 있는 종목: 아래 신호의 target_weight 로 대체된다 → 중복으로 세면 안 된다
    #   - 나머지 보유분   : 호출자가 청산한다면 안 남고, 유지한다면 남는다
    kept_codes = {s.code for s in kept}
    held = 0.0 if liquidate_unsignaled else sum(
        p.weight for c, p in positions.items()
        if c not in forced_exits and c not in kept_codes
    )
    new_gross = sum(s.target_weight for s in kept)
    if held + new_gross > max_gross and new_gross > 0:
        scale = max(max_gross - held, 0.0) / new_gross
        kept = [
            replace(s, target_weight=s.target_weight * scale,
                    reason=s.reason + f" [gross ×{scale:.2f}]")
            for s in kept
        ]

    return RiskDecision(signals=kept, forced_exits=forced_exits,
                        blocked=blocked, blocked_by_reason=reasons)
