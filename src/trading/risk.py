"""리스크 오버레이 — 매매 판단 4단계.

signal.py 가 낸 신호를 **주문 직전에** 걸러낸다.
백테스트와 모의투자가 같은 코드를 쓴다(CLAUDE.md 절대 규칙).

여기서 막는 것:
  - 개별 종목 비중 상한
  - 총 익스포저 상한
  - 손절/익절 (보유 중인 포지션)
  - 일일 최대 거래횟수 — 과적합 신호로 인한 과도거래 방지
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from src.trading.signal import Action, Signal


@dataclass(frozen=True)
class Position:
    code: str
    weight: float          # 현재 총자산 대비 비중
    entry_price: float
    days_held: int = 0

    def pnl_pct(self, current_price: float) -> float:
        if self.entry_price <= 0:
            return 0.0
        return current_price / self.entry_price - 1.0


@dataclass(frozen=True)
class RiskDecision:
    """오버레이를 통과한 최종 주문."""

    signals: list[Signal]
    forced_exits: list[str]        # 손절/익절로 강제 청산되는 종목
    blocked: dict[str, str]        # 종목 → 차단 사유 (리포트용)


def apply_risk_overlay(
    signals: list[Signal],
    positions: dict[str, Position],
    prices: dict[str, float],
    trading_cfg: dict,
    *,
    allow_short: bool = False,
) -> RiskDecision:
    """신호 목록 → 리스크 규칙을 적용한 최종 주문.

    allow_short: 국내 개인 공매도는 사실상 제한적이고 모의투자도 매수 위주라
        기본은 False. SELL 신호는 '보유 중이면 청산, 아니면 무시'로 처리한다.
    """
    risk = trading_cfg.get("risk", {})
    max_pos = float(trading_cfg["sizing"]["max_position_pct"])
    max_gross = float(risk.get("max_gross_exposure", 1.0))
    max_trades = int(risk.get("max_trades_per_day", 10**9))
    stop_loss = float(risk.get("stop_loss_pct", -1.0))
    take_profit = float(risk.get("take_profit_pct", 10.0))

    blocked: dict[str, str] = {}

    # --- 1) 손절/익절: 신호와 무관하게 먼저 강제 청산
    forced_exits = []
    for code, pos in positions.items():
        px = prices.get(code)
        if px is None:
            continue
        pnl = pos.pnl_pct(px)
        if pnl <= stop_loss:
            forced_exits.append(code)
            blocked[code] = f"손절 {pnl:+.1%}"
        elif pnl >= take_profit:
            forced_exits.append(code)
            blocked[code] = f"익절 {pnl:+.1%}"

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
                blocked[s.code] = "공매도 불가 — 무시"
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
            blocked[s.code] = f"일일 거래한도 {max_trades} 초과"
        keep_codes = {s.code for s in new_entries[:max_trades]}
        kept = [s for s in kept if s.target_weight == 0 or s.code in keep_codes]

    # --- 5) 총 익스포저 상한 (기존 보유분 포함)
    held = sum(
        p.weight for c, p in positions.items()
        if c not in forced_exits and c not in {s.code for s in kept}
    )
    new_gross = sum(s.target_weight for s in kept)
    if held + new_gross > max_gross and new_gross > 0:
        scale = max(max_gross - held, 0.0) / new_gross
        kept = [
            replace(s, target_weight=s.target_weight * scale,
                    reason=s.reason + f" [gross ×{scale:.2f}]")
            for s in kept
        ]

    return RiskDecision(signals=kept, forced_exits=forced_exits, blocked=blocked)
