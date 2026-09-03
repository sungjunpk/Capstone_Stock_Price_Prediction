"""리스크 오버레이 — 매매 판단 4단계.

signal.py 가 낸 신호를 **주문 직전에** 걸러낸다.
백테스트와 모의투자가 같은 코드를 쓴다(CLAUDE.md 절대 규칙).

여기서 막는 것:
  - 개별 종목 비중 상한
  - 총 익스포저 상한
  - 손절/익절/보유만료 (보유 중인 포지션)
  - 일일 최대 거래횟수 — 과적합 신호로 인한 과도거래 방지
  - 포트폴리오 CVaR 한도 (기본 비활성) — 위 규칙이 전부 종목 단위라 생기는 구멍
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import numpy as np

from src.trading.signal import Action, Signal

if TYPE_CHECKING:
    import pandas as pd


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
CVAR_LIMIT = "CVaR한도"


@dataclass(frozen=True)
class RiskDecision:
    """오버레이를 통과한 최종 주문."""

    signals: list[Signal]
    forced_exits: list[str]        # 손절/익절로 강제 청산되는 종목
    blocked: dict[str, str]        # 종목 → 차단 사유 (리포트용)
    blocked_by_reason: dict[str, int] = field(default_factory=dict)
    cvar: float | None = None      # 축소 **전** 목표 포트폴리오의 꼬리손실. None = 미측정
    cvar_scale: float = 1.0        # CVaR 한도가 물어서 곱한 축소계수. 1.0 = 안 물었다
    cvar_market: float | None = None   # 같은 시점 시장 균등배분의 꼬리 (상대 한도의 기준자)


def portfolio_cvar(
    weights: dict[str, float],
    returns: pd.DataFrame,
    *,
    alpha: float = 0.05,
    lookback: int = 250,
    min_obs: int = 120,
) -> float | None:
    """목표 비중 + 과거 일별 수익률 → 포트폴리오 꼬리손실. **양수로 반환**한다.

    역사적 시뮬레이션이다. 비중을 과거 수익률에 그대로 대입해 포트폴리오 수익률
    분포를 합성하고, 최악 `alpha` 구간의 **평균**을 취한다.
    0.02 = "나쁜 날이 오면 평균 2% 잃는다".

    VaR(경계선)이 아니라 CVaR(경계 너머의 평균)인 이유: VaR 은 꼬리 안쪽을 보지 않아
    경계 너머가 -2% 든 -20% 든 같은 값을 낸다.

    실현 자산곡선이 아니라 **후보 포트폴리오의 합성 분포**를 쓰는 이유 둘:
      1) 실거래 기록이 8일치라 5% 꼬리를 잡을 수 없다. 합성은 첫날부터 계산된다.
      2) 집중위험은 이 방법으로만 잡힌다 — 같이 움직이는 종목만 담으면 꼬리가 두꺼워진다.

    ⚠️ `returns` 는 **결정 시점까지 잘라서** 넘겨야 한다. 이 함수는 자르지 않는다.
       look-ahead 방지는 호출자 책임이다(백테스트가 signal_date 로 자른다).

    관측이 `min_obs` 에 못 미치면 None 이다. 꼬리를 못 재는데 한도를 걸면
    **표본 부족이 곧 축소**로 이어진다 — 모르면 개입하지 않는다.
    """
    codes = [c for c, w in weights.items() if w > 0 and c in returns.columns]
    if not codes:
        return None
    r = returns[codes].tail(lookback).dropna(how="any")
    if len(r) < min_obs:
        return None
    return _tail_mean(r.to_numpy() @ np.array([weights[c] for c in codes]), alpha)


def market_cvar(
    returns: pd.DataFrame,
    *,
    alpha: float = 0.05,
    lookback: int = 250,
    min_obs: int = 120,
) -> float | None:
    """전 종목 **균등배분(총 노출 1.0)** 의 꼬리손실. 상대 한도의 기준자다.

    `portfolio_cvar` 를 쓰지 않는 이유: 유니버스 전체에 `dropna(how="any")` 를 걸면
    종목 하나가 하루 비어도 그날 행이 통째로 날아가 표본이 남지 않는다.
    날짜별 평균은 결측을 자연스럽게 흡수하면서 같은 값을 준다.
    """
    r = returns.tail(lookback).mean(axis=1).dropna()
    if len(r) < min_obs:
        return None
    return _tail_mean(r.to_numpy(), alpha)


def _tail_mean(port, alpha: float) -> float:
    """최악 alpha 구간의 평균 손실(양수)."""
    k = max(1, int(len(port) * alpha))
    return float(-np.sort(port)[:k].mean())


def apply_risk_overlay(
    signals: list[Signal],
    positions: dict[str, Position],
    prices: dict[str, float],
    trading_cfg: dict,
    *,
    allow_short: bool = False,
    liquidate_unsignaled: bool = True,
    returns: pd.DataFrame | None = None,
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

    returns: 종목별 일별 수익률(index=date, columns=code). CVaR 한도(6단계)에만 쓴다.
        **결정 시점까지 잘라서** 넘겨야 한다 — 이 함수는 자르지 않는다.
        None 이거나 `risk.cvar_limit` 이 없으면 6단계를 건너뛴다(현재 기본값).
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

    # --- 6) 포트폴리오 CVaR 한도 (기본 비활성)
    #
    # 1~5 는 전부 **종목 단위** 규칙이다 — 이 종목 10% 이내, 이 종목 -5%면 손절.
    # 20종목이 전부 같이 무너지는 상황을 보는 눈이 여기서 처음 생긴다.
    #
    # CVaR 은 양의 1차동차다(현금 수익률 0): CVaR(s·w) = s·CVaR(w).
    # 그래서 탐색 없이 scale = 한도/CVaR 로 **정확히** 한도에 맞춘다.
    #
    # ⚠️ 축소는 `kept`(신규·조정분)에만 걸린다. 일봉 트랙은 매 리밸런싱 전량 재구성이라
    #    이게 곧 포트폴리오 전체지만, hold_until_exit(60분봉 트랙)에서는 유지 중인
    #    보유분이 안 줄어든다 — 그 트랙은 판정 대상이 아니라 한계만 적어둔다.
    #
    # 한도는 두 가지로 줄 수 있다. **축소 경로는 하나뿐이다** — 상대 한도는 매 결정
    # 시점에 절대 한도로 해석될 뿐, 분기를 만들지 않는다.
    #   cvar_limit       절대값. 2026-09-03 판정에서 **기각**됐다(국면 간 이식 실패).
    #   cvar_limit_ratio 시장 균등배분 꼬리의 배수. 국면이 험해지면 허용치도 같이 커진다.
    cvar_limit = risk.get("cvar_limit")
    cvar_ratio = risk.get("cvar_limit_ratio")
    cvar = cvar_market = None
    cvar_scale = 1.0
    # ⚠️ `is not None` 이어야 한다. 0.0 은 falsy라 truthy 검사로 두면
    #    "가장 엄격한 한도"가 "게이트 꺼짐"으로 조용히 뒤집힌다.
    if (cvar_limit is not None or cvar_ratio is not None) \
            and returns is not None and kept:
        # 5단계 `held` 와 같은 기준으로 "주문 뒤에 남아 있을 비중"을 모은다
        w_all = {s.code: s.target_weight for s in kept}
        if not liquidate_unsignaled:
            for c, p in positions.items():
                if c not in forced_exits and c not in kept_codes:
                    w_all[c] = p.weight
        kw = {"alpha": float(risk.get("cvar_alpha", 0.05)),
              "lookback": int(risk.get("cvar_lookback", 250)),
              "min_obs": int(risk.get("cvar_min_obs", 120))}
        cvar = portfolio_cvar(w_all, returns, **kw)
        if cvar_ratio is not None:
            cvar_market = market_cvar(returns, **kw)
            cvar_limit = float(cvar_ratio) * cvar_market if cvar_market else None
        if cvar is not None and cvar_limit is not None and cvar > float(cvar_limit):
            cvar_scale = float(cvar_limit) / cvar
            kept = [
                replace(s, target_weight=s.target_weight * cvar_scale,
                        reason=s.reason + f" [CVaR ×{cvar_scale:.2f}]")
                for s in kept
            ]
            # 종목을 막은 게 아니라 전체를 줄인 것이라 `blocked` 에는 넣지 않는다.
            # 사유 카운터에만 올려 "몇 번 물었나"를 백테스트가 그대로 집계하게 한다.
            reasons[CVAR_LIMIT] = reasons.get(CVAR_LIMIT, 0) + 1

    return RiskDecision(signals=kept, forced_exits=forced_exits,
                        blocked=blocked, blocked_by_reason=reasons,
                        cvar=cvar, cvar_scale=cvar_scale, cvar_market=cvar_market)
