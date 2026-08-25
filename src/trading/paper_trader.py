"""모의투자 실행 — 매매 판단 5단계의 마지막 칸.

    예측(inference) → 기권·순위·사이징(signal) → 리스크(risk) → 주문수량 → 전송(broker)
                       └──────── 백테스트와 **완전히 같은 코드** ────────┘

이 파일이 새로 하는 일은 딱 하나다: **목표 비중을 주문 수량으로 바꾸는 것.**
판단은 전부 signal.py / risk.py 가 이미 끝냈다. 여기서 "그런데 실전에서는…" 하고
규칙을 덧대는 순간 백테스트에서 검증한 숫자가 의미를 잃는다(CLAUDE.md 절대 규칙 7).

비중 → 수량에서 백테스트에 없던 현실 제약 셋:

  1) 주식은 정수 주 단위다. 목표 비중이 1주 미만이면 그냥 못 산다.
  2) 매수는 **주문가능금액**을 못 넘는다. 매도 대금이 언제 쓸 수 있게 되는지는
     계좌마다 다르므로 추정하지 않고, 매도를 먼저 보내고 **계좌를 다시 조회**한다.
  3) 매도는 **매도가능수량**까지만. 미결제분은 브로커가 거부한다.

이 셋은 판단이 아니라 체결 제약이라 백테스트와의 등가성을 깨지 않는다.
다만 결과는 달라질 수 있어서, 실행 로그에 계획과 실제를 나란히 남긴다.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.trading.broker import BUY, SELL, AccountSnapshot, OrderResult, PaperBroker
from src.trading.risk import Position, apply_risk_overlay
from src.trading.signal import (
    Action,
    QuantilePrediction,
    Signal,
    generate_signals,
    resolve_abstain_threshold,
    should_trade,
)
from src.utils.config import PROJECT_ROOT
from src.utils.logging import get_logger

log = get_logger(__name__)

STATE_DIR = PROJECT_ROOT / "outputs" / "paper_trading"
STATE_PATH = STATE_DIR / "state.json"
RUNS_DIR = STATE_DIR / "runs"


# ---------------------------------------------------------------- 상태
@dataclass
class TraderState:
    """실행 사이에 남겨야 하는 최소한의 것만 담는다.

    **보유수량·매입가는 여기 두지 않는다** — 브로커가 정본이다.
    로컬에 복제하면 언젠가 어긋나고, 어긋난 쪽으로 주문이 나간다.
    여기 있는 건 브로커가 안 알려주는 두 가지뿐이다: 진입일과 마지막 리밸런싱일.
    """

    last_rebalance: str | None = None
    entry_dates: dict[str, str] = field(default_factory=dict)
    runs: int = 0

    @classmethod
    def load(cls, path: Path = STATE_PATH) -> TraderState:
        if not path.exists():
            return cls()
        try:
            return cls(**json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, TypeError) as exc:
            log.warning("상태 파일을 읽지 못했다(%s) — 빈 상태로 시작한다", exc)
            return cls()

    def save(self, path: Path = STATE_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2),
                        encoding="utf-8")

    def days_held(self, code: str, today: date) -> int:
        """영업일 기준 보유일수. 진입일을 모르면 0 (손절은 매입가 기준이라 무관)."""
        raw = self.entry_dates.get(code)
        if not raw:
            return 0
        return int(np.busday_count(np.datetime64(raw, "D"), np.datetime64(today, "D")))

    def sync_entries(self, held: set[str], today: date) -> None:
        """새로 생긴 보유는 오늘 진입으로, 사라진 보유는 기록에서 지운다."""
        for code in held - set(self.entry_dates):
            self.entry_dates[code] = today.isoformat()
        for code in set(self.entry_dates) - held:
            self.entry_dates.pop(code, None)


# ---------------------------------------------------------------- 계획
@dataclass
class PlannedOrder:
    code: str
    side: str
    quantity: int
    price: float
    weight_from: float
    weight_to: float
    reason: str

    @property
    def amount(self) -> float:
        return self.quantity * self.price

    def to_dict(self) -> dict:
        return {**asdict(self), "amount": round(self.amount, 0)}


@dataclass
class TradingPlan:
    """오늘 무엇을 왜 하는가. 주문 전에 사람이 읽고 판단할 수 있어야 한다."""

    decision_date: str            # 예측이 만들어진 날(= 패널 마지막 거래일)
    generated_at: str
    equity: float
    cash: float                   # 주문가능금액 — 매수 계획의 상한
    rebalancing: bool
    abstain_threshold: float
    orders: list[PlannedOrder] = field(default_factory=list)
    signals: list[dict] = field(default_factory=list)
    forced_exits: list[str] = field(default_factory=list)
    blocked: dict[str, str] = field(default_factory=dict)
    blocked_by_reason: dict[str, int] = field(default_factory=dict)
    stats: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["orders"] = [o.to_dict() for o in self.orders]
        return d


def _to_positions(account: AccountSnapshot, state: TraderState, today: date
                  ) -> dict[str, Position]:
    """계좌 보유분 → 리스크 오버레이가 아는 형태.

    entry_price 는 **브로커의 매입단가**다. 로컬 추정이 아니라서 상태 파일이
    날아가도 손절 기준이 살아 있다.
    """
    return {
        code: Position(
            code=code,
            weight=account.weight_of(code),
            entry_price=h.avg_price,
            days_held=state.days_held(code, today),
        )
        for code, h in account.holdings.items()
    }


def build_plan(
    recent_preds: pd.DataFrame,
    account: AccountSnapshot,
    prices: dict[str, float],
    cfg: dict,
    *,
    state: TraderState,
    today: date | None = None,
    rebalancing: bool = True,
) -> TradingPlan:
    """예측 + 계좌 → 오늘의 주문 계획. **여기서 API 를 호출하지 않는다.**

    순수 함수로 두는 이유: 계획 수립을 계좌 조회/주문 전송과 분리해야
    테스트에서 실제 주문 없이 판단을 검증할 수 있다.

    recent_preds: 최근 구간 예측(code/date/q10/q50/q90).
        **최신 하루가 아니라 구간**을 받는 이유는 기권 임계값이 폭 분포의
        백분위이기 때문이다 — 오늘 하루만 보면 '오늘이 평소보다 불확실한가'를
        판단할 수 없다(`src/models/inference.py: predict_recent`).
    rebalancing: False 면 신규 진입/비중 조정을 하지 않고 **강제 청산만** 본다.
        리밸런싱 주기(기본 5일) 사이에도 손절이 동작하게 하기 위한 경로다.
    """
    tcfg = cfg["trading"]
    today = today or date.today()
    min_trade = float(tcfg["sizing"].get("min_trade_weight", 0.0))

    dates = pd.to_datetime(recent_preds["date"])
    decision_date = dates.max()
    latest = recent_preds[dates == decision_date]

    # 1) 기권 임계값 — 백테스트와 같은 함수, 입력만 '최근 예측 폭'이다
    widths = (recent_preds["q90"] - recent_preds["q10"]).to_numpy()
    max_width = resolve_abstain_threshold(widths, tcfg["abstain"])

    notes: list[str] = []
    preds: list[QuantilePrediction] = []
    for r in latest.itertuples():
        if r.code not in prices:
            continue                      # 현재가를 못 받은 종목은 주문 수량을 못 낸다
        try:
            preds.append(QuantilePrediction(r.code, float(r.q10), float(r.q50), float(r.q90)))
        except ValueError as exc:
            log.warning("분위 교차 무시: %s", exc)

    held = set(account.holdings)
    positions = _to_positions(account, state, today)

    # 2) 판단 — 백테스트와 **같은 함수**. 보유분을 넘겨야 이력 버퍼가 산다
    if rebalancing:
        signals = generate_signals(preds, tcfg, max_width=max_width, held=held)
    else:
        signals = []
        notes.append("리밸런싱 주기가 아니다 — 손절/익절만 본다")

    decision = apply_risk_overlay(
        signals, positions, prices, tcfg, allow_short=False,
        # 신호 없는 보유분을 청산하는 건 리밸런싱 때뿐이다.
        # 비리밸런싱 날에 True 로 두면 신호가 없다는 이유로 전량 청산이 나간다.
        liquidate_unsignaled=rebalancing,
    )

    # 3) 목표 비중 확정
    target: dict[str, float] = {s.code: s.target_weight for s in decision.signals}
    for code in decision.forced_exits:
        target[code] = 0.0
    if rebalancing:
        for code in held:
            target.setdefault(code, 0.0)   # 신호가 없으면 청산 — 백테스트와 같다

    # 4) 비중 → 수량
    orders = _weights_to_orders(target, account, prices, min_trade, decision)

    stats = _signal_stats(signals, decision, max_width, len(preds))
    if account.equity <= 0:
        notes.append("총자산이 0 이다 — 계좌 조회 결과를 확인할 것")

    return TradingPlan(
        decision_date=pd.Timestamp(decision_date).date().isoformat(),
        generated_at=datetime.now().isoformat(timespec="seconds"),
        equity=round(account.equity, 0),
        cash=round(account.cash, 0),
        rebalancing=rebalancing,
        abstain_threshold=round(float(max_width), 5),
        orders=orders,
        signals=[_signal_row(s, prices, account) for s in signals],
        forced_exits=list(decision.forced_exits),
        blocked=dict(decision.blocked),
        blocked_by_reason=dict(decision.blocked_by_reason),
        stats=stats,
        notes=notes,
    )


def _weights_to_orders(
    target: dict[str, float], account: AccountSnapshot, prices: dict[str, float],
    min_trade: float, decision,
) -> list[PlannedOrder]:
    """목표 비중 → 정수 주 주문. 매도를 먼저 배치한다(현금이 매수의 전제).

    ⚠️ 전량 청산은 최소 거래폭 밴드를 통과시킨다 — `should_trade` 가 그렇게 만들어져
       있다. 밴드가 청산을 막으면 손절이 무력화된다.
    """
    equity = account.equity
    reasons = {s.code: s.reason for s in decision.signals}
    sells: list[PlannedOrder] = []
    buys: list[PlannedOrder] = []

    for code, w_new in sorted(target.items()):
        price = prices.get(code)
        if not price or price <= 0:
            continue
        h = account.holdings.get(code)
        have = h.quantity if h else 0
        w_old = account.weight_of(code)

        if not should_trade(w_old, w_new, min_trade):
            continue

        want = int((w_new * equity) // price) if equity > 0 else 0
        delta = want - have
        if delta == 0:
            continue

        reason = reasons.get(code) or (
            decision.blocked.get(code) or ("목표 비중 0 — 청산")
        )
        if delta < 0:
            # 미결제분은 브로커가 거부한다. 팔 수 있는 만큼만 낸다.
            qty = min(-delta, h.sellable if h else 0)
            if qty > 0:
                sells.append(PlannedOrder(code, SELL, qty, price, w_old, w_new, reason))
        else:
            buys.append(PlannedOrder(code, BUY, delta, price, w_old, w_new, reason))

    return sells + buys


def _signal_row(s: Signal, prices: dict[str, float], account: AccountSnapshot) -> dict:
    return {
        "code": s.code,
        "action": s.action.value,
        "target_weight": round(s.target_weight, 4),
        "current_weight": round(account.weight_of(s.code), 4),
        "confidence": round(s.confidence, 4),
        "price": prices.get(s.code),
        "held": s.code in account.holdings,
        "reason": s.reason,
    }


def _signal_stats(signals: list[Signal], decision, max_width: float, n_preds: int) -> dict:
    counts: dict[str, int] = {}
    for s in signals:
        counts[s.action.value] = counts.get(s.action.value, 0) + 1
    n = max(len(signals), 1)
    return {
        "n_candidates": n_preds,
        "abstain": counts.get(Action.ABSTAIN.value, 0),
        "hold": counts.get(Action.HOLD.value, 0),
        "buy": counts.get(Action.BUY.value, 0),
        "sell": counts.get(Action.SELL.value, 0),
        "abstain_rate": round(counts.get(Action.ABSTAIN.value, 0) / n, 4),
        "abstain_threshold": round(float(max_width), 5),
        "target_gross": round(sum(s.target_weight for s in decision.signals), 4),
        "blocked": len(decision.blocked),
    }


# ---------------------------------------------------------------- 실행
def execute_plan(
    broker: PaperBroker, plan: TradingPlan, *, dry_run: bool = True,
    refresh_cash_between_legs: bool = True,
) -> list[OrderResult]:
    """계획 → 실제 주문. **매도를 먼저 보내고 계좌를 다시 조회한 뒤 매수한다.**

    매도 대금이 당일 주문가능금액에 언제 반영되는지는 계좌 설정에 따라 다르다.
    추정해서 매수를 밀어넣으면 주문이 거부되거나(운 좋은 경우) 예상보다 많이
    사게 된다. 추정하지 않고 **다시 조회한 금액**만 쓴다.
    """
    results: list[OrderResult] = []
    sells = [o for o in plan.orders if o.side == SELL]
    buys = [o for o in plan.orders if o.side == BUY]

    for o in sells:
        results.append(broker.place_order(o.code, SELL, o.quantity,
                                          price=o.price, dry_run=dry_run))

    # 매수 예산. 계획 시점 금액으로 출발하고, 실제 매도를 낸 뒤에는 다시 조회한다.
    cash = float(plan.cash)
    if buys and sells and refresh_cash_between_legs and not dry_run:
        cash = float(broker.fetch_deposit().get("orderable", cash))
        log.info("매도 후 주문가능금액 재조회: %s원", f"{cash:,.0f}")

    for o in buys:
        qty = o.quantity
        affordable = int(cash // o.price) if o.price > 0 else 0
        if affordable < qty:
            log.warning("[%s] 주문가능금액 한도 — %d주 → %d주", o.code, qty, affordable)
            qty = max(affordable, 0)
        cash -= qty * o.price
        if qty <= 0:
            results.append(OrderResult(o.code, BUY, 0, o.price, "market", dry_run,
                                       error="주문가능금액 부족"))
            continue
        results.append(broker.place_order(o.code, BUY, qty,
                                          price=o.price, dry_run=dry_run))
    return results


def is_rebalance_day(state: TraderState, today: date, rebalance_days: int) -> bool:
    """마지막 리밸런싱 이후 영업일이 주기를 채웠는가.

    주기를 지키는 이유는 백테스트와 같다 — 5일 예측을 매일 다시 실행하면
    같은 베팅이 겹쳐서 실린다.
    """
    if not state.last_rebalance:
        return True
    gap = int(np.busday_count(
        np.datetime64(state.last_rebalance, "D"), np.datetime64(today, "D")
    ))
    return gap >= int(rebalance_days)


def save_run(plan: TradingPlan, results: list[OrderResult], *, dry_run: bool) -> Path:
    """실행 기록. 절대 규칙 8 과 같은 이유로 덮어쓰지 않는다."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "dry_run": dry_run,
        "plan": plan.to_dict(),
        "orders_sent": [r.to_dict() for r in results],
    }
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = RUNS_DIR / f"run_{stamp}{'_dry' if dry_run else ''}.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                   encoding="utf-8")
    return out
