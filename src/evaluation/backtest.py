"""포트폴리오 백테스트 — 거래비용 반영.

CLAUDE.md 절대 규칙:
    신호 생성은 `src/trading/signal.py`, 리스크는 `src/trading/risk.py` 만 쓴다.
    모의투자와 **동일한 코드**다. 여기에 별도 분기를 만들지 않는다.

look-ahead 방지 (이 파일에서 제일 중요한 부분):
    - t 시점까지의 데이터로 만든 예측은 **t+lag 종가에 체결**된다(기본 lag=1).
      t 종가에 체결한다고 하면 방금 관측한 가격에 거래하는 셈이라 낙관적이다.
    - 리밸런싱 주기는 `backtest.rebalance_days` 가 정한다. 매일 리밸런싱하면
      예측 지평이 겹쳐 같은 베팅을 중복으로 싣게 된다.
      ⚠️ "지평과 맞춘다"는 근거는 2026-08-27 에 기각됐다 — 비용이 0 일 때만 옳다.
      Garleanu-Pedersen(2013): 최적 거래속도는 지평이 아니라 알파 감쇠 대비 비용이
      정한다. 실측으로 5일 → 10일로 바꿨다(configs/config.yaml 주석 참조).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.evaluation.metrics import (
    TRADING_DAYS,
    decile_spread,
    equity_curve,
    rank_ic,
    summarize,
)
from src.trading.risk import Position, apply_risk_overlay

# should_trade / one_way_cost 는 모의투자도 그대로 쓴다 —
# 백테스트에서 검증한 최소 거래폭 규칙이 실거래에서 달라지면 안 된다.
from src.trading.signal import (
    Action,
    QuantilePrediction,
    generate_signals,
    one_way_cost,
    resolve_abstain_threshold,
    round_trip_cost,
    should_trade,
)
from src.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class BacktestResult:
    returns: pd.Series                      # 일별 포트폴리오 수익률
    equity: pd.Series
    metrics: dict
    trades: pd.DataFrame
    signal_stats: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)   # 랭크 IC 등 예측력 진단


def run_backtest(
    predictions: pd.DataFrame,
    prices: pd.DataFrame,
    cfg: dict,
    *,
    allow_short: bool = False,
) -> BacktestResult:
    """
    predictions: code, date, q10, q50, q90  — date 시점 데이터로 만든 예측
    prices:      code, date, close
    """
    tcfg = cfg["trading"]
    bcfg = cfg.get("backtest", {})
    lag = int(bcfg.get("execution_lag_days", 1))
    rebalance_every = int(bcfg.get("rebalance_days", cfg["features"]["return_horizon"]))
    # 연율화 계수. 일봉이면 252, 60분봉이면 봉/년(252 x 7)을 넣는다.
    periods = float(bcfg.get("bars_per_year", TRADING_DAYS))
    # 타점 탐지 모드: 신호가 없는 보유분을 청산하지 않는다.
    #   기본(False) = 매 리밸런싱마다 포트폴리오를 갈아끼우는 일봉 트랙 동작
    #   True        = 진입은 조건이 맞을 때만, 청산은 익절/손절/보유만료로만
    hold_until_exit = bool(bcfg.get("hold_until_exit", False))
    costs = tcfg.get("costs", {})
    min_trade = float(tcfg["sizing"].get("min_trade_weight", 0.0))

    px = (
        prices.pivot_table(index="date", columns="code", values="close")
        .sort_index()
    )
    dates = list(px.index)
    preds_by_date = {d: g for d, g in predictions.groupby("date")}

    # 예측이 시작되기 전 구간은 **수익률 집계에서 뺀다.**
    # 모델은 lookback(120일)이 차야 첫 예측을 내는데, 가격은 그보다 앞에서 시작한다.
    # 그 구간은 신호가 없어 강제로 현금인데, 연율화 분모에는 들어가서 지표를 깎았다
    # (실측: 123일이 섞여 Sharpe 1.84→1.66, CAGR 24.0%→19.1%. 누적수익은 동일).
    # 매수후보유 벤치마크는 그 기간에도 투자돼 있으므로 **전략만 손해 보는 비교**였다.
    # dates 자체는 자르지 않는다 — signal_date = dates[i - lag] 가 첫 거래일에도
    # 하루 전을 제대로 짚어야 하기 때문이다.
    first_pred = predictions["date"].min() if len(predictions) else None
    start = 1
    if first_pred is not None:
        start = max(1, next((i for i, d in enumerate(dates) if d >= first_pred),
                            len(dates)))
    if start > 1:
        log.info("예측 시작 이전 %d일을 수익률 집계에서 제외 (%s 부터 집계)",
                 start - 1, dates[start] if start < len(dates) else "-")

    # 기권 임계값을 예측 폭 분포에서 확정한다.
    # 절대값을 미리 추측하면 거의 항상 틀린다 — 초기 추측 0.05 로는 기권률 95.8%,
    # 거래 0건이 나왔다(5일 수익률의 자연 폭은 0.124).
    widths = (predictions["q90"] - predictions["q10"]).to_numpy()
    max_width = resolve_abstain_threshold(widths, tcfg["abstain"])
    log.info(
        "기권 임계값 %.4f | 예측 폭 분포 p10=%.4f p50=%.4f p90=%.4f",
        max_width, *[float(pd.Series(widths).quantile(q)) for q in (0.1, 0.5, 0.9)],
    )

    # 거래가 안 나올 때 원인을 바로 알 수 있게 q50 분포도 남긴다.
    # 기권을 통과해도 q50 이 방향 임계값(= max(설정값, 왕복비용))을 못 넘으면 HOLD 다.
    q50 = predictions["q50"]
    long_th = max(float(tcfg["direction"]["long_threshold"]), round_trip_cost(costs))
    over = float((q50 > long_th).mean())
    log.info(
        "q50 분포 p10=%.4f p50=%.4f p90=%.4f | 매수 임계값 %.4f 초과 비율 %.1f%%",
        *[float(q50.quantile(q)) for q in (0.1, 0.5, 0.9)], long_th, 100 * over,
    )
    if over < 0.01:
        log.warning(
            "q50 이 매수 임계값을 거의 못 넘는다(%.1f%%) — 거래가 안 나올 것이다. "
            "모델이 조건부 신호를 못 찾았거나 임계값이 과하게 높다.", 100 * over,
        )

    positions: dict[str, Position] = {}
    cash_weight = 1.0
    turnovers: list[float] = []          # 리밸런싱마다 sum|Δw| — 비용에 직결된다
    gross_history: list[float] = []
    n_pos_history: list[int] = []
    holding_days: list[int] = []         # 청산 시점의 보유일수 — 버퍼가 먹혔는지 본다
    total_cost = 0.0                     # 실제 지불한 거래비용 누계 (추정 아님)
    blocked_reasons: dict[str, int] = {}
    daily_returns: list[float] = []
    ret_index: list = []
    trades: list[dict] = []
    stats = {"abstain": 0, "hold": 0, "buy": 0, "sell": 0, "blocked": 0, "decisions": 0}

    last_rebalance = -10**9

    for i in range(start, len(dates)):
        today, prev = dates[i], dates[i - 1]

        # --- 1) 보유 포지션의 오늘 수익률 반영
        port_ret = 0.0
        for code, pos in positions.items():
            p0, p1 = px.at[prev, code], px.at[today, code]
            if pd.notna(p0) and pd.notna(p1) and p0 > 0:
                port_ret += pos.weight * (p1 / p0 - 1.0)
        daily_returns.append(port_ret)
        ret_index.append(today)
        positions = {c: Position(c, p.weight, p.entry_price, p.days_held + 1)
                     for c, p in positions.items()}

        # --- 2) 리밸런싱 시점인가
        if i - last_rebalance < rebalance_every:
            continue
        # 예측은 lag 일 전 것을 쓴다 (오늘 종가 정보로 오늘 거래 금지)
        signal_date = dates[i - lag] if i - lag >= 0 else None
        if signal_date is None or signal_date not in preds_by_date:
            continue

        rows = preds_by_date[signal_date]
        today_px = {c: px.at[today, c] for c in px.columns if pd.notna(px.at[today, c])}

        preds = []
        for r in rows.itertuples():
            if r.code not in today_px:
                continue
            try:
                preds.append(QuantilePrediction(r.code, float(r.q10), float(r.q50), float(r.q90)))
            except ValueError as exc:      # 분위 교차 — 구조상 나오면 안 된다
                log.warning("분위 교차 무시: %s", exc)
        if not preds:
            continue

        # 보유분을 넘겨야 이력 버퍼가 동작한다 — 밀려난 종목을 바로 팔지 않는다
        sigs = generate_signals(preds, tcfg, max_width=max_width,
                                held=set(positions))
        stats["decisions"] += len(sigs)
        for s in sigs:
            stats[s.action.value] = stats.get(s.action.value, 0) + 1

        if hold_until_exit:
            # 보유 중인 종목의 신규 매수 신호는 **행동이 아니다** — 이 모드는
            # 보유분을 재조정하지 않기 때문이다(매 봉 재조정하면 타점과 무관한
            # 잔챙이 거래가 쌓이고, 그 비용이 이 트랙에서 이기려는 대상이다).
            # 오버레이에 넘기면 거래한도를 헛되이 잡아먹고, 총 익스포저 계산에서
            # '유지될 보유분'이 아니라 '새 주문'으로 세어져 한도가 어긋난다.
            sigs = [s for s in sigs
                    if not (s.code in positions and s.action is Action.BUY)]

        decision = apply_risk_overlay(
            sigs, positions, today_px, tcfg, allow_short=allow_short,
            liquidate_unsignaled=not hold_until_exit,
        )
        stats["blocked"] += len(decision.blocked)
        for reason, cnt in decision.blocked_by_reason.items():
            blocked_reasons[reason] = blocked_reasons.get(reason, 0) + cnt

        # --- 3) 목표 비중으로 이동 + 거래비용
        target = {s.code: s.target_weight for s in decision.signals}
        for c in decision.forced_exits:
            target[c] = 0.0
        for c in positions:
            # 신호가 없을 때: 일봉 트랙은 청산, 타점 탐지 트랙은 유지
            target.setdefault(c, positions[c].weight if hold_until_exit else 0.0)

        cost_today = 0.0
        turnover_today = 0.0
        new_positions: dict[str, Position] = {}
        for code, w_new in target.items():
            w_old = positions[code].weight if code in positions else 0.0
            delta = w_new - w_old

            # 잔챙이 거래를 막는다. 단 **전량 청산은 항상 통과**시킨다 —
            # 밴드가 청산을 막으면 손절이 무력화된다.
            is_exit = w_new <= 1e-6 and w_old > 1e-6
            if should_trade(w_old, w_new, min_trade):
                if abs(delta) > 1e-6:
                    cost_today += abs(delta) * one_way_cost(costs, selling=delta < 0)
                    turnover_today += abs(delta)
                    trades.append({
                        "date": today, "code": code, "from": round(w_old, 4),
                        "to": round(w_new, 4), "price": today_px.get(code),
                    })
                if is_exit:
                    holding_days.append(positions[code].days_held)
            else:
                w_new = w_old            # 거래하지 않으므로 기존 비중을 유지한다

            if w_new > 1e-6:
                entry = positions[code].entry_price if code in positions and w_old > 0 \
                    else today_px.get(code, 0.0)
                new_positions[code] = Position(code, w_new, entry,
                                               positions[code].days_held if code in positions else 0)

        positions = new_positions
        gross = sum(p.weight for p in positions.values())
        cash_weight = 1.0 - gross
        turnovers.append(turnover_today)
        gross_history.append(gross)
        n_pos_history.append(len(positions))
        total_cost += cost_today
        daily_returns[-1] -= cost_today       # 비용은 거래 당일에 반영
        last_rebalance = i

    returns = pd.Series(daily_returns, index=pd.Index(ret_index, name="date"))

    n = max(stats["decisions"], 1)
    n_entries = sum(1 for t in trades if t["from"] <= 1e-6 < t["to"])
    signal_stats = {
        **stats,
        "abstain_rate": round(stats.get("abstain", 0) / n, 4),
        "trade_rate": round((stats.get("buy", 0) + stats.get("sell", 0)) / n, 4),
        "n_trades": len(trades),
        "round_trip_cost": round(round_trip_cost(costs), 5),
        "cash_weight_end": round(cash_weight, 4),
        "abstain_threshold": round(float(max_width), 5),
        "width_p50": round(float(pd.Series(widths).median()), 5),
        "q50_p50": round(float(q50.median()), 5),
        "q50_p90": round(float(q50.quantile(0.9)), 5),
        "long_threshold": round(long_th, 5),
        "q50_over_threshold_rate": round(over, 4),
        # --- 실제로 포지션을 잡았는가. 이게 0 에 가까우면 성과 지표는 읽을 가치가 없다
        "mode": str(tcfg["direction"].get("mode", "absolute")),
        "top_n": int(tcfg["direction"].get("top_n", 0)),
        "exposure_scaling": bool(tcfg["sizing"].get("exposure_scaling", False)),
        "avg_gross_exposure": round(_mean(gross_history), 4),
        "avg_n_positions": round(_mean(n_pos_history), 2),
        "avg_turnover_per_rebalance": round(_mean(turnovers), 4),
        "annual_turnover": round(
            _mean(turnovers) * periods / max(rebalance_every, 1), 2
        ),
        "n_rebalances": len(turnovers),
        # --- 실제로 지불한 비용. 회전율 x 단가로 추정하지 않는다
        "total_cost_pct": round(total_cost, 5),
        "annual_cost_pct": round(total_cost * periods / max(len(returns), 1), 5),
        # --- 타점 탐지 트랙의 1차 판정 기준: 얼마나 드물게, 얼마나 정확하게 들어갔나
        "hold_until_exit": hold_until_exit,
        "n_entries": n_entries,
        "annual_entries": round(n_entries * periods / max(len(returns), 1), 1),
        "flat_rate": round(
            float(pd.Series(gross_history).le(1e-6).mean()) if gross_history else 0.0, 4
        ),
        "avg_holding_days": round(_mean(holding_days), 1),
        "exit_rank": int(tcfg["direction"].get("exit_rank", 0)),
        "min_trade_weight": round(min_trade, 4),
        "blocked_by_reason": dict(sorted(blocked_reasons.items())),
    }

    # --- 예측력 진단: 거래 규칙을 통째로 우회해 "방향을 맞히는가"만 직접 본다.
    # 성과가 안 나올 때 '모델이 못 맞힌 것'인지 '거래 규칙이 막은 것'인지 여기서 갈린다.
    diagnostics = {}
    if "target" in predictions.columns:
        diagnostics = {
            "rank_ic": rank_ic(predictions),
            "decile_spread": decile_spread(predictions),
        }
        ic = diagnostics["rank_ic"]
        log.info(
            "랭크 IC %.4f (t=%.2f, %d일, 양수비율 %.1f%%) | 십분위 스프레드 %.4f (t=%.2f)",
            ic["ic_mean"], ic["t_stat"], ic["n_dates"], 100 * ic["ic_positive_rate"],
            diagnostics["decile_spread"]["spread_mean"],
            diagnostics["decile_spread"]["t_stat"],
        )
        if abs(ic["t_stat"]) < 2.0:
            log.warning(
                "방향 예측력이 통계적으로 확인되지 않는다 (|t| < 2). "
                "매매 규칙을 아무리 손봐도 성과는 안 나온다 — 피처/타깃 설계 문제다."
            )

    if signal_stats["avg_n_positions"] < 1:
        log.warning(
            "평균 보유 종목이 1개 미만이다 — 사실상 현금만 들고 있었다. "
            "성과 지표(Sharpe 등)를 해석하지 말 것."
        )

    return BacktestResult(
        returns=returns,
        equity=equity_curve(returns, float(bcfg.get("initial_capital", 1.0))),
        metrics=summarize(returns, {
            "n_trades": len(trades),
            "annual_turnover": signal_stats["annual_turnover"],
        }, periods_per_year=periods),
        trades=pd.DataFrame(trades),
        signal_stats=signal_stats,
        diagnostics=diagnostics,
    )


def _mean(xs: list) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0


def buy_and_hold(prices: pd.DataFrame, codes: list[str] | None = None) -> pd.Series:
    """동일가중 매수후보유 — 비교 기준선."""
    px = prices.pivot_table(index="date", columns="code", values="close").sort_index()
    if codes:
        px = px[[c for c in codes if c in px.columns]]
    return px.pct_change(fill_method=None).mean(axis=1).fillna(0.0)
