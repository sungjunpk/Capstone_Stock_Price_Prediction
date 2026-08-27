"""백테스트 체결 규칙 — 최소 거래폭과 비용 집계.

잔챙이 거래를 막되 청산은 절대 막지 않는다. 이 불변식이 깨지면 손절이 무력화된다.
"""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.backtest import run_backtest, should_trade

BAND = 0.01


class TestShouldTrade:
    def test_full_exit_always_executes_even_below_band(self):
        """청산은 밴드보다 우선한다 — 안 그러면 팔지 못한 포지션이 영원히 남는다."""
        assert should_trade(0.005, 0.0, BAND) is True

    def test_small_reweight_is_skipped(self):
        assert should_trade(0.090, 0.085, BAND) is False

    def test_large_reweight_executes(self):
        assert should_trade(0.090, 0.070, BAND) is True

    def test_new_entry_below_band_is_skipped(self):
        assert should_trade(0.0, 0.005, BAND) is False

    def test_new_entry_above_band_executes(self):
        assert should_trade(0.0, 0.090, BAND) is True

    def test_no_change_is_never_a_trade(self):
        assert should_trade(0.09, 0.09, BAND) is False
        assert should_trade(0.0, 0.0, 0.0) is False


def _cfg(min_trade: float, exit_rank: int) -> dict:
    return {
        "features": {"return_horizon": 5},
        "backtest": {"initial_capital": 1.0, "execution_lag_days": 1,
                     "rebalance_days": 5},
        "trading": {
            "abstain": {"max_interval_width": 0.05, "percentile": 30},
            "direction": {"mode": "cross_sectional", "top_n": 10,
                          # 테스트 유니버스가 60종목이라 기권 30% 통과 시 생존이 ~18개다.
                          # 실제 설정값 20을 그대로 쓰면 전원 관망이 되어 아무것도 검증 못 한다.
                          "exit_rank": exit_rank, "min_candidates": 10,
                          "long_threshold": 0.004, "short_threshold": -0.004},
            "sizing": {"method": "rank_normalized", "exposure_scaling": False,
                       "max_position_pct": 0.10, "min_trade_weight": min_trade},
            "risk": {"max_gross_exposure": 0.90, "max_trades_per_day": 20,
                     "stop_loss_pct": -0.05, "take_profit_pct": 0.10},
            "costs": {"commission_bps": 1.5, "tax_bps": 18.0, "slippage_bps": 5.0},
        },
    }


@pytest.fixture(scope="module")
def panel():
    """종목마다 고유 변동성을 준다 — 실제 데이터의 변동성 군집성 때문에
    기권을 통과하는 종목 집합이 시점 간에 유지되고, 그래야 버퍼를 검증할 수 있다."""
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2024-01-01", periods=160)
    codes = [f"{i:03d}" for i in range(60)]

    vol = {c: abs(rng.normal(0.02, 0.006)) + 0.006 for c in codes}
    px = {c: 100 * np.exp(np.cumsum(rng.normal(0.0005, vol[c], len(dates))))
          for c in codes}
    prices = pd.DataFrame([{"code": c, "date": d, "close": px[c][i]}
                           for c in codes for i, d in enumerate(dates)])

    rows = []
    for i, d in enumerate(dates):
        for c in codes:
            fwd = px[c][min(i + 5, len(dates) - 1)] / px[c][i] - 1.0
            q50 = 0.02 * fwd + rng.normal(0, 0.004)
            w = vol[c] * 7.6 + abs(rng.normal(0, 0.01))
            rows.append({"code": c, "date": d, "q10": q50 - w / 2, "q50": q50,
                         "q90": q50 + w / 2, "target": fwd})
    return pd.DataFrame(rows), prices


def test_band_cuts_trade_count_without_gutting_turnover(panel):
    """밴드는 잔챙이만 걸러야 한다 — 체결 수는 줄지만 회전율은 거의 그대로."""
    preds, prices = panel
    off = run_backtest(preds, prices, _cfg(0.0, 20)).signal_stats
    on = run_backtest(preds, prices, _cfg(BAND, 20)).signal_stats

    assert on["n_trades"] < off["n_trades"]
    assert on["annual_turnover"] == pytest.approx(off["annual_turnover"], rel=0.15)


def test_buffer_reduces_turnover_and_extends_holding(panel):
    preds, prices = panel
    no_buf = run_backtest(preds, prices, _cfg(0.0, 10)).signal_stats
    buf = run_backtest(preds, prices, _cfg(0.0, 20)).signal_stats

    assert buf["annual_turnover"] < no_buf["annual_turnover"]
    assert buf["avg_holding_days"] > no_buf["avg_holding_days"]


def test_costs_are_measured_not_estimated(panel):
    """실지불 비용이 0보다 크고, 회전율 x 편도단가 범위 안에 있어야 한다."""
    preds, prices = panel
    s = run_backtest(preds, prices, _cfg(BAND, 20)).signal_stats

    assert s["total_cost_pct"] > 0
    # 편도 비용은 매수 6.5bp ~ 매도 24.5bp 사이다
    implied = s["annual_cost_pct"] / max(s["annual_turnover"], 1e-9)
    assert 0.00065 <= implied <= 0.00245


def test_blocked_reasons_are_categorised(panel):
    preds, prices = panel
    s = run_backtest(preds, prices, _cfg(BAND, 20)).signal_stats
    assert set(s["blocked_by_reason"]) <= {"손절", "익절", "거래한도", "공매도불가"}


def test_take_profit_removal_shows_up_in_reasons(panel):
    """익절을 끄면 익절 차단이 0이어야 한다 — 집계가 실제 동작을 반영하는지."""
    preds, prices = panel
    cfg = _cfg(BAND, 20)
    cfg["trading"]["risk"]["take_profit_pct"] = 99.0
    s = run_backtest(preds, prices, cfg).signal_stats
    assert s["blocked_by_reason"].get("익절", 0) == 0


# ---------------------------------------------------------------- 타점 탐지 모드
# 주기적 리밸런싱이 아니라 "조건이 맞을 때만 진입, 익절/손절/만료로만 청산".
# 검증 대상은 수익이 아니라 **동작**이다 — 안 팔아야 할 때 안 파는가.

def _event_cfg(*, hold_until_exit: bool, max_holding: int = 0) -> dict:
    cfg = _cfg(BAND, 20)
    cfg["trading"]["direction"]["mode"] = "absolute"
    cfg["backtest"]["rebalance_days"] = 1        # 매 봉 탐지
    cfg["backtest"]["hold_until_exit"] = hold_until_exit
    if max_holding:
        cfg["trading"]["risk"]["max_holding_bars"] = max_holding
    return cfg


def test_hold_until_exit_stops_churning_unsignaled_positions(panel):
    """신호가 사라졌다고 파는 건 정보가 아니라 노이즈에 반응하는 것이다."""
    preds, prices = panel
    churn = run_backtest(preds, prices, _event_cfg(hold_until_exit=False)).signal_stats
    hold = run_backtest(preds, prices, _event_cfg(hold_until_exit=True)).signal_stats

    assert hold["avg_holding_days"] > churn["avg_holding_days"]
    assert hold["annual_turnover"] < churn["annual_turnover"]
    assert hold["hold_until_exit"] is True


def test_expiry_bounds_holding_period(panel):
    """만료가 없으면 손익절에 안 걸린 포지션이 영원히 남는다."""
    preds, prices = panel
    forever = run_backtest(preds, prices, _event_cfg(hold_until_exit=True)).signal_stats
    capped = run_backtest(
        preds, prices, _event_cfg(hold_until_exit=True, max_holding=7)
    ).signal_stats

    assert capped["avg_holding_days"] < forever["avg_holding_days"]
    assert capped["blocked_by_reason"].get("보유만료", 0) > 0


def test_entry_count_matches_trades(panel):
    """진입 횟수가 이 트랙의 1차 판정 기준이라 집계가 정확해야 한다."""
    preds, prices = panel
    res = run_backtest(preds, prices, _event_cfg(hold_until_exit=True))
    from_zero = ((res.trades["from"] <= 1e-6) & (res.trades["to"] > 1e-6)).sum()

    assert res.signal_stats["n_entries"] == from_zero > 0
    assert res.signal_stats["annual_entries"] > 0


def test_bars_per_year_changes_annualisation(panel):
    """60분봉 Sharpe 를 252로 연율화하면 sqrt(7) 배 어긋난다."""
    preds, prices = panel
    daily = run_backtest(preds, prices, _event_cfg(hold_until_exit=True))

    cfg = _event_cfg(hold_until_exit=True)
    cfg["backtest"]["bars_per_year"] = 252 * 7
    intraday = run_backtest(preds, prices, cfg)

    ratio = intraday.metrics["sharpe"] / daily.metrics["sharpe"]
    assert ratio == pytest.approx(np.sqrt(7), rel=0.01)


# ------------------------------------------------- 예측 시작 이전 구간 제외
class TestWarmupExclusion:
    """가격이 예측보다 먼저 시작하는 구간을 연율화 분모에서 뺀다.

    모델은 lookback 이 차야 첫 예측을 낸다. 그 전 구간은 신호가 없어 강제로 현금인데,
    집계에 들어가면 Sharpe·CAGR 만 깎인다 — 매수후보유는 그동안 투자돼 있으므로
    전략만 불리한 비교가 된다.
    """

    @staticmethod
    def _data(n_days: int, pred_from: int):
        rng = np.random.default_rng(0)
        dates = pd.bdate_range("2024-01-01", periods=n_days).date
        codes = [f"{i:06d}" for i in range(30)]
        px = pd.DataFrame(
            {"date": np.repeat(dates, len(codes)),
             "code": np.tile(codes, len(dates)),
             "close": 100 * np.exp(np.cumsum(
                 rng.normal(0, 0.01, len(dates) * len(codes))))}
        )
        pd_dates = dates[pred_from:]
        preds = pd.DataFrame(
            {"date": np.repeat(pd_dates, len(codes)),
             "code": np.tile(codes, len(pd_dates))}
        )
        q = rng.normal(0.002, 0.01, len(preds))
        preds["q10"], preds["q50"], preds["q90"] = q - 0.05, q, q + 0.05
        return preds, px

    def test_returns_start_at_the_first_prediction(self):
        preds, px = self._data(200, 120)
        res = run_backtest(preds, px, _cfg(0.01, 20))
        assert min(res.returns.index) >= preds["date"].min()

    def test_dead_warmup_does_not_shorten_the_series_it_only_trims_the_front(self):
        """예측 구간이 같으면 앞에 가격을 더 붙여도 집계 길이가 변하면 안 된다."""
        short = run_backtest(*self._data(200, 120), _cfg(0.01, 20))
        # 같은 예측 구간(마지막 80일)이지만 가격이 40일 더 앞에서 시작한다
        preds, px = self._data(240, 160)
        long_ = run_backtest(preds, px, _cfg(0.01, 20))
        assert len(short.returns) == len(long_.returns)

    def test_no_predictions_fails_loudly(self):
        """예측 0건은 조용히 0% 수익으로 넘어가면 안 된다 — 설정 사고이기 때문."""
        _, px = self._data(60, 0)
        empty = pd.DataFrame(columns=["date", "code", "q10", "q50", "q90"])
        with pytest.raises(ValueError, match="관측된 폭"):
            run_backtest(empty, px, _cfg(0.01, 20))


def test_benchmark_window_matches_the_strategy_window():
    """벤치마크가 전략보다 긴 구간에서 재지면 비교가 전략 쪽으로 기운다."""
    from src.evaluation.backtest import buy_and_hold

    preds, px = TestWarmupExclusion._data(200, 120)
    res = run_backtest(preds, px, _cfg(0.01, 20))
    bench = buy_and_hold(px[px["date"] >= preds["date"].min()])
    assert len(bench) == len(res.returns)
