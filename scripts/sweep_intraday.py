#!/usr/bin/env python
"""60분봉 타점 탐지 트랙 — 청산 규칙 스윕 + 구간별 예측력.

한 세션에서 예측을 **한 번만** 계산하고 규칙만 갈아끼운다.
결과는 `outputs/reports/` 에 남긴다 (CLAUDE.md 절대 규칙 8 — 덮어쓰지 않는다).

사용:
    python scripts/sweep_intraday.py                 # val + test 둘 다
    python scripts/sweep_intraday.py --split test

⚠️ **test 에서 고른 규칙은 검증이 아니다.** 같은 스윕을 val 에서도 돌려
   순위가 유지되는지 보는 것이 이 스크립트의 목적이다.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.evaluation.backtest import run_backtest  # noqa: E402
from src.evaluation.metrics import decile_spread, rank_ic  # noqa: E402
from src.models.inference import (  # noqa: E402
    load_features,
    load_model,
    predict_split,
)
from src.utils.config import PROJECT_ROOT, load_config  # noqa: E402
from src.utils.logging import get_logger, setup_logging  # noqa: E402

log = get_logger("sweep_intraday")

# 모의투자 계좌의 **실측** 비용. 백테스트 가정(편도 1.5bp)의 23배다.
# 실측 근거: outputs/paper_trading/fills.jsonl — 10,011,720 매수에 수수료 35,040.
MOCK_COSTS = {"commission_bps": 35.0, "tax_bps": 18.0, "slippage_bps": 5.0}

# 청산 규칙 격자. 진입 규칙(순위 top3 · 기권 10분위)은 고정한다 —
# 1차 스윕에서 진입 조건보다 **보유기간**이 성과를 훨씬 크게 갈랐기 때문이다.
HOLD_BARS = (7, 35, 70)          # 1일 / 5일 / 10일
TARGETS = ((0.03, -0.02), (0.05, -0.03), (0.10, -0.05))


def round_trip_returns(trades: pd.DataFrame) -> dict:
    """체결 로그 → 왕복 거래별 손익(진입가 대비 청산가).

    비용 차감 전 총수익이다. 이 값이 왕복비용을 못 넘으면 규칙이 틀린 것이다.
    """
    if trades.empty:
        return {"n": 0}
    rets, entry_px = [], {}
    for t in trades.sort_values("date").itertuples():
        w_from, w_to = getattr(t, "_3"), t.to
        if w_from <= 1e-6 < w_to:
            entry_px[t.code] = t.price
        elif w_to <= 1e-6 and t.code in entry_px:
            px0 = entry_px.pop(t.code)
            if px0 and t.price:
                rets.append(t.price / px0 - 1.0)
    if not rets:
        return {"n": 0}
    s = pd.Series(rets)
    return {
        "n": len(s),
        "mean": round(float(s.mean()), 5),
        "median": round(float(s.median()), 5),
        "win_rate": round(float((s > 0).mean()), 4),
    }


def _variant_cfg(base: dict, hold: int, tp: float, sl: float) -> dict:
    cfg = copy.deepcopy(base)
    d = cfg["trading"]["direction"]
    d["mode"] = "cross_sectional"
    d["top_n"] = d["exit_rank"] = 3
    d["min_candidates"] = 10
    cfg["trading"]["abstain"]["percentile"] = 10
    r = cfg["trading"]["risk"]
    r["max_holding_bars"], r["take_profit_pct"], r["stop_loss_pct"] = hold, tp, sl
    cfg["trading"]["costs"] = MOCK_COSTS
    return cfg


def run_split(loaded, bundle, cfg: dict, split: str) -> dict:
    preds, prices = predict_split(loaded, bundle, cfg, split)
    ic, ds = rank_ic(preds), decile_spread(preds)
    log.info("[%s] 랭크 IC %+.4f (t=%+.2f, %d봉) | 십분위 스프레드 %+.4f (t=%+.2f)",
             split, ic["ic_mean"], ic["t_stat"], ic["n_dates"],
             ds["spread_mean"], ds["t_stat"])

    rows = []
    for hold in HOLD_BARS:
        for tp, sl in TARGETS:
            res = run_backtest(preds, prices, _variant_cfg(cfg, hold, tp, sl))
            s, m = res.signal_stats, res.metrics
            by = s["blocked_by_reason"]
            rows.append({
                "max_holding_bars": hold, "take_profit": tp, "stop_loss": sl,
                "n_entries": s["n_entries"], "flat_rate": s["flat_rate"],
                "avg_holding_bars": s["avg_holding_days"],
                "round_trip": round_trip_returns(res.trades),
                "exits": {"익절": by.get("익절", 0), "손절": by.get("손절", 0),
                          "만료": by.get("보유만료", 0)},
                "sharpe": m["sharpe"], "cagr": m["cagr"],
                "annual_cost_pct": s["annual_cost_pct"],
            })
    return {"rank_ic": ic, "decile_spread": ds, "variants": rows}


def _print(split: str, out: dict) -> None:
    ic = out["rank_ic"]
    print(f"\n[{split}] 랭크 IC {ic['ic_mean']:+.4f} (t={ic['t_stat']:+.2f}, "
          f"{ic['n_dates']}봉) | 십분위 스프레드 "
          f"{out['decile_spread']['spread_mean']:+.4f}")
    print(f"  {'만료':>4}{'익절':>7}{'손절':>7}{'진입':>6}{'보유봉':>7}"
          f"{'왕복':>6}{'거래수익':>9}{'승률':>7}{'Sharpe':>8}{'CAGR':>8}{'비용/년':>8}")
    for v in out["variants"]:
        rt = v["round_trip"]
        mean = rt.get("mean")
        win = rt.get("win_rate")
        print(f"  {v['max_holding_bars']:>4}{v['take_profit']:>7.0%}"
              f"{v['stop_loss']:>7.0%}{v['n_entries']:>6}"
              f"{v['avg_holding_bars']:>7.1f}{rt.get('n', 0):>6}"
              f"{(mean if mean is not None else float('nan')):>9.2%}"
              f"{(win if win is not None else float('nan')):>7.1%}"
              f"{v['sharpe']:>8.2f}{v['cagr']:>8.1%}{v['annual_cost_pct']:>8.1%}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], action="append",
                    help="기본은 val, test 둘 다")
    ap.add_argument("--checkpoint")
    args = ap.parse_args()
    splits = args.split or ["val", "test"]

    setup_logging(run_name="sweep_intraday")
    cfg = load_config(profile="intraday").raw

    from scripts.backtest import find_checkpoint
    ckpt = find_checkpoint(args.checkpoint, cfg)
    loaded = load_model(ckpt)
    bundle = load_features(cfg, loaded)

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "checkpoint": ckpt.name,
        "costs": MOCK_COSTS,
        "note": "진입 규칙 고정(순위 top3 · 기권 10분위), 청산 규칙만 스윕",
        "splits": {},
    }
    for split in splits:
        report["splits"][split] = run_split(loaded, bundle, cfg, split)
        _print(split, report["splits"][split])

    out = PROJECT_ROOT / "outputs" / "reports" / (
        f"sweep_intraday_{datetime.now():%Y%m%d_%H%M%S}_{ckpt.stem}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {out.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
