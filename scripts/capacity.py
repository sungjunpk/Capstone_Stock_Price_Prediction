#!/usr/bin/env python
"""운용 가능 자금(capacity) 추정 — 시장충격이 알파를 먹는 지점은 어디인가.

왜 '손익분기 운용자금'이 아니라 capacity 인가:
    국내 팩터투자 연구의 "거래비용을 고려한 손익분기 운용자금"은 **고정비**
    (데이터·시스템·인건비)를 회수하는 최소 자금을 묻는다. 그런데 이 봇은 고정비가
    사실상 0 이고, 한국 주식 비용은 전부 **비례(bps)** 다 — 수수료 1.5bp / 세금 18bp /
    슬리피지 5bp 어느 것도 자금이 커진다고 희석되지 않는다.
    즉 하한선은 없고, 실제로 존재하는 건 **상한**이다:
        "주문이 커져 시장충격이 알파를 다 먹는 자금 규모는 얼마인가."

무엇을 쓰는가:
    - 실제 백테스트가 낸 **체결 목록**(종목·날짜·비중변화). 가정한 회전율이 아니다.
    - 그 종목·그 날짜의 20일 평균 거래대금(ADV)과 20일 실현변동성.
      ⚠️ 패널의 `value` 는 **백만원 단위**다(005930 실측 대조로 확인).

충격 모형 (제곱근 법칙, Almgren et al. 2005 / Bouchaud):
    impact = eta x sigma_daily x sqrt(주문금액 / ADV)
    eta 는 1.0 을 기본으로 둔다. **정확한 값이 아니라 자릿수를 재는 것**이므로
    `--eta` 로 흔들어 보고 결론이 뒤집히는지 확인할 것.

무엇과 비교하는가 (판정선 두 개):
    1) 백테스트가 가정한 슬리피지(편도 5bp) — 이 선을 넘으면 **백테스트 가정이 깨진다**
    2) 순열검정이 실측한 신호 기여분(CAGR +1.7%p) — 이 선을 넘으면
       **모델이 만드는 초과분을 시장충격이 전부 먹는다**

사용:
    python scripts/capacity.py
    python scripts/capacity.py --eta 0.5 --split val
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scripts.backtest import find_checkpoint  # noqa: E402
from src.evaluation.backtest import run_backtest  # noqa: E402
from src.models.inference import load_features, load_model, predict_split  # noqa: E402
from src.utils.config import PROJECT_ROOT, load_config  # noqa: E402
from src.utils.logging import get_logger, setup_logging  # noqa: E402

log = get_logger("capacity")
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"

VALUE_UNIT = 1_000_000        # 패널 `value` 는 백만원 단위
AUM_GRID = [1e8, 5e8, 1e9, 5e9, 1e10, 5e10, 1e11, 5e11]

# 순열검정(2026-08-31)이 실측한 신호 기여분. 이걸 넘으면 예측이 만든 초과분을
# 시장충격이 전부 먹는다는 뜻이다.
SIGNAL_CAGR_CONTRIBUTION = 0.017


def trade_context(panel: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    """체결에 그 시점의 ADV·변동성을 붙인다. 둘 다 **과거 20일**만 본다."""
    p = panel[["code", "date", "value", "rvol_20"]].copy()
    p = p.sort_values(["code", "date"])
    p["adv"] = (
        p.groupby("code")["value"]
        .transform(lambda s: s.rolling(20, min_periods=10).mean())
        * VALUE_UNIT
    )
    out = trades.merge(p[["code", "date", "adv", "rvol_20"]], on=["code", "date"],
                       how="left")
    out["dw"] = (out["to"] - out["from"]).abs()
    return out.dropna(subset=["adv", "rvol_20"]).query("adv > 0 and dw > 0")


def impact_cost(ctx: pd.DataFrame, aum: float, eta: float, years: float) -> dict:
    """주어진 운용자금에서의 연환산 시장충격 비용(NAV 대비)."""
    notional = ctx["dw"] * aum
    participation = notional / ctx["adv"]
    impact = eta * ctx["rvol_20"] * np.sqrt(participation)   # 체결가 훼손 비율
    cost = float((impact * ctx["dw"]).sum() / years)         # NAV 대비 연환산
    return {
        "aum": aum,
        "participation_mean": float(participation.mean()),
        "participation_p95": float(participation.quantile(0.95)),
        "impact_bps_mean": float((impact * 1e4).mean()),
        "annual_impact_cost": cost,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--eta", type=float, default=1.0,
                    help="제곱근 충격 모형의 계수. 자릿수 확인용으로 흔들어 볼 것")
    args = ap.parse_args()

    setup_logging(run_name="capacity")
    cfg = load_config()

    ckpt = find_checkpoint(args.checkpoint, cfg)
    loaded = load_model(ckpt)
    bundle = load_features(cfg, loaded)
    preds, prices = predict_split(loaded, bundle, cfg, args.split)
    result = run_backtest(preds, prices, cfg)

    trades = result.trades
    if trades.empty:
        raise SystemExit("체결이 없다 — 백테스트 설정을 먼저 확인할 것.")

    days = len(result.returns)
    years = days / 252.0
    ctx = trade_context(bundle.raw_panel, trades)

    rows = [impact_cost(ctx, a, args.eta, years) for a in AUM_GRID]

    # 제곱근 법칙에서 비용은 sqrt(AUM) 에 비례한다 → 기준점 하나로 역산할 수 있다.
    ref = impact_cost(ctx, 1e9, args.eta, years)
    k = ref["annual_impact_cost"] / np.sqrt(1e9)

    def breakeven(target: float) -> float:
        return float((target / k) ** 2) if k > 0 else float("inf")

    # 백테스트가 가정한 슬리피지의 연환산 값 — 충격이 이걸 넘으면 가정이 깨진다
    slip = float(cfg["trading"]["costs"]["slippage_bps"]) / 1e4
    assumed_slippage_annual = result.signal_stats["annual_turnover"] * slip

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "checkpoint": ckpt.name,
        "split": args.split,
        "eta": args.eta,
        "n_trades": int(len(ctx)),
        "years": round(years, 2),
        "annual_turnover": result.signal_stats["annual_turnover"],
        "backtest_cagr": result.metrics["cagr"],
        "assumed_slippage_annual": round(assumed_slippage_annual, 5),
        "signal_cagr_contribution": SIGNAL_CAGR_CONTRIBUTION,
        "grid": rows,
        "breakeven_aum": {
            "vs_assumed_slippage": breakeven(assumed_slippage_annual),
            "vs_signal_contribution": breakeven(SIGNAL_CAGR_CONTRIBUTION),
            "vs_full_cagr": breakeven(result.metrics["cagr"]),
        },
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = REPORTS_DIR / f"capacity_{stamp}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str),
                   encoding="utf-8")

    def won(x: float) -> str:
        return f"{x/1e8:,.0f}억" if x < 1e12 else f"{x/1e12:,.1f}조"

    print(f"\n체결 {len(ctx):,}건 / {years:.1f}년 / 연 회전율 "
          f"{result.signal_stats['annual_turnover']:.1f} / eta={args.eta}\n")
    print(f"  {'운용자금':>10} {'평균 참여율':>12} {'p95 참여율':>12} "
          f"{'평균 충격':>10} {'연 충격비용':>12}")
    for r in rows:
        print(f"  {won(r['aum']):>10} {r['participation_mean']:>11.3%} "
              f"{r['participation_p95']:>11.3%} {r['impact_bps_mean']:>9.1f}bp "
              f"{r['annual_impact_cost']:>11.2%}")

    print(f"\n  백테스트가 가정한 연 슬리피지: {assumed_slippage_annual:.2%}")
    print(f"  → 충격이 그 가정을 넘는 자금: "
          f"{won(report['breakeven_aum']['vs_assumed_slippage'])}")
    print(f"  → 신호 기여분(CAGR +{SIGNAL_CAGR_CONTRIBUTION:.1%})을 다 먹는 자금: "
          f"{won(report['breakeven_aum']['vs_signal_contribution'])}")
    print(f"  → 백테스트 CAGR 전체({result.metrics['cagr']:.1%})를 먹는 자금: "
          f"{won(report['breakeven_aum']['vs_full_cagr'])}")
    print(f"\n리포트: {out.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
