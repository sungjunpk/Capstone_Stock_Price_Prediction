#!/usr/bin/env python
"""Diebold-Mariano 검정 — 두 모델의 예측 손실 차이가 유의한가.

논문의 중심 주장은 "TFT 와 대등한 정확도"다. 지금까지 근거는 핀볼 손실 두 개를
나란히 둔 것뿐이었고, 그 차이가 표본 변동보다 큰지는 재본 적이 없다.

DM 은 같은 표본 위의 손실 계열 차 d_t 를 보는 검정이라 **재학습이 필요 없다.**
타깃이 5일 앞을 보므로 인접 d_t 가 겹친다 — Newey-West 로 h-1=4 시차까지 보정한다.
횡단면 종속(같은 날 종목들이 시장 요인을 공유)은 날짜 단위로 평균 내어 흡수한다.

사용:
    python scripts/dm_test.py --a phase1_eadf265f.pt --b phase1_f5e8f4e4_tft.pt
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
from scipy import stats  # noqa: E402

from scripts.backtest import find_checkpoint  # noqa: E402
from src.models.inference import load_features, load_model, predict_split  # noqa: E402
from src.utils.config import PROJECT_ROOT, load_config  # noqa: E402
from src.utils.logging import get_logger, setup_logging  # noqa: E402

log = get_logger("dm_test")
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
QUANTILES = [0.1, 0.5, 0.9]
QCOLS = ["q10", "q50", "q90"]


def pinball(p: pd.DataFrame) -> pd.Series:
    """표본별 3분위 핀볼 손실 합."""
    loss = 0.0
    for tau, col in zip(QUANTILES, QCOLS):
        e = p["target"] - p[col]
        loss = loss + np.maximum(tau * e, (tau - 1) * e)
    return loss


def dm(d: np.ndarray, lag: int) -> dict:
    """Newey-West 보정 DM 통계량. d 는 날짜별 손실차 계열."""
    n = len(d)
    dbar = d.mean()
    dev = d - dbar
    gamma0 = (dev @ dev) / n
    var = gamma0
    for k in range(1, lag + 1):
        g = (dev[k:] @ dev[:-k]) / n
        var += 2 * (1 - k / (lag + 1)) * g       # Bartlett 가중
    se = np.sqrt(max(var, 1e-18) / n)
    stat = dbar / se
    # Harvey-Leybourne-Newbold 소표본 보정
    hln = np.sqrt((n + 1 - 2 * (lag + 1) + (lag + 1) * lag / n) / n)
    stat_hln = stat * hln
    return {
        "n_dates": int(n),
        "mean_loss_diff": float(dbar),
        "se": float(se),
        "dm_stat": float(stat),
        "p_value": float(2 * (1 - stats.norm.cdf(abs(stat)))),
        "dm_stat_hln": float(stat_hln),
        "p_value_hln": float(2 * (1 - stats.t.cdf(abs(stat_hln), df=n - 1))),
        "nw_lag": lag,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="모델 A 체크포인트 (제안 모형)")
    ap.add_argument("--b", required=True, help="모델 B 체크포인트 (대조군)")
    ap.add_argument("--nw-lag", type=int, default=4, help="Newey-West 시차. 지평 5일 → 4")
    args = ap.parse_args()

    setup_logging(run_name="dm_test")
    cfg = load_config()

    out = {"generated_at": datetime.now().isoformat(timespec="seconds"),
           "a": args.a, "b": args.b, "splits": {}}

    preds = {}
    for tag, ck in (("a", args.a), ("b", args.b)):
        loaded = load_model(find_checkpoint(ck, cfg))
        bundle = load_features(cfg, loaded)
        preds[tag] = {}
        for split in ("val", "test"):
            p, _ = predict_split(loaded, bundle, cfg, split)
            p = p.dropna(subset=["target"]).copy()
            p["loss"] = pinball(p)
            preds[tag][split] = p[["code", "date", "loss"]]
        log.info("%s 예측 완료 (%s)", tag.upper(), ck)

    for split in ("val", "test"):
        m = preds["a"][split].merge(preds["b"][split], on=["code", "date"],
                                    suffixes=("_a", "_b"))
        daily = m.groupby("date").agg(a=("loss_a", "mean"), b=("loss_b", "mean"))
        d = (daily["a"] - daily["b"]).to_numpy()
        r = dm(d, args.nw_lag)
        r["n_samples"] = int(len(m))
        r["mean_loss_a"] = float(m["loss_a"].mean())
        r["mean_loss_b"] = float(m["loss_b"].mean())
        out["splits"][split] = r
        print(f"\n[{split}] 공통 표본 {r['n_samples']:,}건 / {r['n_dates']}일")
        print(f"  평균 핀볼  A {r['mean_loss_a']:.6f}   B {r['mean_loss_b']:.6f}"
              f"   차 {r['mean_loss_diff']:+.6f}")
        print(f"  DM = {r['dm_stat']:+.3f}  (p = {r['p_value']:.3f})"
              f"   HLN 보정 {r['dm_stat_hln']:+.3f} (p = {r['p_value_hln']:.3f})")

    path = REPORTS_DIR / f"dm_test_{datetime.now():%Y%m%d_%H%M%S}.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n리포트: {path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
