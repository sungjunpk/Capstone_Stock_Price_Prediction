#!/usr/bin/env python
"""분위 GBDT 를 <Table 4>·<Table 6> 과 같은 표본·같은 지표로 잰다 (심사 대비 ②).

`gbdt_baseline.py` 는 랭크 IC·백테스트 비교라서 핀볼 손실의 표본이 Phase1 과 달랐다
(val 45,656 행 대 23,086 윈도우). 여기서는 GBDT 를 같은 방식으로 학습하되 예측을
제안 모형의 (code, date) 윈도우 표본에 맞춘 뒤,
  - val·test 핀볼 손실과 기준선 대비 개선 (<Table 4> 와 같은 기준선)
  - 기권 지표 A1~A6 (`abstain_quality.measure_preds` — 기권 규칙·통제군·부트스트랩 동일)
을 낸다. GBDT 는 무작위성이 없어(부표본 추출 없음) 한 번만 돌린다.

사용: python scripts/gbdt_quantile_compare.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scripts.abstain_quality import measure_preds, pinball  # noqa: E402
from scripts.backtest import find_checkpoint  # noqa: E402
from scripts.gbdt_baseline import (  # noqa: E402
    QUANTILES,
    _sort_quantiles,
    build_table,
    fit_quantile,
)
from src.models.inference import load_features, load_model, predict_split  # noqa: E402
from src.training.split import SplitSpec, split_by_date  # noqa: E402
from src.utils.config import PROJECT_ROOT, load_config  # noqa: E402
from src.utils.logging import get_logger, setup_logging  # noqa: E402

log = get_logger("gbdt_quantile_compare")
REPORTS = PROJECT_ROOT / "outputs" / "reports"
REF_CKPT = "outputs/checkpoints/phase1_b113237e_seed42.pt"   # 표본(윈도우)만 빌린다
QCOLS = ["q10", "q50", "q90"]
LAGS = [1, 5, 20]                                            # gbdt_baseline 기본값과 같다


def main() -> int:
    setup_logging(run_name="gbdt_quantile_compare")
    cfg = load_config()
    raw = cfg.raw

    # --- 1) GBDT 학습 — gbdt_baseline 과 같은 표·같은 초매개변수·날짜 기준 조기종료
    table, feat_cols = build_table(raw, LAGS)
    parts = split_by_date(table, SplitSpec.from_config(raw))
    usable = {k: v.dropna(subset=feat_cols + ["target"]) for k, v in parts.items()}
    X = {k: v[feat_cols].to_numpy("float32") for k, v in usable.items()}
    ytr = usable["train"]["target"].to_numpy("float64")
    yva = usable["val"]["target"].to_numpy("float64")
    base_q = np.quantile(ytr, QUANTILES)

    va_cols, te_cols, trees = [], [], {}
    for q in QUANTILES:
        pv, pt, n, _ = fit_quantile(q, X["train"], ytr, X["val"], yva, X["test"],
                                    chunk=25, max_iter=600, patience=4, seed=42,
                                    learning_rate=0.03, max_leaf=31, l2=1.0)
        va_cols.append(pv)
        te_cols.append(pt)
        trees[f"q{int(q * 100)}"] = n

    # --- 2) 제안 모형의 윈도우 표본에 맞춘다 (target·rvol_20 도 그쪽 것을 쓴다)
    loaded = load_model(find_checkpoint(REF_CKPT, cfg))
    bundle = load_features(cfg, loaded)
    res: dict = {}
    for split, cols in (("val", va_cols), ("test", te_cols)):
        g, crossed = _sort_quantiles(np.column_stack(cols))
        gb = usable[split][["code", "date", "target"]].copy()
        gb["date"] = pd.to_datetime(gb["date"])
        gb[QCOLS] = g
        ref, _ = predict_split(loaded, bundle, cfg, split)
        ref = ref.dropna(subset=["target"])
        ref["date"] = pd.to_datetime(ref["date"])
        m = ref.drop(columns=QCOLS).merge(gb, on=["code", "date"], how="inner",
                                          suffixes=("", "_gbdt"))
        if len(m) != len(ref):
            raise SystemExit(f"{split}: 표본 불일치 {len(m)} / {len(ref)}")
        gap = float((m["target"] - m["target_gbdt"]).abs().max())
        if gap > 1e-6:
            raise SystemExit(f"{split}: 목표변수 불일치 (최대 차 {gap})")
        m = m.drop(columns=["target_gbdt"])
        loss = float(pinball(m).mean())
        bl = float(pinball(m.assign(q10=base_q[0], q50=base_q[1], q90=base_q[2])).mean())
        res[split] = {"n": int(len(m)), "pinball": round(loss, 6), "baseline": round(bl, 6),
                      "improvement_pct": round(100 * (bl - loss) / bl, 3),
                      "crossings_sorted": crossed}
        log.info("%s n=%d pinball %.6f 기준선 %.6f (%+.2f%%)", split, len(m), loss, bl,
                 res[split]["improvement_pct"])
        if split == "test":
            test_preds = m

    out = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": "sklearn HistGradientBoostingRegressor(loss='quantile'), 분위별 1개",
        "features": {"n": len(feat_cols), "lags": LAGS},
        "trees_per_quantile": trees,
        "sample_from": REF_CKPT,
        "accuracy": res,
        "abstain": measure_preds(test_preds, "gbdt_quantile", dict(raw["trading"]["abstain"]),
                                 point=False),
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    path = REPORTS / f"gbdt_quantile_compare_{datetime.now():%Y%m%d_%H%M%S}.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("trees_per_quantile", "accuracy")}, indent=1))
    print(f"리포트: {path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
