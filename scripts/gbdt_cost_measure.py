#!/usr/bin/env python
"""분위 GBDT 의 계산 비용을 <Table 5> 와 같은 장비(Mac)에서 잰다.

`gbdt_quantile_compare.py` 가 조기종료로 정한 트리 수(75·25·275)로 세 분위 모형을
한 번씩 학습한 시간과, 201종목 1회 추론(136개 피처 × 201행, 세 모형) 시간의 중앙값을 잰다.
GBDT 는 에폭이 없으므로 학습 시간은 '전체 1회 학습'이고 CPU(전 코어)에서 돈다.

사용: python scripts/gbdt_cost_measure.py
"""

from __future__ import annotations

import json
import os
import pickle
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402

from scripts.gbdt_baseline import QUANTILES, build_table  # noqa: E402
from src.training.split import SplitSpec, split_by_date  # noqa: E402
from src.utils.config import PROJECT_ROOT, load_config  # noqa: E402

REPORTS = PROJECT_ROOT / "outputs" / "reports"
TREES = {0.1: 75, 0.5: 25, 0.9: 275}       # gbdt_quantile_compare_20261019_153401.json
LAGS = [1, 5, 20]
REPS = 21


def main() -> int:
    raw = load_config().raw
    table, feat_cols = build_table(raw, LAGS)
    parts = split_by_date(table, SplitSpec.from_config(raw))
    tr = parts["train"].dropna(subset=feat_cols + ["target"])
    te = parts["test"].dropna(subset=feat_cols + ["target"])
    Xtr, ytr = tr[feat_cols].to_numpy("float32"), tr["target"].to_numpy("float64")
    last = te["date"].max()
    Xday = te.loc[te["date"] == last, feat_cols].to_numpy("float32")

    models, fit_s = [], {}
    for q in QUANTILES:
        m = HistGradientBoostingRegressor(
            loss="quantile", quantile=q, learning_rate=0.03, max_leaf_nodes=31,
            l2_regularization=1.0, early_stopping=False, max_iter=TREES[q], random_state=42)
        t0 = time.perf_counter()
        m.fit(Xtr, ytr)
        fit_s[f"q{int(q * 100)}"] = round(time.perf_counter() - t0, 2)
        models.append(m)

    times = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        for m in models:
            m.predict(Xday)
        times.append(time.perf_counter() - t0)
    universe_ms = statistics.median(times) * 1000

    out = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "device": f"CPU, {os.cpu_count()} cores (sklearn OpenMP)",
        "train_rows": int(len(Xtr)), "features": len(feat_cols), "trees": TREES,
        "fit_seconds": fit_s, "fit_seconds_total": round(sum(fit_s.values()), 2),
        "universe_rows": int(len(Xday)), "universe_pass_ms": round(universe_ms, 2),
        "latency_per_stock_ms": round(universe_ms / len(Xday), 4),
        "model_bytes": len(pickle.dumps(models)),
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    path = REPORTS / f"gbdt_cost_measure_{datetime.now():%Y%m%d_%H%M%S}.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"리포트: {path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
