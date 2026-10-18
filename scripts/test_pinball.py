#!/usr/bin/env python
"""네 구조의 **검정구간** 핀볼 손실 — <Table 4> 의 Test 열.

검증구간은 조기종료 판정에도 쓰이므로 그 값만으로 정확도를 주장하면 약하다.
Diebold-Mariano 검정을 원고에서 뺀 뒤 검정구간 비교가 사라지는 것을 메운다.

네 모형을 **같은 장비(MPS)·같은 패널**에서 평가하는 것이 요점이다. 제안 모형의
공개 체크포인트는 외부 GPU 혼합정밀도 실행이라 대조군(MPS)과 섞이므로,
`scripts/train.py --seed 42` 로 로컬에서 다시 학습한 것을 쓴다.

사용: python scripts/test_pinball.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scripts.backtest import find_checkpoint  # noqa: E402
from src.models.inference import load_features, load_model, predict_split  # noqa: E402
from src.utils.config import PROJECT_ROOT, load_config  # noqa: E402
from src.utils.logging import get_logger, setup_logging  # noqa: E402

log = get_logger("test_pinball")
QUANTILES = [0.1, 0.5, 0.9]
QCOLS = ["q10", "q50", "q90"]

MODELS = {
    "patchtst":   "outputs/checkpoints/phase1_7b693204_patchtst.pt",
    "patchtst_q": "outputs/checkpoints/phase1_5263fac6_patchtst_q.pt",
    "tft":        "outputs/checkpoints/phase1_f5e8f4e4_tft.pt",
    "ours_mps":   "outputs/checkpoints/phase1_b113237e_seed42.pt",
}


def pinball(p: pd.DataFrame) -> pd.Series:
    """세 분위의 **평균**. 학습 손실·<Table 4> 와 같은 척도다."""
    loss = 0.0
    for tau, col in zip(QUANTILES, QCOLS):
        e = p["target"] - p[col]
        loss = loss + np.maximum(tau * e, (tau - 1) * e)
    return loss / len(QUANTILES)


def main() -> int:
    setup_logging(run_name="test_pinball")
    cfg = load_config()
    out = {"generated_at": datetime.now().isoformat(timespec="seconds"), "models": {}}

    for name, ck in MODELS.items():
        loaded = load_model(find_checkpoint(ck, cfg))
        bundle = load_features(cfg, loaded)
        rec = {"checkpoint": Path(ck).name, "val_loss_at_save": float(loaded.val_loss)
               if getattr(loaded, "val_loss", None) is not None else None}
        for split in ("val", "test"):
            p, _ = predict_split(loaded, bundle, cfg, split)
            p = p.dropna(subset=["target"])
            rec[split] = {"n": int(len(p)), "pinball": round(float(pinball(p).mean()), 6)}
        out["models"][name] = rec
        log.info("%s 완료", name)

    # 무조건부 분위 기준선 — 학습구간 분위수로 세 상수를 내는 예측기
    base = out["baseline"] = {}
    loaded = load_model(find_checkpoint(MODELS["ours_mps"], cfg))
    bundle = load_features(cfg, loaded)
    from src.training.split import split_by_date
    tr = split_by_date(bundle.panel, bundle.spec)["train"]
    tcol = [c for c in tr.columns if c.startswith("target")][0]
    qs = np.quantile(tr[tcol].dropna(), QUANTILES)
    for split in ("val", "test"):
        p, _ = predict_split(loaded, bundle, cfg, split)
        p = p.dropna(subset=["target"]).copy()
        for col, q in zip(QCOLS, qs):
            p[col] = q
        base[split] = round(float(pinball(p).mean()), 6)

    path = PROJECT_ROOT / "outputs" / "reports" / f"test_pinball_{datetime.now():%Y%m%d_%H%M%S}.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n기준선  val {base['val']:.6f}   test {base['test']:.6f}\n")
    for name, r in out["models"].items():
        v, te = r["val"], r["test"]
        print(f"{name:<12} val {v['pinball']:.6f} ({(base['val']/v['pinball']-1)*100:+.2f}%)"
              f"   test {te['pinball']:.6f} ({(base['test']/te['pinball']-1)*100:+.2f}%)"
              f"   n={te['n']:,}")
    print(f"\n리포트: {path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
