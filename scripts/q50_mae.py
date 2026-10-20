#!/usr/bin/env python
"""〈Table 4〉 표본에서 q50 의 점 예측 오차(MAE)를 낸다 — 핀볼 손실만으로는
"q50 의 점 오차는?" 이라는 질문에 답이 없다.

⚠️ 수집 자동화가 `data/processed/` 를 매일 다시 쓰므로 동결본으로 재야 한다.
    unzip outputs/train_bundle.zip -d /tmp/frozen
    python scripts/q50_mae.py --data /tmp/frozen/data/processed
"""
import argparse
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

import src.models.inference as inf  # noqa: E402
from src.models.inference import load_features, load_model, predict_split  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.logging import setup_logging  # noqa: E402

QUANTILES = [0.1, 0.5, 0.9]
QCOLS = ["q10", "q50", "q90"]
STEMS = {"ours": "phase1_b113237e",
         "tft": "phase1_f5e8f4e4_tft",
         "patchtst_q": "phase1_5263fac6_patchtst_q"}
CKPT = ROOT / "outputs" / "checkpoints"


def pinball(y, preds):
    loss = 0.0
    for i, tau in enumerate(QUANTILES):
        e = y - preds[:, i]
        loss = loss + np.maximum(tau * e, (tau - 1) * e)
    return float(np.mean(loss) / len(QUANTILES))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, help="panel/macro/static 이 있는 디렉터리 (동결본)")
    args = ap.parse_args()
    if args.data:
        inf.PROCESSED_DIR = args.data
    setup_logging(run_name="q50_mae")
    cfg = load_config()
    out = {"generated_at": datetime.now().isoformat(timespec="seconds"),
           "data": str(inf.PROCESSED_DIR),
           "note": "mae/rmse 는 q50 기준. baseline 은 학습구간 무조건부 분위수 상수.",
           "models": {}}
    for key, stem in STEMS.items():
        runs = []
        for seed in (42, 43, 44, 45, 46):
            hits = [h for h in sorted(CKPT.glob(f"{stem}*seed{seed}*.pt"))
                    if "smoke" not in h.name and "fp32" not in h.name]
            if not hits:
                print(f"!! {stem} seed{seed} 없음")
                continue
            loaded = load_model(hits[0])
            bundle = load_features(cfg, loaded)
            base_q = np.asarray(loaded.meta["baseline_quantiles"], dtype=float)
            r = {"checkpoint": hits[0].name, "seed": seed}
            for split in ("val", "test"):
                p, _ = predict_split(loaded, bundle, cfg, split)
                p = p.dropna(subset=["target"])
                y = p["target"].to_numpy(dtype=float)
                q = p[QCOLS].to_numpy(dtype=float)
                r[split] = {
                    "n": int(len(p)),
                    "pinball": round(pinball(y, q), 6),
                    "mae": round(float(np.mean(np.abs(y - q[:, 1]))), 6),
                    "rmse": round(float(np.sqrt(np.mean((y - q[:, 1]) ** 2))), 6),
                    "baseline_mae": round(float(np.mean(np.abs(y - base_q[1]))), 6),
                    "baseline_rmse": round(float(np.sqrt(np.mean((y - base_q[1]) ** 2))), 6),
                    "hit_rate": round(float(np.mean(np.sign(q[:, 1]) == np.sign(y))), 4),
                }
            print(key, seed, {s: (r[s]["mae"], r[s]["baseline_mae"], r[s]["hit_rate"])
                              for s in ("val", "test")}, flush=True)
            runs.append(r)
        agg = {}
        for split in ("val", "test"):
            for field in ("mae", "rmse", "hit_rate", "pinball", "baseline_mae"):
                v = [r[split][field] for r in runs]
                agg[f"{split}_{field}_mean"] = round(statistics.fmean(v), 6)
                if len(v) > 1:
                    agg[f"{split}_{field}_sd"] = round(statistics.stdev(v), 6)
            agg[f"{split}_mae_vs_baseline_pct"] = round(
                (agg[f"{split}_baseline_mae_mean"] - agg[f"{split}_mae_mean"])
                / agg[f"{split}_baseline_mae_mean"] * 100, 3)
        out["models"][key] = {"stem": stem, "runs": runs, "agg": agg}
        print("==", key, agg, flush=True)

    path = ROOT / "outputs" / "reports" / f"q50_mae_{datetime.now():%Y%m%d_%H%M%S}.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("wrote", path)


if __name__ == "__main__":
    main()
