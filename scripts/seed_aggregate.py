#!/usr/bin/env python
"""시드별 체크포인트를 모아 〈Table 4〉의 수치와 검정을 만든다 — 그 표의 출처 파일.

왜 필요한가: 〈Table 4〉의 5시드 평균·표준편차·Welch·부호검정은 캐글 세션에서 손으로
집계한 값이고 리포가 근거를 갖고 있지 않다. 재현 요청에 답할 수 없다.

무엇을 하는가: `outputs/checkpoints/{stem}*seed{N}*.pt` 를 시드마다 찾아
**같은 코드로** 검증·검정 두 구간의 핀볼 손실을 다시 계산한다. 학습 리포트에 seed 필드가
없어서(`src/training/train.py` 는 config 해시로만 이름을 만든다) 체크포인트를 기준으로 삼는다.
기준선은 같은 표본 위에서 학습구간 무조건부 분위수를 상수로 낸 예측기다 — 4.3절과 같은 정의다.

두 모형을 주면 시드를 짝지어 부호검정(단측)과 Welch 검정(양측)까지 낸다.

사용:
    # 캐글: 5시드 학습이 끝난 뒤
    python scripts/seed_aggregate.py --a phase1_b113237e --b phase1_f5e8f4e4 \\
        --seeds 42 43 44 45 46 --test --freeze outputs/reports/freeze_20261018_231605.json

    # 한 모형만 (검정 없이 집계만)
    python scripts/seed_aggregate.py --a phase1_b113237e --seeds 42
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

from src.models.inference import load_features, load_model, predict_split  # noqa: E402
from src.utils.config import PROJECT_ROOT, load_config  # noqa: E402
from src.utils.logging import get_logger, setup_logging  # noqa: E402

log = get_logger("seed_aggregate")
CKPT_DIR = PROJECT_ROOT / "outputs" / "checkpoints"
REPORTS = PROJECT_ROOT / "outputs" / "reports"
QUANTILES = [0.1, 0.5, 0.9]
QCOLS = ["q10", "q50", "q90"]


def pinball(target: np.ndarray, preds: np.ndarray) -> float:
    """세 분위의 **평균** 핀볼 손실. preds: (n, 3)"""
    loss = 0.0
    for i, tau in enumerate(QUANTILES):
        e = target - preds[:, i]
        loss = loss + np.maximum(tau * e, (tau - 1) * e)
    return float(np.mean(loss) / len(QUANTILES))


def find(stem: str, seed: int) -> Path | None:
    hits = sorted(CKPT_DIR.glob(f"{stem}*seed{seed}*.pt"))
    hits = [h for h in hits if "smoke" not in h.name]
    if not hits:
        return None
    if len(hits) > 1:
        log.warning("seed %d 후보가 여러 개다 — %s 를 쓴다", seed, hits[0].name)
    return hits[0]


def evaluate(path: Path, cfg: dict, *, with_test: bool) -> dict:
    """한 체크포인트의 구간별 핀볼 손실과 같은 표본의 상수 기준선."""
    loaded = load_model(path)
    bundle = load_features(cfg, loaded)
    base_q = np.asarray(loaded.meta["baseline_quantiles"], dtype=float)
    out = {"checkpoint": path.name,
           "val_loss_in_checkpoint": float(loaded.val_loss),
           "best_epoch": int(loaded.epoch)}
    for split in (("val", "test") if with_test else ("val",)):
        p, _ = predict_split(loaded, bundle, cfg, split)
        p = p.dropna(subset=["target"])
        y = p["target"].to_numpy(dtype=float)
        model = pinball(y, p[QCOLS].to_numpy(dtype=float))
        base = pinball(y, np.repeat(base_q[None, :], len(y), axis=0))
        out[split] = {
            "n": int(len(p)),
            "pinball": round(model, 6),
            "baseline": round(base, 6),
            "improvement_pct": round((base - model) / base * 100, 3),
        }
    return out


def summarize(runs: list[dict], split: str) -> dict:
    v = [r[split]["pinball"] for r in runs]
    imp = [r[split]["improvement_pct"] for r in runs]
    return {
        "n_seeds": len(v),
        "pinball_mean": round(statistics.fmean(v), 6),
        "pinball_sd": round(statistics.stdev(v), 6) if len(v) > 1 else None,
        "pinball_range": [round(min(v), 6), round(max(v), 6)],
        "improvement_pct_mean": round(statistics.fmean(imp), 3),
        "best_epoch_range": [min(r["best_epoch"] for r in runs),
                             max(r["best_epoch"] for r in runs)],
    }


def compare(a: list[dict], b: list[dict], split: str) -> dict:
    """a 가 b 보다 낮은가 — 시드를 짝지은 부호검정과 Welch 검정."""
    va = np.array([r[split]["pinball"] for r in a])
    vb = np.array([r[split]["pinball"] for r in b])
    wins = int((va < vb).sum())
    sign = stats.binomtest(wins, len(va), 0.5, alternative="greater")
    t = stats.ttest_ind(va, vb, equal_var=False)
    return {
        "a_wins": wins, "n_pairs": len(va),
        "sign_test_p_one_sided": round(float(sign.pvalue), 4),
        "welch_t": round(float(t.statistic), 3),
        "welch_df": round(float(t.df), 2),
        "welch_p_two_sided": round(float(t.pvalue), 4),
        "mean_gap": round(float(vb.mean() - va.mean()), 6),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="체크포인트 이름 앞부분 (예: phase1_b113237e)")
    ap.add_argument("--b", help="대조군. 주면 부호검정·Welch 검정까지 낸다")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--test", action="store_true",
                    help="검정구간도 산출한다 (추론만, 시드마다 1회)")
    ap.add_argument("--freeze", type=Path,
                    help="freeze_panel.py 기록 — 어느 자료로 낸 값인지 리포트에 박는다")
    args = ap.parse_args()

    setup_logging(run_name="seed_aggregate")
    cfg = load_config()
    splits = ("val", "test") if args.test else ("val",)

    models, missing = {}, {}
    for key, stem in (("a", args.a), ("b", args.b)):
        if stem is None:
            continue
        runs, gone = [], []
        for sd in args.seeds:
            path = find(stem, sd)
            if path is None:
                gone.append(sd)
                continue
            log.info("%s seed %d — %s", key, sd, path.name)
            runs.append({"seed": sd, **evaluate(path, cfg, with_test=args.test)})
        if not runs:
            print(f"❌ {stem} 에 해당하는 체크포인트가 없다 (찾은 시드 0개)")
            return 1
        models[key] = {"stem": stem, "runs": runs,
                       **{s: summarize(runs, s) for s in splits}}
        missing[key] = gone

    out = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "seeds_requested": args.seeds,
        "seeds_missing": {k: v for k, v in missing.items() if v},
        "splits": list(splits),
        "pinball": "세 분위 평균 (합/3)",
        "baseline": "학습구간 무조건부 분위수 세 상수, 같은 표본 위에서 평가",
        "models": models,
    }
    if args.freeze and args.freeze.exists():
        fz = json.loads(args.freeze.read_text(encoding="utf-8"))
        out["data"] = {"freeze_report": args.freeze.name,
                       "sha256": {k: v["sha256"] for k, v in fz["files"].items()},
                       "windows": fz["windows"]}
    if "b" in models:
        out["comparison"] = {s: compare(models["a"]["runs"], models["b"]["runs"], s)
                             for s in splits}

    REPORTS.mkdir(parents=True, exist_ok=True)
    path = REPORTS / f"seed_aggregate_{datetime.now():%Y%m%d_%H%M%S}.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    for s in splits:
        print(f"\n=== {s} ({models['a'][s]['n_seeds']}시드)")
        print(f"{'model':<22}{'pinball':>12}{'sd':>11}{'range':>24}"
              f"{'vs baseline':>13}{'best epoch':>12}")
        for m in models.values():
            u = m[s]
            sd = f"{u['pinball_sd']:.6f}" if u["pinball_sd"] is not None else "-"
            rng = f"{u['pinball_range'][0]:.6f}-{u['pinball_range'][1]:.6f}"
            print(f"{m['stem']:<22}{u['pinball_mean']:>12.6f}{sd:>11}{rng:>24}"
                  f"{u['improvement_pct_mean']:>+12.3f}%"
                  f"{str(u['best_epoch_range']):>12}")
        if "comparison" in out:
            c = out["comparison"][s]
            print(f"  a 승 {c['a_wins']}/{c['n_pairs']} (부호검정 p = "
                  f"{c['sign_test_p_one_sided']}, 단측)   "
                  f"Welch t = {c['welch_t']} (df {c['welch_df']}, "
                  f"p = {c['welch_p_two_sided']}, 양측)")
    if out["seeds_missing"]:
        print(f"\n⚠️ 못 찾은 시드: {out['seeds_missing']}")
    print(f"\n저장: {path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
