#!/usr/bin/env python
"""예측구간 진단 — 기권 로직이 딛고 선 구간이 실제로 맞는가.

이 프로젝트의 핵심 차별점은 "폭이 넓으면 기권"이다. 그런데 그 구간이 **실제로
얼마를 덮는지 한 번도 재본 적이 없다.** 세 가지를 묻는다. 전부 읽기 전용이다.

(1) 커버리지 — [q10, q90] 의 명목 커버리지는 80% 다. 실측은?
    분위회귀는 조건부 신호가 약하면 중앙값으로 수축한다(60분봉 트랙에서 실측:
    예측 p90 +0.0016 vs 실제 p90 +0.0580). 그러면 구간이 좁게 나오고,
    "확신한다"는 판단이 근거를 잃는다.

(2) Conformal 보정 (CQR, Romano et al. 2019)
    val 구간의 적합도 점수로 스칼라 하나를 잡아 구간을 넓히면 목표 커버리지가
    보장된다. **중요:** 가법이든 승법이든 보정은 모든 종목의 폭에 같은 방향으로
    작용하므로 **폭의 순서를 바꾸지 않는다.** 기권 임계값은 폭 분포의 백분위라서
    (`resolve_abstain_threshold`) 순서만 보존되면 선택이 그대로다.
    → 보정을 채택해도 **주문은 한 건도 바뀌지 않는다.** 진행 중인 실거래 기록에
      영향이 없다는 뜻이고, 이 스크립트가 그것까지 확인한다.

(3) 폭은 무엇을 맞히고 있나
    방향 예측력(랭크 IC)은 0.024 로 약하다. 그런데 같은 모델의 **폭**이 실현
    변동성을 얼마나 맞히는지는 따로 재야 한다. 폭의 IC 가 방향의 IC 보다 크면
    "이 모델은 방향 예측기가 아니라 불확실성 추정기"라는 뜻이고, 그건 순열검정
    결과(성과가 신호가 아니라 리스크 관리에서 나왔다)와 같은 곳을 가리킨다.

사용:
    python scripts/coverage.py
    python scripts/coverage.py --alpha 0.2      # 목표 커버리지 80%
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
from src.evaluation.metrics import rank_ic  # noqa: E402
from src.models.inference import load_features, load_model, predict_split  # noqa: E402
from src.trading.signal import resolve_abstain_threshold  # noqa: E402
from src.utils.config import PROJECT_ROOT, load_config  # noqa: E402
from src.utils.logging import get_logger, setup_logging  # noqa: E402

log = get_logger("coverage")
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"


def coverage_stats(p: pd.DataFrame) -> dict:
    """[q10, q90] 이 실제로 얼마를 덮는가. 명목 80%."""
    y = p["target"]
    return {
        "n": int(len(p)),
        "coverage": round(float(((y >= p["q10"]) & (y <= p["q90"])).mean()), 4),
        "below_q10": round(float((y < p["q10"]).mean()), 4),
        "above_q90": round(float((y > p["q90"]).mean()), 4),
        "width_mean": round(float((p["q90"] - p["q10"]).mean()), 5),
        "realized_p10_p90": round(float(y.quantile(0.9) - y.quantile(0.1)), 5),
    }


def cqr_scale(cal: pd.DataFrame, alpha: float) -> float:
    """승법 CQR 보정계수 — val 구간에서 잡는다.

    적합도 점수 E_i = max(q10-y, y-q90) / (q90-q10) 의 (1-alpha) 분위.
    구간을 중앙 기준으로 (1 + 2E*) 배 넓히면 목표 커버리지가 된다.
    """
    w = (cal["q90"] - cal["q10"]).replace(0, np.nan)
    e = pd.concat([(cal["q10"] - cal["target"]) / w,
                   (cal["target"] - cal["q90"]) / w], axis=1).max(axis=1).dropna()
    n = len(e)
    q = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)   # 유한표본 보정
    return float(e.quantile(q))


def apply_cqr(p: pd.DataFrame, e: float) -> pd.DataFrame:
    """중앙(q50)은 그대로 두고 폭만 넓힌다."""
    out = p.copy()
    half = (p["q90"] - p["q10"]) / 2.0
    mid = (p["q90"] + p["q10"]) / 2.0
    out["q10"] = mid - half * (1 + 2 * e)
    out["q90"] = mid + half * (1 + 2 * e)
    return out


def width_predictive_power(p: pd.DataFrame) -> dict:
    """폭이 실현 변동성을 맞히는가 — 방향 IC 와 같은 함수로 잰다.

    `rank_ic` 는 q50 을 점수로 보므로 폭을 그 자리에 넣고, 타깃은 |실현수익|
    (5일 지평의 실현 변동 크기) 으로 바꾼다.
    """
    sub = pd.DataFrame({
        "date": p["date"],
        "q50": p["q90"] - p["q10"],
        "target": p["target"].abs(),
    })
    return rank_ic(sub)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint")
    ap.add_argument("--alpha", type=float, default=0.2,
                    help="1-alpha 가 목표 커버리지. 0.2 → 80% ([q10,q90] 의 명목값)")
    args = ap.parse_args()

    setup_logging(run_name="coverage")
    cfg = load_config()

    ckpt = find_checkpoint(args.checkpoint, cfg)
    loaded = load_model(ckpt)
    bundle = load_features(cfg, loaded)

    splits = {}
    for name in ("val", "test"):
        preds, _ = predict_split(loaded, bundle, cfg, name)
        splits[name] = preds.dropna(subset=["target"]).reset_index(drop=True)

    val, test = splits["val"], splits["test"]

    # --- (1) 보정 전 커버리지
    before = {k: coverage_stats(v) for k, v in splits.items()}

    # --- (2) val 에서 보정계수를 잡아 test 에 적용
    e = cqr_scale(val, args.alpha)
    test_cal = apply_cqr(test, e)
    after = coverage_stats(test_cal)

    # 보정이 기권 선택을 바꾸는가 — 순서가 보존되면 안 바뀌어야 한다
    acfg = cfg["trading"]["abstain"]
    w_raw = test["q90"] - test["q10"]
    w_cal = test_cal["q90"] - test_cal["q10"]
    thr_raw = resolve_abstain_threshold(w_raw, acfg)
    thr_cal = resolve_abstain_threshold(w_cal, acfg)
    same = bool(((w_raw <= thr_raw) == (w_cal <= thr_cal)).all())
    rank_corr = float(w_raw.rank().corr(w_cal.rank()))

    # --- (3) 폭 vs 방향
    width_ic = {k: width_predictive_power(v) for k, v in splits.items()}
    dir_ic = {k: rank_ic(v) for k, v in splits.items()}

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "checkpoint": ckpt.name,
        "alpha": args.alpha,
        "coverage_before": before,
        "cqr_scale_from_val": round(e, 4),
        "coverage_after_test": after,
        "abstain_selection_unchanged": same,
        "width_rank_correlation": round(rank_corr, 6),
        "width_rank_ic": width_ic,
        "direction_rank_ic": dir_ic,
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = REPORTS_DIR / f"coverage_{stamp}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str),
                   encoding="utf-8")

    target = 1 - args.alpha
    print(f"\n체크포인트 {ckpt.name} | 목표 커버리지 {target:.0%}\n")
    print(f"  {'구간':>6} {'n':>8} {'커버리지':>9} {'q10 아래':>9} {'q90 위':>8} "
          f"{'평균 폭':>9} {'실제 p10~p90':>13}")
    for k, s in before.items():
        print(f"  {k:>6} {s['n']:>8,} {s['coverage']:>9.1%} {s['below_q10']:>9.1%} "
              f"{s['above_q90']:>8.1%} {s['width_mean']:>9.4f} "
              f"{s['realized_p10_p90']:>13.4f}")
    print(f"\n  CQR 보정계수(val 에서): {e:+.4f} → test 커버리지 "
          f"{before['test']['coverage']:.1%} → {after['coverage']:.1%} "
          f"(평균 폭 {before['test']['width_mean']:.4f} → {after['width_mean']:.4f})")
    print(f"  기권 선택 변화 없음: {same} (폭 순위상관 {rank_corr:.6f})")

    print("\n  방향 vs 폭 — 같은 모델, 같은 평가 함수")
    print(f"  {'구간':>6} {'방향 IC':>9} {'t':>7}   {'폭 IC':>9} {'t':>7}")
    for k in splits:
        d, w = dir_ic[k], width_ic[k]
        print(f"  {k:>6} {d['ic_mean']:>9.4f} {d['t_stat']:>7.2f}   "
              f"{w['ic_mean']:>9.4f} {w['t_stat']:>7.2f}")
    print(f"\n리포트: {out.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
