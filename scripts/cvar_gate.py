#!/usr/bin/env python
"""포트폴리오 CVaR 한도 판정 — 켤 값어치가 있는가. **읽기 전용이다.**

리스크 오버레이 1~5 단계는 전부 **종목 단위**다. 20종목이 전부 같이 무너지는
상황을 보는 눈이 없다. 6단계 CVaR 한도가 그 구멍을 메우는 층인데,
`configs/config.yaml` 에서 기본은 꺼져 있다(`cvar_limit: null`). 여기서 판정한다.

⚠️ **이 실험의 핵심은 통제군이다.**
   `exposure_scaling` 을 껐던 이유가 "위험하면 현금을 늘린다"는 판단의 **타이밍
   능력이 검증된 적 없다** 였다(실측: 평균 노출 61%→81%가 수익률 차이의 대부분).
   CVaR 게이트도 결국 노출을 줄인다. 그래서 게이트를 켠 런과 **같은 평균 노출**을
   상수로 고정한 런을 나란히 돌린다. 게이트가 통제군을 못 이기면 그건 꼬리를 보고
   타이밍을 잡은 게 아니라 그냥 덜 투자한 것이다.

채택 기준 (셋 다):
  1) val·test **양쪽**에서 baseline 대비 Sharpe 개선
  2) 양쪽에서 **동일 노출 통제군**보다 우위
  3) 바인딩률이 0% 도 100% 도 아님 — 상수가 아니라 게이트여야 한다

사용:
    python scripts/cvar_gate.py
    python scripts/cvar_gate.py --split test
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.backtest import find_checkpoint  # noqa: E402
from src.evaluation.backtest import run_backtest  # noqa: E402
from src.models.inference import (  # noqa: E402
    load_features,
    load_model,
    predict_split,
)
from src.utils.config import PROJECT_ROOT, load_config  # noqa: E402
from src.utils.logging import get_logger, setup_logging  # noqa: E402

log = get_logger("cvar_gate")
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"

# 한도 격자 두 벌. **절대 한도는 이미 기각됐다**(2026-09-03, 국면 간 이식 실패) —
# 상대 한도와 나란히 두는 건 그 기각이 재현되는지 같이 보기 위해서다.
#
#   절대: 일간 CVaR95 그 자체. 20종목 100% 노출의 자연 수준이 2~4% 근처다.
#   상대: 시장 균등배분(총 노출 1.0) 꼬리의 배수. 우리 책은 노출 0.81 에 20종목이라
#         집중 프리미엄을 감안하면 1.0 근처가 자연 수준이다.
LIMITS = (0.015, 0.020, 0.025, 0.030, 0.040)
RATIOS = (0.70, 0.80, 0.90, 1.00, 1.10)

VARIANTS = (
    [("cvar_limit", v, f"절대 {v:.3f}") for v in LIMITS]
    + [("cvar_limit_ratio", v, f"시장×{v:.2f}") for v in RATIOS]
)


def _with(base: dict, **risk) -> dict:
    cfg = copy.deepcopy(base)
    cfg["trading"]["risk"].update(risk)
    return cfg


def _row(res, label: str, kind: str, key: str | None = None) -> dict:
    m, s = res.metrics, res.signal_stats
    return {
        "run": label, "kind": kind, "key": key,
        "sharpe": m["sharpe"], "cagr": m["cagr"], "calmar": m["calmar"],
        "max_drawdown": m["max_drawdown"], "total_return": m["total_return"],
        "avg_gross_exposure": s["avg_gross_exposure"],
        "annual_turnover": s["annual_turnover"],
        "avg_cvar": s["avg_cvar"], "avg_cvar_ratio": s["avg_cvar_ratio"],
        "cvar_n": s["cvar_n"], "n_rebalances": s["n_rebalances"],
        "cvar_bind_rate": s["cvar_bind_rate"],
    }


def matched_control(preds, prices, cfg: dict, target: float, tries: int = 5):
    """평균 노출을 `target` 에 맞춘 정액 런.

    ⚠️ `max_gross_exposure` 를 target 으로 그냥 두면 안 된다. 그건 **상한**이라
    평균은 항상 그보다 낮게 나온다(실측: 상한 0.54 → 평균 0.43). 그러면 통제군이
    게이트보다 덜 투자한 상태로 비교돼 **게이트에 유리하게 기운다.**
    상한을 평균이 맞을 때까지 비례 조정한다.
    """
    gross = min(target, 1.0)
    res = None
    for _ in range(tries):
        res = run_backtest(preds, prices,
                           _with(cfg, cvar_limit=None, max_gross_exposure=gross))
        got = res.signal_stats["avg_gross_exposure"]
        if got <= 0 or abs(got - target) < 0.005 or gross >= 1.0:
            break
        gross = min(gross * target / got, 1.0)
    return res


def run_split(loaded, bundle, cfg: dict, split: str) -> dict:
    preds, prices = predict_split(loaded, bundle, cfg, split)

    # 1) baseline — 현재 전략 그대로
    base = run_backtest(preds, prices, _with(cfg, cvar_limit=None))
    rows = [_row(base, "baseline", "baseline")]

    # 2) 게이트를 켠 런 + 3) 그 런과 **같은 평균 노출**을 상수로 고정한 통제군
    for field, value, label in VARIANTS:
        gated = run_backtest(preds, prices, _with(cfg, **{field: value}))
        exposure = gated.signal_stats["avg_gross_exposure"]
        control = matched_control(preds, prices, cfg, exposure)
        rows.append(_row(gated, label, "gated", label))
        rows.append(_row(control, "  통제군(동일노출)", "control", label))

    return {"rows": rows, "n_days": base.metrics["n_days"]}


def _print(split: str, out: dict) -> None:
    print(f"\n=== {split} ({out['n_days']}일) ===")
    print(f"{'런':<22}{'Sharpe':>8}{'CAGR':>9}{'MDD':>9}"
          f"{'노출':>7}{'CVaR':>7}{'시장비':>7}{'측정':>7}{'바인딩':>8}")
    for r in out["rows"]:
        print(f"{r['run']:<22}{r['sharpe']:>8.2f}{r['cagr']:>9.1%}"
              f"{r['max_drawdown']:>9.1%}{r['avg_gross_exposure']:>7.2f}"
              f"{r['avg_cvar']:>7.3f}{r['avg_cvar_ratio']:>7.2f}"
              f"{r['cvar_n']:>4}/{r['n_rebalances']:<2}{r['cvar_bind_rate']:>8.1%}")


def verdict(report: dict) -> dict:
    """채택 기준 3개를 기계적으로 판정한다. 눈으로 고르면 test 에 맞추게 된다."""
    passed = []
    for _field, _value, label in VARIANTS:
        why = []
        for split, out in report["splits"].items():
            pick = {r["kind"]: r for r in out["rows"] if r["key"] == label}
            baseline = next(r for r in out["rows"] if r["kind"] == "baseline")
            gated, control = pick["gated"], pick["control"]
            if gated["sharpe"] <= baseline["sharpe"]:
                why.append(f"{split}: baseline 미달 "
                           f"({gated['sharpe']:.2f} ≤ {baseline['sharpe']:.2f})")
            if gated["sharpe"] <= control["sharpe"]:
                why.append(f"{split}: 통제군 미달 "
                           f"({gated['sharpe']:.2f} ≤ {control['sharpe']:.2f})")
            if not 0.0 < gated["cvar_bind_rate"] < 1.0:
                why.append(f"{split}: 바인딩 {gated['cvar_bind_rate']:.0%} — 게이트가 아니라 상수다")
        if why:
            log.info("%s 기각 — %s", label, " | ".join(why))
        else:
            passed.append(label)
    return {"adopted": passed,
            "conclusion": ("채택 후보 " + ", ".join(passed)) if passed
            else "전부 기각 — cvar_limit / cvar_limit_ratio 둘 다 null 로 둔다"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], action="append",
                    help="기본은 val + test 둘 다")
    ap.add_argument("--checkpoint", type=str, default=None)
    args = ap.parse_args()
    setup_logging(run_name="cvar_gate")

    cfg = load_config().raw
    loaded = load_model(find_checkpoint(args.checkpoint))
    bundle = load_features(cfg, loaded)

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "limits": list(LIMITS), "ratios": list(RATIOS),
        "cvar": {k: cfg["trading"]["risk"][k]
                 for k in ("cvar_alpha", "cvar_lookback", "cvar_min_obs")},
        "splits": {},
    }
    for split in args.split or ["val", "test"]:
        report["splits"][split] = run_split(loaded, bundle, cfg, split)
        _print(split, report["splits"][split])

    report["verdict"] = verdict(report)
    print(f"\n판정: {report['verdict']['conclusion']}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = REPORTS_DIR / f"cvar_gate_{stamp}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("리포트: %s", out.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
