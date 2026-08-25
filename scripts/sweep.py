#!/usr/bin/env python
"""정규화 강도를 바꿔가며 여러 설정을 한 번에 비교한다.

왜 필요한가:
    명목 학습샘플은 26만이지만 인접 윈도우가 99.2% 겹쳐 실질 독립 정보는
    거래일 수(약 1,900일) 수준이다. 파라미터 1.88M 은 여기에 과하다.
    한 설정씩 손으로 돌리면 GPU 시간이 아까우니 한 세션에서 비교한다.

각 설정은 짧게(기본 15 epoch) 돌리고 **기준선 대비 개선율**로 줄세운다.
승자를 고른 뒤 그 설정으로만 길게 학습하면 된다.

사용:
    python scripts/sweep.py                    # 기본 4개 설정
    python scripts/sweep.py --epochs 20
    python scripts/sweep.py --only tiny,small  # 일부만
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.train import train  # noqa: E402
from src.utils.config import PROJECT_ROOT, load_config  # noqa: E402
from src.utils.logging import get_logger, setup_logging  # noqa: E402

log = get_logger("sweep")

# 실질 정보량(~1,900 독립 사건)에 맞춰 용량을 줄이고 정규화를 올린 순서
PRESETS: dict[str, dict] = {
    "current": {  # 지금 설정 — 비교 기준
        "d_model": 128, "n_layers": 3, "d_ff": 256,
        "dropout": 0.2, "weight_decay": 1e-4, "lr": 3e-4,
    },
    "small": {
        "d_model": 64, "n_layers": 2, "d_ff": 128,
        "dropout": 0.3, "weight_decay": 1e-3, "lr": 3e-4,
    },
    "tiny": {
        "d_model": 48, "n_layers": 2, "d_ff": 96,
        "dropout": 0.4, "weight_decay": 1e-2, "lr": 2e-4,
    },
    "minimal": {
        "d_model": 32, "n_layers": 1, "d_ff": 64,
        "dropout": 0.5, "weight_decay": 1e-2, "lr": 2e-4,
    },
}


def apply_preset(cfg: dict, p: dict) -> dict:
    c = copy.deepcopy(cfg)
    enc = c["model"]["encoder"]
    enc["d_model"], enc["n_layers"] = p["d_model"], p["n_layers"]
    enc["d_ff"], enc["dropout"] = p["d_ff"], p["dropout"]
    c["model"]["vsn"]["dropout"] = p["dropout"]
    c["model"]["cross_attention"]["dropout"] = p["dropout"]
    # d_model 이 줄면 헤드 수도 나눠떨어져야 한다
    enc["n_heads"] = min(enc["n_heads"], max(1, p["d_model"] // 16))
    c["model"]["cross_attention"]["n_heads"] = min(
        c["model"]["cross_attention"]["n_heads"], max(1, p["d_model"] // 16)
    )
    c["training"]["weight_decay"] = p["weight_decay"]
    c["training"]["lr"] = p["lr"]
    # 과적합이 2 epoch 만에 오므로 오래 기다릴 이유가 없다
    c["training"]["warmup_epochs"] = 2
    c["training"]["early_stopping"]["patience"] = 5
    return c


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--only", help="쉼표로 구분한 프리셋 이름")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    setup_logging(run_name="sweep")
    base = load_config().raw

    names = args.only.split(",") if args.only else list(PRESETS)
    unknown = [n for n in names if n not in PRESETS]
    if unknown:
        log.error("모르는 프리셋: %s (가능: %s)", unknown, list(PRESETS))
        return 1

    results = []
    for i, name in enumerate(names, 1):
        p = PRESETS[name]
        log.info("=" * 62)
        log.info("[%d/%d] %s — d_model %d, layers %d, dropout %.1f, wd %.0e, lr %.0e",
                 i, len(names), name, p["d_model"], p["n_layers"],
                 p["dropout"], p["weight_decay"], p["lr"])
        log.info("=" * 62)

        cfg = apply_preset(base, p)
        cfg["training"]["batch_size"] = args.batch_size
        t0 = time.time()
        try:
            r = train(cfg, smoke=args.smoke, max_epochs=args.epochs)
        except Exception as exc:  # noqa: BLE001 — 한 설정이 죽어도 나머지는 돌린다
            log.error("%s 실패: %s", name, exc)
            continue

        results.append({
            "preset": name, **p,
            "n_params": r["n_params"],
            "baseline": r["baseline_val_loss"],
            "best_val": r["best_val_loss"],
            "improve_pct": r["improvement_vs_baseline_pct"],
            "best_epoch": r["best_epoch"],
            "minutes": round((time.time() - t0) / 60, 1),
        })

    if not results:
        log.error("성공한 설정이 없다")
        return 1

    results.sort(key=lambda r: -r["improve_pct"])

    print("\n" + "=" * 78)
    print("스윕 결과 — 기준선 대비 개선율 순")
    print("=" * 78)
    print(f"{'프리셋':<10}{'파라미터':>10}{'best val':>11}{'개선':>9}{'epoch':>7}{'분':>6}")
    for r in results:
        flag = "" if r["improve_pct"] > 0 else "  ❌기준선 미달"
        print(f"{r['preset']:<10}{r['n_params']/1e6:9.2f}M{r['best_val']:11.6f}"
              f"{r['improve_pct']:8.2f}%{r['best_epoch']:7d}{r['minutes']:6.1f}{flag}")

    win = results[0]
    print(f"\n승자: {win['preset']} (기준선 대비 {win['improve_pct']:+.2f}%)")
    if win["best_epoch"] <= 1:
        print("⚠️ 승자의 최고 epoch 이 1 이다 — 여전히 즉시 과적합. 더 줄이거나 정규화를 더 걸 것")

    out = PROJECT_ROOT / "outputs" / "reports" / f"sweep_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"저장: {out.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
