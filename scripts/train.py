#!/usr/bin/env python
"""Phase 1 학습 엔트리포인트.

로컬:
    python scripts/train.py --smoke        # 6종목 2epoch, 배관 점검용
    python scripts/train.py                # 전체

클라우드 GPU (Colab/Kaggle):
    notebooks/train_colab.ipynb 참고. 같은 명령이 그대로 돈다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.train import train  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.logging import get_logger, setup_logging  # noqa: E402

log = get_logger("train")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="소규모로 배관만 점검")
    ap.add_argument("--epochs", type=int, help="config 의 epochs 를 덮어쓴다")
    ap.add_argument("--batch-size", type=int, help="config 의 batch_size 를 덮어쓴다")
    ap.add_argument("--lr", type=float)
    args = ap.parse_args()

    setup_logging(run_name="train")
    cfg = load_config().raw

    if args.batch_size:
        cfg["training"]["batch_size"] = args.batch_size
    if args.lr:
        cfg["training"]["lr"] = args.lr

    report = train(cfg, smoke=args.smoke, max_epochs=args.epochs)

    log.info("최고 val loss %.6f (epoch %d)", report["best_val_loss"], report["best_epoch"])
    log.info("기준선 %.6f → 개선 %+.2f%%",
             report["baseline_val_loss"], report["improvement_vs_baseline_pct"])
    log.info("피처 중요도 상위 5:")
    for k, v in list(report["feature_importance"].items())[:5]:
        log.info("    %-14s %.4f", k, v)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
