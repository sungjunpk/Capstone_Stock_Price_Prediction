#!/usr/bin/env python
"""raw 데이터 → 모델 입력 3종 (panel / macro / static).

look-ahead 방지: 라벨(forward return)은 마지막에 한 번만 붙이고,
피처 계산 함수에는 절대 넘기지 않는다.

사용:
    python scripts/build_features.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.data.storage import PROCESSED_DIR  # noqa: E402
from src.features.build import build_macro, build_panel, build_static  # noqa: E402
from src.training.split import SplitSpec  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.logging import get_logger, setup_logging  # noqa: E402

log = get_logger("build_features")


def _save(df: pd.DataFrame, name: str) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DIR / f"{name}.parquet"
    df.to_parquet(path, index=False)
    log.info("저장 %s — %d행 × %d컬럼", path.name, len(df), df.shape[1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-macro", action="store_true")
    args = ap.parse_args()

    setup_logging(run_name="build_features")
    cfg = load_config()
    spec = SplitSpec.from_config(cfg.raw)

    codes = [u["code"] for u in cfg["data"]["universe"]]
    log.info("대상 종목 %d개", len(codes))

    panel = build_panel(cfg.raw)
    _save(panel, "panel")
    log.info("panel 기간 %s ~ %s", panel["date"].min(), panel["date"].max())

    static = build_static(cfg.raw, spec.train_end)
    _save(static, "static")

    if not args.skip_macro:
        macro = build_macro(cfg.raw)
        _save(macro, "macro")
        log.info("macro 기간 %s ~ %s, 피처 %d개",
                 macro["date"].min(), macro["date"].max(), macro.shape[1] - 1)

    # 학습가능 행 수 요약 — 지표 워밍업과 라벨 결측을 뺀 실제 수
    base = {"code", "date", "open", "high", "low", "close", "volume", "value"}
    feat = [c for c in panel.columns if c not in base | {"target"}]
    usable = panel.dropna(subset=feat + ["target"])
    log.info("동적 피처 %d개 | 전체 %d행 → 학습가능 %d행",
             len(feat), len(panel), len(usable))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
