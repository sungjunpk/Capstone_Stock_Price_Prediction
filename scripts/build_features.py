#!/usr/bin/env python
"""raw 일봉 → 기술적 지표 + 라벨 → data/processed/features.parquet

look-ahead 방지: 라벨(forward return)은 마지막에 한 번만 붙이고,
피처 계산 함수에는 절대 넘기지 않는다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.data import storage  # noqa: E402
from src.features.technical import (  # noqa: E402
    add_technical_features,
    drop_halted_days,
    forward_log_return,
)
from src.utils.config import load_config  # noqa: E402
from src.utils.logging import get_logger, setup_logging  # noqa: E402

log = get_logger("build_features")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="features", help="data/processed/{out}.parquet")
    args = ap.parse_args()

    setup_logging(run_name="build_features")
    cfg = load_config()
    feat_cfg = cfg["features"]
    horizon = int(feat_cfg["return_horizon"])

    # daily_chart 에는 유니버스 종목 외에 ETF(매크로용)도 섞여 있다.
    # 피처 테이블은 **예측 대상 종목만** 담는다 — ETF/지수는 매크로 시퀀스로 따로 붙인다.
    codes = [u["code"] for u in cfg["data"]["universe"]]
    raw = storage.load_kind("daily_chart", codes=codes)
    if raw.empty:
        log.error("data/raw/daily_chart 가 비었다. 먼저 scripts/collect.py 를 실행할 것.")
        return 1
    log.info("대상 종목 %d개: %s", len(codes), ", ".join(codes))

    frames = []
    total_halted = 0
    for code, part in raw.groupby("code", sort=True):
        part = part.sort_values("date").reset_index(drop=True)

        # 거래정지일 제거 — 지표 계산 전에 해야 rolling 창이 오염되지 않는다
        before = len(part)
        part = drop_halted_days(part)
        halted = before - len(part)
        total_halted += halted

        feats = add_technical_features(part, feat_cfg.get("technical", {}))
        feats["target"] = forward_log_return(feats["close"], horizon)
        frames.append(feats)
        log.info("%s: %d행 (거래정지 %d행 제거)", code, len(feats), halted)

    log.info("거래정지일 총 %d행 제거됨", total_halted)

    out = pd.concat(frames, ignore_index=True).sort_values(["date", "code"])
    path = storage.processed_path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)

    log.info("저장 %s — %d행 %d컬럼", path.name, len(out), out.shape[1])
    log.info("기간 %s ~ %s", out["date"].min(), out["date"].max())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
