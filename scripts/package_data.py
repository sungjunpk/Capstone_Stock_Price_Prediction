#!/usr/bin/env python
"""클라우드 GPU 로 옮길 학습 데이터 묶음을 만든다.

수집은 로컬에서만 가능하고(키움 API + IP 등록), 학습은 GPU 가 필요하다.
그 사이를 잇는 게 이 스크립트다.

경량화 방법:
  - 학습에 안 쓰는 원본 OHLCV 컬럼 제거 (close 만 백테스트용으로 남긴다)
  - float64 → float32 (정밀도는 충분하고 용량은 절반)

사용:
    python scripts/package_data.py            # outputs/train_bundle.zip 생성
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.data.storage import PROCESSED_DIR  # noqa: E402
from src.training.dataset import dynamic_feature_columns  # noqa: E402
from src.utils.config import PROJECT_ROOT  # noqa: E402
from src.utils.logging import get_logger, setup_logging  # noqa: E402

log = get_logger("package_data")

# 학습에 필요 없는 원본 컬럼. close 는 백테스트에서 쓰므로 남긴다.
_DROP = ("open", "high", "low", "volume", "value")


def _shrink(df: pd.DataFrame) -> pd.DataFrame:
    out = df.drop(columns=[c for c in _DROP if c in df.columns])
    for c in out.columns:
        if out[c].dtype == "float64":
            out[c] = out[c].astype("float32")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/train_bundle.zip")
    args = ap.parse_args()

    setup_logging(run_name="package_data")

    panel_path = PROCESSED_DIR / "panel.parquet"
    if not panel_path.exists():
        log.error("panel.parquet 이 없다. scripts/build_features.py 를 먼저 실행할 것.")
        return 1

    out_zip = PROJECT_ROOT / args.out
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    staging = out_zip.parent / "_bundle"
    staging.mkdir(exist_ok=True)

    total_before = 0
    for name in ("panel", "macro", "static"):
        src = PROCESSED_DIR / f"{name}.parquet"
        if not src.exists():
            log.warning("%s.parquet 없음 — 건너뛴다", name)
            continue
        total_before += src.stat().st_size
        df = _shrink(pd.read_parquet(src))
        dst = staging / f"{name}.parquet"
        df.to_parquet(dst, index=False, compression="zstd")
        log.info("%-8s %6.1fMB → %5.1fMB  (%d행 × %d컬럼)",
                 name, src.stat().st_size / 1e6, dst.stat().st_size / 1e6,
                 len(df), df.shape[1])
        if name == "panel":
            log.info("  동적 피처 %d개", len(dynamic_feature_columns(df)))

    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(staging.glob("*.parquet")):
            z.write(f, arcname=f"data/processed/{f.name}")

    for f in staging.glob("*.parquet"):
        f.unlink()
    staging.rmdir()

    size = out_zip.stat().st_size
    log.info("생성: %s — %.1fMB (원본 %.0fMB 대비 %.0f%%)",
             out_zip.relative_to(PROJECT_ROOT), size / 1e6,
             total_before / 1e6, 100 * size / max(total_before, 1))
    log.info("이 파일을 Colab/Kaggle 로 올리면 된다. notebooks/train_colab.ipynb 참고")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
