#!/usr/bin/env python
"""클라우드 GPU 로 옮길 학습 데이터 묶음을 만든다.

수집은 로컬에서만 가능하고(키움 API + IP 등록), 학습은 GPU 가 필요하다.
그 사이를 잇는 게 이 스크립트다.

경량화 방법:
  - 학습에 안 쓰는 원본 OHLCV 컬럼 제거 (close 만 백테스트용으로 남긴다)
  - float64 → float32 (정밀도는 충분하고 용량은 절반)

사용:
    python scripts/package_data.py                  # 일봉 기본 트랙
    python scripts/package_data.py --profile xs     # 횡단면 피처 트랙
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
from src.utils.config import PROJECT_ROOT, load_config  # noqa: E402
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
    ap.add_argument("--out")
    ap.add_argument("--profile", help="configs/config.yaml 의 profiles.<이름>")
    args = ap.parse_args()

    setup_logging(run_name="package_data")

    # 프로파일마다 산출물 접미사가 다르다 (panel_xs.parquet 등).
    # 묶음 안에서도 접미사를 **그대로 유지한다** — Kaggle 에서 같은
    # `--profile` 로 돌려야 체크포인트가 같은 태그로 저장되고, 일봉 운영
    # 체크포인트와 섞이지 않는다 (scripts/paper_trade.py 는 무태그만 받는다).
    cfg = load_config(profile=args.profile).raw
    sfx = cfg["data"].get("processed_suffix", "")

    out_zip = PROJECT_ROOT / (
        args.out or f"outputs/train_bundle{sfx or ''}.zip"
    )

    panel_path = PROCESSED_DIR / f"panel{sfx}.parquet"
    if not panel_path.exists():
        log.error("%s 이 없다. `scripts/build_features.py%s` 를 먼저 실행할 것.",
                  panel_path.name,
                  f" --profile {args.profile}" if args.profile else "")
        return 1
    log.info("프로파일 %s | 접미사 %r", args.profile or "(기본)", sfx)

    out_zip.parent.mkdir(parents=True, exist_ok=True)
    staging = out_zip.parent / "_bundle"
    staging.mkdir(exist_ok=True)

    total_before = 0
    for name in ("panel", "macro", "static"):
        src = PROCESSED_DIR / f"{name}{sfx}.parquet"
        if not src.exists():
            log.warning("%s.parquet 없음 — 건너뛴다", name)
            continue
        total_before += src.stat().st_size
        df = _shrink(pd.read_parquet(src))
        dst = staging / f"{name}{sfx}.parquet"
        df.to_parquet(dst, index=False, compression="zstd")
        log.info("%-12s %6.1fMB → %5.1fMB  (%d행 × %d컬럼)",
                 dst.name, src.stat().st_size / 1e6, dst.stat().st_size / 1e6,
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
    log.info("이 파일을 Kaggle 로 올린다. 절차는 docs/KAGGLE_SETUP.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
