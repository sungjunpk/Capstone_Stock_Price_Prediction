#!/usr/bin/env python
"""수집된 데이터 확인용. parquet 은 이진 파일이라 에디터로 못 여니 이걸로 본다.

사용:
    python scripts/peek.py                    # 전체 현황 요약
    python scripts/peek.py 005930             # 특정 종목 원본 데이터
    python scripts/peek.py features           # 가공된 피처 테이블
    python scripts/peek.py 005930 --rows 20   # 더 많이 보기
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.data.storage import PROCESSED_DIR, RAW_DIR  # noqa: E402


def _summary() -> int:
    """data/ 아래 뭐가 얼마나 있는지 한눈에."""
    found = False

    for kind_dir in sorted(RAW_DIR.glob("*")):
        if not kind_dir.is_dir():
            continue
        files = sorted(kind_dir.glob("*.parquet"))
        if not files:
            continue
        found = True
        print(f"\n[원본] data/raw/{kind_dir.name}/  — {len(files)}개 종목")
        for f in files:
            df = pd.read_parquet(f)
            span = (
                f"{df['date'].min()} ~ {df['date'].max()}"
                if "date" in df.columns
                else "날짜 컬럼 없음"
            )
            print(f"    {f.stem:10s} {len(df):>6,}행   {span}")

    for f in sorted(PROCESSED_DIR.glob("*.parquet")):
        found = True
        df = pd.read_parquet(f)
        codes = df["code"].nunique() if "code" in df.columns else "?"
        print(f"\n[가공] data/processed/{f.name}")
        print(f"    {len(df):,}행 × {df.shape[1]}컬럼, 종목 {codes}개")
        if "date" in df.columns:
            print(f"    기간 {df['date'].min()} ~ {df['date'].max()}")

    if not found:
        print("data/ 가 비어있다. 먼저 실행할 것:")
        print("    python scripts/collect.py")
        return 1

    print("\n자세히 보려면: python scripts/peek.py <종목코드|features>")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("target", nargs="?", help="종목코드(예: 005930) 또는 features")
    ap.add_argument("--rows", type=int, default=10, help="출력할 행 수 (기본 10)")
    ap.add_argument("--head", action="store_true", help="끝이 아니라 앞부분 보기")
    args = ap.parse_args()

    if not args.target:
        return _summary()

    # processed 먼저 찾고, 없으면 raw 에서 종목코드로 찾는다
    path = PROCESSED_DIR / f"{args.target}.parquet"
    if not path.exists():
        matches = list(RAW_DIR.glob(f"*/{args.target}.parquet"))
        if not matches:
            print(f"'{args.target}' 를 찾을 수 없다. 현황:")
            return _summary()
        path = matches[0]

    df = pd.read_parquet(path)
    print(f"파일: {path.relative_to(Path.cwd())}")
    print(f"크기: {len(df):,}행 × {df.shape[1]}컬럼")
    if "date" in df.columns:
        print(f"기간: {df['date'].min()} ~ {df['date'].max()}")

    na = df.isna().sum()
    if na.any():
        # 지표 워밍업 구간의 NaN 은 정상이다 (MA60 은 앞 60행이 비어있을 수밖에 없다)
        print(f"\n결측치 있는 컬럼: {dict(na[na > 0])}")

    print(f"\n{'앞' if args.head else '뒤'} {args.rows}행:")
    view = df.head(args.rows) if args.head else df.tail(args.rows)
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(view.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
