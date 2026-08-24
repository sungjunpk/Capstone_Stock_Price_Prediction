"""로컬 parquet 저장소 — 증분(idempotent) 저장 전담.

규칙(CLAUDE.md): 같은 수집 명령을 두 번 돌려도 중복 행이 생기면 안 된다.
그래서 저장은 항상 "기존 읽기 → concat → 키 기준 dedup → 정렬 → 덮어쓰기".

레이아웃:
    data/raw/{kind}/{code}.parquet     예) data/raw/daily_chart/005930.parquet
    data/processed/{name}.parquet
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.config import PROJECT_ROOT
from src.utils.logging import get_logger

log = get_logger(__name__)

RAW_DIR = PROJECT_ROOT / "data" / "raw"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def _display(path: Path) -> str:
    """로그용 짧은 경로. 프로젝트 밖(테스트 tmpdir 등)이면 절대경로 그대로."""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def raw_path(kind: str, code: str) -> Path:
    return RAW_DIR / kind / f"{code}.parquet"


def processed_path(name: str) -> Path:
    return PROCESSED_DIR / f"{name}.parquet"


def read_parquet(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_parquet(path)


def upsert(
    df: pd.DataFrame,
    path: Path,
    *,
    key: list[str],
    sort_by: list[str] | None = None,
) -> pd.DataFrame:
    """기존 파일과 병합 후 key 기준 중복 제거하여 저장. 최종 DataFrame 반환.

    같은 key 가 겹치면 **새 데이터가 이긴다** (수정주가 소급 반영을 반영하기 위함).
    """
    if df is None or df.empty:
        log.debug("upsert 스킵 (빈 DataFrame): %s", path.name)
        return read_parquet(path) if path.exists() else pd.DataFrame()

    path.parent.mkdir(parents=True, exist_ok=True)
    existing = read_parquet(path)

    if existing is not None and not existing.empty:
        before = len(existing)
        merged = pd.concat([existing, df], ignore_index=True)
        merged = merged.drop_duplicates(subset=key, keep="last")
        added = len(merged) - before
    else:
        merged = df.drop_duplicates(subset=key, keep="last")
        added = len(merged)

    merged = merged.sort_values(sort_by or key).reset_index(drop=True)
    merged.to_parquet(path, index=False)
    log.info("%s: 총 %d행 (신규 %d행)", _display(path), len(merged), added)
    return merged


def last_date(path: Path, date_col: str = "date"):
    """증분 수집 시작점. 저장된 마지막 날짜(없으면 None)."""
    df = read_parquet(path)
    if df is None or df.empty or date_col not in df.columns:
        return None
    return pd.to_datetime(df[date_col]).max().date()


def load_kind(kind: str, codes: list[str] | None = None) -> pd.DataFrame:
    """한 종류의 raw 데이터를 종목별로 모아 하나의 long-format DataFrame 으로."""
    base = RAW_DIR / kind
    if not base.exists():
        return pd.DataFrame()

    frames = []
    for f in sorted(base.glob("*.parquet")):
        code = f.stem
        if codes is not None and code not in codes:
            continue
        part = pd.read_parquet(f)
        # stock_info 처럼 응답 자체에 code 가 들어있는 TR 도 있다.
        # 파일명이 정본이므로 기존 컬럼은 덮어쓰고, 항상 맨 앞에 둔다.
        part = part.drop(columns=["code"], errors="ignore")
        part.insert(0, "code", code)
        frames.append(part)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
