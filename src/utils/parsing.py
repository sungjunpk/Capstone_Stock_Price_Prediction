"""키움 REST API 응답 파싱 유틸.

키움 응답은 모든 값이 문자열로 온다:
    "+70,000"  → 70000       (부호는 등락 방향 표시일 뿐 값의 부호가 아닌 경우가 있음)
    "-1,234"   → -1234
    "1,234,567"→ 1234567
    ""/"-"     → None
숫자 변환을 개별 수집 코드에 흩어놓지 말고 전부 여기를 거치게 한다.
"""

from __future__ import annotations

import re
from datetime import date, datetime

import pandas as pd

_NUM_CLEAN = re.compile(r"[,\s%]")
_NULLISH = {"", "-", "--", "None", "null", "N/A"}


def to_float(value, *, abs_value: bool = False) -> float | None:
    """키움 문자열을 float로. 변환 불가면 None.

    Args:
        abs_value: True면 부호를 버린다. 키움은 가격 필드에도 등락 방향 부호를
            붙여주는 TR이 있어서(예: 현재가 "+70000"), 가격류는 abs_value=True로 쓴다.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return abs(float(value)) if abs_value else float(value)

    s = _NUM_CLEAN.sub("", str(value)).strip()
    if s in _NULLISH:
        return None
    if s.startswith("+"):
        s = s[1:]
    try:
        out = float(s)
    except ValueError:
        return None
    return abs(out) if abs_value else out


def to_int(value, *, abs_value: bool = False) -> int | None:
    f = to_float(value, abs_value=abs_value)
    return None if f is None else int(f)


def to_date(value) -> date | None:
    """'20240315' 또는 '2024-03-15' → date."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    s = str(value).strip().replace("-", "").replace("/", "")
    if len(s) != 8 or not s.isdigit():
        return None
    try:
        return datetime.strptime(s, "%Y%m%d").date()
    except ValueError:
        return None


def to_datetime(value) -> datetime | None:
    """'20260827114500' → datetime. 분봉 TR의 체결시각(cntr_tm) 형식이다.

    초 자리가 없는 '202608271145' 도 받는다 — 관측된 건 14자리뿐이지만
    길이 하나 때문에 전체 수집이 죽는 걸 막는다.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    s = str(value).strip().replace("-", "").replace(":", "").replace(" ", "")
    if s in _NULLISH:
        return None
    fmt = {14: "%Y%m%d%H%M%S", 12: "%Y%m%d%H%M"}.get(len(s))
    if fmt is None or not s.isdigit():
        return None
    try:
        return datetime.strptime(s, fmt)
    except ValueError:
        return None


def parse_records(
    records: list[dict],
    schema: dict[str, tuple[str, str]],
) -> pd.DataFrame:
    """TR 응답 레코드 리스트를 스키마대로 DataFrame으로 변환.

    schema: {출력컬럼명: (원본키, 타입)} — 타입은 'date' | 'datetime' | 'int'
            | 'float' | 'abs_int' | 'abs_float' | 'str'
    응답에 없는 키는 None으로 채운다(TR별로 필드가 빠지는 경우가 흔하다).
    """
    casters = {
        "date": to_date,
        "datetime": to_datetime,
        "int": to_int,
        "float": to_float,
        "abs_int": lambda v: to_int(v, abs_value=True),
        "abs_float": lambda v: to_float(v, abs_value=True),
        "str": lambda v: None if v is None or str(v).strip() in _NULLISH else str(v).strip(),
    }

    rows = []
    for rec in records:
        row = {}
        for out_col, (src_key, kind) in schema.items():
            if kind not in casters:
                raise ValueError(f"알 수 없는 타입: {kind} (컬럼 {out_col})")
            row[out_col] = casters[kind](rec.get(src_key))
        rows.append(row)

    return pd.DataFrame(rows, columns=list(schema.keys()))
