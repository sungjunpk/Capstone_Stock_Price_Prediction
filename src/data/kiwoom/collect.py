"""TR 호출 → 파싱 → parquet 증분 저장을 잇는 수집 레이어.

여기 함수들은 전부 재실행 안전(idempotent). 이미 받은 구간은 다시 받지 않는다.
"""

from __future__ import annotations

import time
from datetime import date, timedelta

import pandas as pd

from src.data import storage
from src.data.kiwoom import endpoints as ep
from src.data.kiwoom.client import KiwoomAPIError, KiwoomClient
from src.utils.logging import get_logger
from src.utils.parsing import parse_records

log = get_logger(__name__)

# 마지막 저장일 이후 며칠을 겹쳐 다시 받을지(수정주가 소급 반영 대비)
_OVERLAP_DAYS = 5


def _records(data: dict, spec: ep.TRSpec) -> list[dict]:
    """응답 body 에서 레코드 배열을 꺼낸다. list_key 가 비면 body 자체가 레코드."""
    if not spec.list_key:
        return [data]
    value = data.get(spec.list_key)
    if value is None:
        log.warning("[%s] 응답에 list_key '%s' 없음 — 키: %s",
                    spec.name, spec.list_key, list(data)[:12])
        return []
    return value if isinstance(value, list) else [value]


def _collect_paged(
    client: KiwoomClient,
    spec: ep.TRSpec,
    body: dict,
    *,
    stop_before: date | None = None,
) -> pd.DataFrame:
    """연속조회를 돌며 파싱. stop_before 이전 날짜가 나오면 조기 종료(증분 수집)."""
    frames: list[pd.DataFrame] = []
    for page in client.paginate(spec, body):
        recs = _records(page, spec)
        if not recs:
            break
        df = parse_records(recs, spec.schema)
        frames.append(df)

        if stop_before is not None and "date" in df.columns:
            dates = df["date"].dropna()
            if not dates.empty and dates.min() <= stop_before:
                log.debug("[%s] 이미 보유한 구간 도달 — 조기 종료", spec.name)
                break

    if not frames:
        return pd.DataFrame(columns=list(spec.schema))
    return pd.concat(frames, ignore_index=True)


def collect_daily_chart(
    client: KiwoomClient,
    code: str,
    *,
    start_date: date | None = None,
    end_date: date | None = None,
    full_refresh: bool = False,
) -> pd.DataFrame:
    """일봉 OHLCV 증분 수집. 수정주가 기준."""
    spec = ep.DAILY_CHART
    path = storage.raw_path(spec.name, code)

    # 조기 종료 기준. 응답이 최신→과거 순이라 이 날짜에 닿으면 더 받을 필요가 없다.
    #  - 재수집(증분): 이미 보유한 구간에 닿으면 중단
    #  - 최초 수집:    start_date 아래로는 버릴 데이터이므로 중단
    #                  (삼성전자는 1985년까지 19페이지가 오는데 대부분 불필요)
    stop_before = start_date
    if not full_refresh:
        have_until = storage.last_date(path)
        if have_until is not None:
            stop_before = max(
                filter(None, [start_date, have_until - timedelta(days=_OVERLAP_DAYS)])
            )

    base_dt = (end_date or date.today()).strftime("%Y%m%d")
    body = {  # UNVERIFIED — 요청 필드명 확인 필요
        "stk_cd": code,
        "base_dt": base_dt,
        "upd_stkpc_tp": "1",  # 수정주가 반영. 끄면 액면분할 구간이 망가진다.
    }

    df = _collect_paged(client, spec, body, stop_before=stop_before)
    if df.empty:
        log.warning("[daily_chart] %s: 수집 결과 없음", code)
        return df

    df = df.dropna(subset=["date"])
    if start_date is not None:
        df = df[df["date"] >= start_date]
    if end_date is not None:
        # base_dt 를 줘도 응답에 그 뒤 날짜가 섞여 올 수 있다. 장중 미완성 봉을 막는 게
        # 목적이라 여기서 한 번 더 자른다.
        df = df[df["date"] <= end_date]
    return storage.upsert(df, path, key=["date"], sort_by=["date"])


def collect_investor_flow(
    client: KiwoomClient, code: str, *, start_date: date | None = None,
    end_date: date | None = None,
) -> pd.DataFrame:
    """외국인/기관/개인 순매수 증분 수집."""
    spec = ep.INVESTOR_FLOW
    path = storage.raw_path(spec.name, code)
    have_until = storage.last_date(path)
    stop_before = have_until - timedelta(days=_OVERLAP_DAYS) if have_until else None

    body = {  # UNVERIFIED
        "stk_cd": code,
        "dt": (end_date or date.today()).strftime("%Y%m%d"),
        "amt_qty_tp": "1",   # 1=수량
        "trde_tp": "0",      # 0=순매수
        "unit_tp": "1000",
    }
    df = _collect_paged(client, spec, body, stop_before=stop_before)
    if df.empty:
        return df
    df = df.dropna(subset=["date"])
    if start_date is not None:
        df = df[df["date"] >= start_date]
    if end_date is not None:
        df = df[df["date"] <= end_date]
    return storage.upsert(df, path, key=["date"], sort_by=["date"])


def collect_stock_info(client: KiwoomClient, code: str) -> pd.DataFrame:
    """종목 기본정보(업종/시총/PER/PBR) — static covariate 용. 스냅샷 누적."""
    spec = ep.STOCK_INFO
    data, _ = client.request(spec, {"stk_cd": code})  # UNVERIFIED
    df = parse_records(_records(data, spec), spec.schema)
    if df.empty:
        return df
    df.insert(0, "snapshot_date", date.today())
    return storage.upsert(
        df, storage.raw_path(spec.name, code),
        key=["snapshot_date"], sort_by=["snapshot_date"],
    )


def collect_index_daily(
    client: KiwoomClient, index_code: str, *, start_date: date | None = None,
    end_date: date | None = None,
) -> pd.DataFrame:
    """KOSPI/KOSDAQ 등 지수 일봉 — 매크로 시퀀스."""
    spec = ep.INDEX_DAILY
    path = storage.raw_path(spec.name, index_code)
    have_until = storage.last_date(path)
    stop_before = have_until - timedelta(days=_OVERLAP_DAYS) if have_until else None

    body = {  # UNVERIFIED
        "inds_cd": index_code,
        "base_dt": (end_date or date.today()).strftime("%Y%m%d"),
    }
    df = _collect_paged(client, spec, body, stop_before=stop_before)
    if df.empty:
        return df
    df = df.dropna(subset=["date"])
    if start_date is not None:
        df = df[df["date"] >= start_date]
    if end_date is not None:
        df = df[df["date"] <= end_date]
    return storage.upsert(df, path, key=["date"], sort_by=["date"])


def is_trading_day(client: KiwoomClient, on: date) -> bool:
    """오늘 장이 열렸는가. **공휴일 테이블을 만들지 않는다 — 데이터에 묻는다.**

    지수 일봉을 읽기 전용으로 1회 호출해 최신 봉 날짜가 `on` 인지 본다.
    휴장일이면 그날 봉이 아예 없다. 장중에 부르면 진행 중인 봉이 오므로
    개장 여부 판정에는 그대로 쓸 수 있다.

    ⚠️ 저장하지 않는다. 장중 미완성 봉이 raw 에 들어가면 안 되기 때문이다.
    """
    spec = ep.INDEX_DAILY
    data, _ = client.request(spec, {"inds_cd": "001",
                                    "base_dt": on.strftime("%Y%m%d")})
    df = parse_records(data.get(spec.list_key) or [], spec.schema)
    if df.empty:
        return False
    latest = pd.to_datetime(df["date"]).max()
    return bool(pd.notna(latest) and latest.date() == on)


def collect_universe(
    client: KiwoomClient,
    codes: list[str],
    *,
    start_date: date | None = None,
    end_date: date | None = None,
    with_chart: bool = True,
    with_flow: bool = True,
    with_info: bool = True,
) -> dict[str, str]:
    """유니버스 전체 수집. 한 종목이 실패해도 나머지는 계속 진행한다.

    TR별로 켜고 끌 수 있다. 수급(investor_flow)은 페이지당 100건이라
    종목당 30~50초가 걸리는 병목이므로, 급하지 않으면 나중에 따로 돌린다.
    """
    status: dict[str, str] = {}
    t0 = time.monotonic()

    for i, code in enumerate(codes, 1):
        try:
            if with_chart:
                collect_daily_chart(client, code, start_date=start_date,
                                    end_date=end_date)
            if with_flow:
                collect_investor_flow(client, code, start_date=start_date,
                                      end_date=end_date)
            if with_info:
                collect_stock_info(client, code)
            status[code] = "ok"
        except KiwoomAPIError as exc:
            log.error("%s 수집 실패: %s", code, exc)
            status[code] = f"fail: {exc}"
        except Exception as exc:  # noqa: BLE001 — 한 종목 때문에 전체가 죽으면 안 된다
            log.error("%s 예기치 못한 오류: %s", code, exc)
            status[code] = f"error: {exc}"

        # 긴 수집이라 진행률과 남은 시간을 주기적으로 알린다
        if i % 10 == 0 or i == len(codes):
            elapsed = time.monotonic() - t0
            eta = elapsed / i * (len(codes) - i)
            fails = sum(1 for v in status.values() if v != "ok")
            log.info(
                "진행 %d/%d (%.0f%%) | 실패 %d | 경과 %.0f분 | 남은시간 약 %.0f분",
                i, len(codes), 100 * i / len(codes), fails, elapsed / 60, eta / 60,
            )
    return status
