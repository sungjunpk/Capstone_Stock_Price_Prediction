#!/usr/bin/env python
"""데이터 수집 엔트리포인트 (증분).

사용:
    python scripts/collect.py                 # config 유니버스 전체
    python scripts/collect.py --codes 005930  # 특정 종목만
    python scripts/collect.py --dry-run       # 호출 없이 계획만 출력
    python scripts/collect.py --end-date 2026-08-25   # 그날까지만

⚠️ 장중에 --end-date 없이 돌리면 **오늘의 미완성 일봉**이 종가 자리에 들어간다.
   장 마감 전에 수집한다면 전 거래일을 --end-date 로 지정할 것.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.data import storage  # noqa: E402
from src.data.kiwoom.client import KiwoomClient  # noqa: E402
from src.data.kiwoom.collect import (  # noqa: E402
    collect_daily_chart,
    collect_index_daily,
    collect_universe,
)
from src.data.kiwoom.endpoints import unverified_specs  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.logging import get_logger, setup_logging  # noqa: E402

log = get_logger("collect")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes", nargs="*", help="종목코드 (미지정 시 config 유니버스)")
    ap.add_argument("--skip-index", action="store_true", help="지수 수집 생략")
    ap.add_argument(
        "--tr", nargs="*", default=["all"],
        choices=["all", "chart", "flow", "info"],
        help="수집할 TR 선택. flow(수급)는 종목당 30~50초로 느리다. 기본 all",
    )
    ap.add_argument("--end-date", help="이 날짜까지만 수집 (YYYY-MM-DD). "
                                       "장중 실행 시 미완성 일봉을 막는다")
    ap.add_argument("--dry-run", action="store_true", help="API 호출 없이 계획만")
    args = ap.parse_args()

    setup_logging(run_name="collect")
    cfg = load_config()

    universe = cfg["data"]["universe"]
    # 국내상장 해외지수 ETF 도 일반 종목이라 daily_chart 로 받는다.
    # (키움에 해외 일봉 TR 이 없어서 이게 글로벌 시퀀스의 대체재다 — docs/KIWOOM_VERIFY.md)
    etfs = cfg["data"]["macro"].get("overseas_etf_fallback", [])
    codes = args.codes or [u["code"] for u in universe] + [e["code"] for e in etfs]
    start_date = pd.Timestamp(cfg["data"]["start_date"]).date()
    end_date = pd.Timestamp(args.end_date).date() if args.end_date else None

    unverified = unverified_specs()
    if unverified:
        log.warning(
            "⚠️ 미검증 TR: %s — MCP 로 응답 스키마 확인 후 endpoints.py 갱신할 것 "
            "(docs/KIWOOM_VERIFY.md)",
            ", ".join(unverified),
        )

    if args.dry_run:
        log.info("[dry-run] 종목 %d개, 시작일 %s, 종료일 %s",
                 len(codes), start_date, end_date or "오늘")
        for code in codes:
            path = storage.raw_path("daily_chart", code)
            log.info("  %s — 보유 마지막일: %s", code, storage.last_date(path) or "없음")
        return 0

    # ETF 는 수급/기본정보 TR 이 의미 없거나 미제공이라 일봉만 받는다
    etf_codes = {e["code"] for e in etfs} - set(args.codes or [])
    stock_codes = [c for c in codes if c not in etf_codes]

    want = set(args.tr)
    do_all = "all" in want
    with_chart, with_flow, with_info = (
        do_all or "chart" in want,
        do_all or "flow" in want,
        do_all or "info" in want,
    )
    log.info("수집 TR: chart=%s flow=%s info=%s", with_chart, with_flow, with_info)

    with KiwoomClient() as client:
        status = collect_universe(
            client, stock_codes, start_date=start_date, end_date=end_date,
            with_chart=with_chart, with_flow=with_flow, with_info=with_info,
        )
        for code in sorted(etf_codes):
            log.info("[ETF] %s 일봉 수집", code)
            try:
                collect_daily_chart(client, code, start_date=start_date,
                                    end_date=end_date)
                status[code] = "ok"
            except Exception as exc:  # noqa: BLE001
                log.error("ETF %s 실패: %s", code, exc)
                status[code] = f"fail: {exc}"

        if not args.skip_index and with_chart:
            for idx in cfg["data"]["macro"]["indices"]:
                try:
                    collect_index_daily(client, idx["code"], start_date=start_date,
                                        end_date=end_date)
                    status[idx["name"]] = "ok"
                except Exception as exc:  # noqa: BLE001 — 지수 실패로 전체를 죽이지 않는다
                    log.error("지수 %s 실패: %s", idx["name"], exc)
                    status[idx["name"]] = f"fail: {exc}"

    failed = {k: v for k, v in status.items() if v != "ok"}
    log.info("완료: 성공 %d / 실패 %d", len(status) - len(failed), len(failed))
    for k, v in failed.items():
        log.error("  %s → %s", k, v)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
