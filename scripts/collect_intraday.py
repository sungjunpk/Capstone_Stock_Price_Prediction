#!/usr/bin/env python
"""분봉 수집 엔트리포인트 (증분) — 타점 탐지 트랙의 입력.

사용:
    python scripts/collect_intraday.py                # 60분봉, 유니버스 전체 + 지수
    python scripts/collect_intraday.py --codes 005930
    python scripts/collect_intraday.py --tic 30       # 30분봉으로
    python scripts/collect_intraday.py --dry-run      # 호출 없이 계획만

⚠️ **분봉 이력은 약 13개월 롤링이다** (실측 2026-08-27 기준 2025-08-01 까지).
   일봉처럼 "나중에 다시 받으면 된다"가 성립하지 않는다 — 안 받으면 사라진다.

장중에 돌려도 된다. 진행 중인 봉은 `drop_incomplete_bars` 가 버린다
(일봉 수집의 `--end-date` 경고에 해당하는 방어가 여기서는 자동이다).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import storage  # noqa: E402
from src.data.kiwoom.client import KiwoomAPIError, KiwoomClient  # noqa: E402
from src.data.kiwoom.collect import (  # noqa: E402
    collect_index_minute,
    collect_minute_chart,
    minute_kind,
)
from src.utils.config import load_config  # noqa: E402
from src.utils.logging import get_logger, setup_logging  # noqa: E402

log = get_logger("collect_intraday")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes", nargs="*", help="종목코드 (미지정 시 config 유니버스)")
    ap.add_argument("--tic", default="60",
                    choices=["1", "3", "5", "10", "15", "30", "45", "60"],
                    help="틱범위(분). 기본 60")
    ap.add_argument("--skip-index", action="store_true", help="지수 수집 생략")
    ap.add_argument("--full-refresh", action="store_true",
                    help="증분 조기종료 없이 받을 수 있는 만큼 전부 다시 받는다")
    ap.add_argument("--dry-run", action="store_true", help="API 호출 없이 계획만")
    args = ap.parse_args()

    setup_logging(run_name="collect_intraday")
    cfg = load_config()

    etfs = cfg["data"]["macro"].get("overseas_etf_fallback", [])
    codes = args.codes or (
        [u["code"] for u in cfg["data"]["universe"]] + [e["code"] for e in etfs]
    )
    kind = minute_kind(args.tic)

    if args.dry_run:
        log.info("[dry-run] %s분봉 — 종목 %d개", args.tic, len(codes))
        for code in codes:
            have = storage.last_timestamp(storage.raw_path(kind, code))
            log.info("  %s — 보유 마지막 봉: %s", code, have or "없음")
        return 0

    status: dict[str, str] = {}
    t0 = time.monotonic()

    with KiwoomClient() as client:
        for i, code in enumerate(codes, 1):
            try:
                df = collect_minute_chart(client, code, tic_scope=args.tic,
                                          full_refresh=args.full_refresh)
                status[code] = "ok"
                log.debug("%s: 누적 %d행", code, len(df))
            except KiwoomAPIError as exc:
                log.error("%s 수집 실패: %s", code, exc)
                status[code] = f"fail: {exc}"
            except Exception as exc:  # noqa: BLE001 — 한 종목이 전체를 죽이면 안 된다
                log.error("%s 예기치 못한 오류: %s", code, exc)
                status[code] = f"error: {exc}"

            if i % 10 == 0 or i == len(codes):
                elapsed = time.monotonic() - t0
                eta = elapsed / i * (len(codes) - i)
                log.info("진행 %d/%d — 경과 %.0fs, 남은 시간 약 %.0fs",
                         i, len(codes), elapsed, eta)

        if not args.skip_index:
            for idx in cfg["data"]["macro"]["indices"]:
                try:
                    collect_index_minute(client, idx["code"], tic_scope=args.tic,
                                         full_refresh=args.full_refresh)
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
