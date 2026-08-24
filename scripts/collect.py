#!/usr/bin/env python
"""데이터 수집 엔트리포인트 (증분).

사용:
    python scripts/collect.py                 # config 유니버스 전체
    python scripts/collect.py --codes 005930  # 특정 종목만
    python scripts/collect.py --dry-run       # 호출 없이 계획만 출력
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.data import storage  # noqa: E402
from src.data.kiwoom.client import KiwoomClient  # noqa: E402
from src.data.kiwoom.collect import collect_index_daily, collect_universe  # noqa: E402
from src.data.kiwoom.endpoints import unverified_specs  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.logging import get_logger, setup_logging  # noqa: E402

log = get_logger("collect")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes", nargs="*", help="종목코드 (미지정 시 config 유니버스)")
    ap.add_argument("--skip-index", action="store_true", help="지수 수집 생략")
    ap.add_argument("--dry-run", action="store_true", help="API 호출 없이 계획만")
    args = ap.parse_args()

    setup_logging(run_name="collect")
    cfg = load_config()

    universe = cfg["data"]["universe"]
    codes = args.codes or [u["code"] for u in universe]
    start_date = pd.Timestamp(cfg["data"]["start_date"]).date()

    unverified = unverified_specs()
    if unverified:
        log.warning(
            "⚠️ 미검증 TR: %s — MCP 로 응답 스키마 확인 후 endpoints.py 갱신할 것 "
            "(docs/KIWOOM_VERIFY.md)",
            ", ".join(unverified),
        )

    if args.dry_run:
        log.info("[dry-run] 종목 %d개, 시작일 %s", len(codes), start_date)
        for code in codes:
            path = storage.raw_path("daily_chart", code)
            log.info("  %s — 보유 마지막일: %s", code, storage.last_date(path) or "없음")
        return 0

    with KiwoomClient() as client:
        status = collect_universe(client, codes, start_date=start_date)

        if not args.skip_index:
            for idx in cfg["data"]["macro"]["indices"]:
                try:
                    collect_index_daily(client, idx["code"], start_date=start_date)
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
