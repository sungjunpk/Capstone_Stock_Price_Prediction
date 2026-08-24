#!/usr/bin/env python
"""ka10099 종목 리스트로 유니버스를 자동 선정해 configs/universe.yaml 에 쓴다.

목표 학습샘플 수에서 필요한 종목 수를 역산한다.
종목당 샘플 ≈ (거래일수 - lookback - 지표워밍업). 2015년 이후 상장 유지 종목 기준 약 2,360개.

사용:
    python scripts/build_universe.py --target-samples 300000
    python scripts/build_universe.py --count 150        # 종목 수 직접 지정
    python scripts/build_universe.py --dry-run          # 저장 안 하고 결과만
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from src.data.kiwoom import endpoints as ep  # noqa: E402
from src.data.kiwoom.client import KiwoomClient  # noqa: E402
from src.utils.config import PROJECT_ROOT, load_config  # noqa: E402
from src.utils.logging import get_logger, setup_logging  # noqa: E402
from src.utils.parsing import parse_records  # noqa: E402

log = get_logger("build_universe")

# 제외 대상 — 주가 시계열 모델링에 부적합하거나 데이터가 특이한 종목들
_EXCLUDE_NAME = (
    "스팩", "SPAC", "리츠", "REIT", "ETN",
    "우B", "우C",           # 종류주
)
_EXCLUDE_SECTOR = ("기타금융", "投資회사", "투자회사")
# 주식이 아닌 상품(인프라펀드 등)은 시장 이름으로 걸러진다
# ETF 는 개별 종목이 아니라 바스켓이라 예측 대상에서 뺀다
# (매크로 시퀀스로는 쓰지만 그건 config.yaml 의 overseas_etf_fallback 에서 따로 지정)
_EXCLUDE_MARKET = ("인프라투자금융", "부동산투자회사", "ETF", "ETN")


def fetch_stock_list(client: KiwoomClient, market: str) -> pd.DataFrame:
    """market: '0'=KOSPI, '10'=KOSDAQ"""
    spec = ep.STOCK_LIST
    data, _ = client.request(spec, {"mrkt_tp": market})
    recs = data.get(spec.list_key) or []
    df = parse_records(recs, spec.schema)
    log.info("%s: %d종목 수신", "KOSPI" if market == "0" else "KOSDAQ", len(df))
    return df


def select(df: pd.DataFrame, *, need: int, start_date: date) -> pd.DataFrame:
    """유동성·이력 기준으로 거른 뒤 상장주식수 순으로 상위 need 개."""
    n0 = len(df)
    df = df.dropna(subset=["code", "listing_date"])

    # 1) 전체 학습 구간의 이력이 있어야 한다 (start_date 이전 상장)
    df = df[df["listing_date"] <= start_date]
    log.info("  상장일 <= %s: %d종목 (전체 %d)", start_date, len(df), n0)

    # 2) 관리종목/거래정지 제외
    df = df[df["audit"].fillna("") == "정상"]
    log.info("  감사의견 정상: %d종목", len(df))

    # 3) 보통주만 (우선주는 코드 끝자리가 0이 아니다)
    df = df[df["code"].str.len() == 6]
    df = df[df["code"].str.endswith("0")]
    log.info("  보통주만: %d종목", len(df))

    # 4) 스팩·리츠 등 제외
    name = df["name"].fillna("")
    df = df[~name.str.contains("|".join(_EXCLUDE_NAME), case=False, regex=True)]
    sector = df["sector"].fillna("")
    df = df[~sector.isin(_EXCLUDE_SECTOR)]
    log.info("  스팩/리츠 등 제외: %d종목", len(df))

    # 5) 주식이 아닌 상품(인프라펀드/리츠 등) 제외
    df = df[~df["market"].fillna("").isin(_EXCLUDE_MARKET)]
    log.info("  펀드/리츠 시장 제외: %d종목", len(df))

    # 6) 시가총액 순으로 상위 need 개.
    #    상장주식수만으로 정렬하면 저가 대형주(주식수만 많은 종목)가 앞에 온다 —
    #    유동성 대용으로는 시총(주식수 × 종가)이 맞다.
    df = df.assign(market_cap=df["listed_shares"] * df["last_price"])
    df = df.dropna(subset=["market_cap"])
    df = df.sort_values("market_cap", ascending=False)

    out = df.head(need).reset_index(drop=True)
    log.info("  최종 선정: %d종목", len(out))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-samples", type=int, default=300_000)
    ap.add_argument("--count", type=int, help="종목 수 직접 지정 (target-samples 무시)")
    ap.add_argument("--samples-per-stock", type=int, default=2360,
                    help="종목당 예상 학습샘플 (기본: 2015~현재 기준 실측값)")
    ap.add_argument("--margin", type=float, default=1.15, help="여유 배수")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    setup_logging(run_name="build_universe")
    cfg = load_config()
    start_date = pd.Timestamp(cfg["data"]["start_date"]).date()

    need = args.count or int(args.target_samples / args.samples_per_stock * args.margin)
    log.info("목표 %s → %d종목 선정 시도", f"{args.target_samples:,}샘플", need)

    with KiwoomClient() as client:
        frames = []
        for mrkt in ("0", "10"):
            try:
                frames.append(fetch_stock_list(client, mrkt))
            except Exception as exc:  # noqa: BLE001 — 한쪽 시장 실패해도 진행
                log.error("시장 %s 리스트 실패: %s", mrkt, exc)
    if not frames:
        log.error("종목 리스트를 하나도 받지 못했다")
        return 1

    allstk = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["code"])
    picked = select(allstk, need=need, start_date=start_date)

    est = len(picked) * args.samples_per_stock
    log.info("예상 학습샘플: 약 %s개", f"{est:,}")
    if est < args.target_samples:
        log.warning("목표(%s)에 못 미친다. --samples-per-stock 을 조정하거나 "
                    "start_date 를 앞당길 것.", f"{args.target_samples:,}")

    print("\n시장별:", picked["market"].value_counts().to_dict())
    print("규모별:", picked["size_class"].value_counts().to_dict())
    print(f"업종 {picked['sector'].nunique()}종")
    show = picked.assign(시총_조=(picked["market_cap"] / 1e12).round(1))
    print(show[["code", "name", "sector", "size_class", "market", "시총_조"]]
          .head(12).to_string(index=False))
    print("  ...")
    print(show[["code", "name", "sector", "size_class", "market", "시총_조"]]
          .tail(3).to_string(index=False))

    if args.dry_run:
        log.info("[dry-run] 저장 안 함")
        return 0

    out_path = PROJECT_ROOT / "configs" / "universe.yaml"
    payload = {
        "generated": date.today().isoformat(),
        "source": "ka10099 종목정보 리스트",
        "criteria": {
            "listing_date_before": start_date.isoformat(),
            "audit": "정상",
            "common_stock_only": True,
            "target_samples": args.target_samples,
        },
        "universe": [
            {"code": r.code, "name": r.name,
             "sector": r.sector if isinstance(r.sector, str) and r.sector else "미분류",
             "size": r.size_class, "market": r.market}
            for r in picked.itertuples()
        ],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)
    log.info("저장: configs/universe.yaml (%d종목)", len(picked))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
