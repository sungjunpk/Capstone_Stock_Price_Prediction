#!/usr/bin/env python
"""유니버스를 자동 선정해 configs/universe.yaml 에 쓴다.

두 갈래가 있다.

**지수 구성종목(기본, --index kospi200)** — ka20002 로 코스피200 구성종목을 받는다.
거래소가 유동성·업종대표성으로 관리하는 **외부 기준**이라 "왜 이 종목들인가"를
리포트에서 방어하기 쉽다. 시총 컷을 우리가 정하지 않는다.
정기변경(6·12월)이 있으므로 반기에 한 번 다시 돌린다.

**시총 상위 N개(--rank)** — 목표 학습샘플 수에서 종목 수를 역산한다.
종목당 샘플 ≈ (거래일수 - lookback - 지표워밍업), 2015년 이후 상장 유지 기준 약 2,360개.

사용:
    python scripts/build_universe.py                    # 코스피200 (기본)
    python scripts/build_universe.py --dry-run          # 저장 안 하고 결과만
    python scripts/build_universe.py --rank --count 150 # 시총 상위 150개
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

# 업종코드 — ka20002 의 inds_cd. 코스피200 은 '201' 이다(지수 일봉의 지수코드와 같다).
_INDEX_CODE = {"kospi200": "201"}


def fetch_stock_list(client: KiwoomClient, market: str) -> pd.DataFrame:
    """market: '0'=KOSPI, '10'=KOSDAQ"""
    spec = ep.STOCK_LIST
    data, _ = client.request(spec, {"mrkt_tp": market})
    recs = data.get(spec.list_key) or []
    df = parse_records(recs, spec.schema)
    log.info("%s: %d종목 수신", "KOSPI" if market == "0" else "KOSDAQ", len(df))
    return df


def fetch_index_members(client: KiwoomClient, inds_cd: str) -> list[str]:
    """지수 구성종목 코드를 지수 편입 순서대로 반환한다.

    ⚠️ 페이지당 100건이라 연속조회가 필수다. 한 번만 부르면 코스피200 이 100개로 보인다.
    """
    spec = ep.SECTOR_STOCKS
    codes: list[str] = []
    cont_yn = next_key = None
    while True:
        data, hdr = client.request(
            spec, {"mrkt_tp": "0", "inds_cd": inds_cd, "stex_tp": "1"},
            cont_yn=cont_yn, next_key=next_key,
        )
        page = parse_records(data.get(spec.list_key) or [], spec.schema)
        if page.empty:
            break
        codes += page["code"].tolist()
        if hdr.get("cont-yn") != "Y":
            break
        cont_yn, next_key = "Y", hdr.get("next-key")
    log.info("지수 %s: %d종목 수신", inds_cd, len(codes))
    return codes


def _join_members(allstk: pd.DataFrame, members: list[str]) -> pd.DataFrame:
    """지수 구성종목에 ka10099 의 업종/규모/시총을 붙인다. 편입 순서를 유지한다.

    걸러내지 않는다 — 코스피200 자체가 이미 거래소가 관리하는 목록이고,
    여기서 우리 기준으로 더 빼면 "유니버스 = 코스피200" 이라는 말이 거짓이 된다.
    """
    meta = allstk.drop_duplicates(subset=["code"]).set_index("code")
    missing = [c for c in members if c not in meta.index]
    if missing:
        log.warning("ka10099 에 없는 구성종목 %d개 — 업종/규모 없이 넣는다: %s",
                    len(missing), missing)
    out = meta.reindex(members).reset_index()
    out["market_cap"] = out["listed_shares"] * out["last_price"]
    log.info("  구성종목 %d개, 업종 %d종", len(out), out["sector"].nunique())
    return out


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
    ap.add_argument("--index", default="kospi200", choices=["kospi200"],
                    help="지수 구성종목을 유니버스로 쓴다 (기본)")
    ap.add_argument("--rank", action="store_true",
                    help="지수 대신 시총 상위 N개로 뽑는다 (--index 무시)")
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
    if args.rank:
        log.info("목표 %s → %d종목 선정 시도", f"{args.target_samples:,}샘플", need)
    else:
        log.info("지수 구성종목 기준: %s", args.index)

    # 지수 경로에서도 ka10099 는 받는다 — 업종/규모구간이 거기에만 있고,
    # 그 둘이 VSN 의 static covariate 라서 없으면 안 된다.
    markets = ("0",) if not args.rank else ("0", "10")
    with KiwoomClient() as client:
        frames = []
        for mrkt in markets:
            try:
                frames.append(fetch_stock_list(client, mrkt))
            except Exception as exc:  # noqa: BLE001 — 한쪽 시장 실패해도 진행
                log.error("시장 %s 리스트 실패: %s", mrkt, exc)
        members = None if args.rank else fetch_index_members(client, _INDEX_CODE[args.index])
    if not frames:
        log.error("종목 리스트를 하나도 받지 못했다")
        return 1

    allstk = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["code"])
    if args.rank:
        picked = select(allstk, need=need, start_date=start_date)
    else:
        picked = _join_members(allstk, members)

    est = len(picked) * args.samples_per_stock
    log.info("예상 학습샘플: 약 %s개", f"{est:,}")
    if args.rank and est < args.target_samples:
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
    if args.rank:
        source = "ka10099 종목정보 리스트 (시총 상위)"
        criteria = {
            "listing_date_before": start_date.isoformat(),
            "audit": "정상",
            "common_stock_only": True,
            "target_samples": args.target_samples,
        }
    else:
        source = f"ka20002 지수 구성종목 (inds_cd={_INDEX_CODE[args.index]}) "\
                 "+ ka10099 업종/규모"
        # 우리가 거른 게 없다는 사실 자체를 남긴다.
        criteria = {"index": args.index, "filtered": False}

    payload = {
        "generated": date.today().isoformat(),
        "source": source,
        "criteria": criteria,
        "universe": [
            {"code": r.code, "name": r.name,
             "sector": r.sector if isinstance(r.sector, str) and r.sector else "미분류",
             "size": r.size_class if isinstance(r.size_class, str) and r.size_class
                     else "미분류",
             "market": r.market}
            for r in picked.itertuples()
        ],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)
    log.info("저장: configs/universe.yaml (%d종목)", len(picked))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
