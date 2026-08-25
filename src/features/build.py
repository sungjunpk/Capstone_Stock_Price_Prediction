"""모델 입력 조립 — panel / macro / static 3종 산출물 생성.

역할 분담:
    technical.py  종목 하나의 OHLCV → 기술적 지표
    build.py      여러 종목 + 매크로 + 고정특성을 모델이 먹을 형태로 조립

look-ahead 방지 원칙은 technical.py 와 동일하다. 매크로도 t 시점까지만 쓴다.
"""

from __future__ import annotations

import pandas as pd

from src.data import storage
from src.features.technical import (
    add_technical_features,
    drop_halted_days,
    forward_log_return,
    log_return,
    realized_vol,
    volume_features,
)
from src.utils.logging import get_logger

log = get_logger(__name__)

# 매크로 시퀀스를 구성하는 자산들. 지수는 index_daily, ETF 는 daily_chart 에 있다.
_MACRO_INDEX_KIND = "index_daily"
_MACRO_ETF_KIND = "daily_chart"


# --------------------------------------------------------------- panel
def build_panel(cfg: dict) -> pd.DataFrame:
    """유니버스 종목별 동적 피처 + 타깃. (code, date) 키."""
    feat_cfg = cfg["features"]
    horizon = int(feat_cfg["return_horizon"])
    codes = [u["code"] for u in cfg["data"]["universe"]]

    raw = storage.load_kind("daily_chart", codes=codes)
    if raw.empty:
        raise RuntimeError("data/raw/daily_chart 가 비었다. scripts/collect.py 를 먼저 실행할 것.")

    flow = _load_flow(codes)

    frames, halted_total = [], 0
    for code, part in raw.groupby("code", sort=True):
        part = part.sort_values("date").reset_index(drop=True)

        before = len(part)
        part = drop_halted_days(part)      # 지표 계산 전에 제거해야 rolling 이 안 오염된다
        halted_total += before - len(part)

        feats = add_technical_features(part, feat_cfg.get("technical", {}))
        if flow is not None:
            feats = _join_flow(feats, flow[flow["code"] == code])
        feats["target"] = forward_log_return(feats["close"], horizon)
        frames.append(feats)

    log.info("거래정지일 총 %d행 제거", halted_total)
    panel = pd.concat(frames, ignore_index=True).sort_values(["date", "code"])
    return panel.reset_index(drop=True)


def _load_flow(codes: list[str]) -> pd.DataFrame | None:
    """수급 데이터. **유니버스 전 종목이 갖춰졌을 때만** 사용한다.

    일부 종목에만 있으면 종목마다 피처 차원이 달라져 패널이 깨진다.
    부분 수집 상태에서는 아예 안 쓰는 편이 낫다 — 수집이 끝나면 자동으로 켜진다.
    """
    flow = storage.load_kind("investor_flow", codes=codes)
    if flow.empty:
        log.warning("수급 데이터 없음 — 수급 피처 없이 진행한다")
        return None

    have = set(flow["code"].unique())
    missing = set(codes) - have
    if missing:
        log.warning(
            "수급 데이터가 %d/%d 종목만 있다 (부족: %d개) — 수급 피처를 건너뛴다. "
            "`scripts/collect.py --tr flow` 완료 후 다시 실행하면 자동 포함된다.",
            len(have), len(codes), len(missing),
        )
        return None

    log.info("수급 데이터 %d종목 사용", len(have))
    return flow


def _join_flow(feats: pd.DataFrame, flow_one: pd.DataFrame) -> pd.DataFrame:
    """순매수를 거래량 대비 비율로 변환해 붙인다.

    원단위 순매수는 종목 규모에 따라 스케일이 수천 배 차이나 그대로 못 쓴다.
    거래량으로 나누면 '오늘 거래 중 외국인 순매수 비중'이 되어 종목 간 비교가 된다.
    """
    cols = ["individual", "foreign", "institution", "pension"]
    f = flow_one[["date", *[c for c in cols if c in flow_one.columns]]].copy()
    out = feats.merge(f, on="date", how="left")

    vol = out["volume"].replace(0, pd.NA)
    for c in cols:
        if c in out.columns:
            out[f"flow_{c}"] = (out[c] / vol).astype("float64").clip(-5, 5)
            out = out.drop(columns=[c])
    return out


# --------------------------------------------------------------- macro
def build_macro(cfg: dict) -> pd.DataFrame:
    """날짜별 매크로 피처. 크로스어텐션의 Key/Value 시퀀스가 된다."""
    macro_cfg = cfg["data"]["macro"]
    out: pd.DataFrame | None = None

    for idx in macro_cfg["indices"]:
        df = storage.load_kind(_MACRO_INDEX_KIND, codes=[idx["code"]])
        if df.empty:
            log.warning("지수 %s(%s) 없음 — 건너뛴다", idx["name"], idx["code"])
            continue
        out = _merge_macro(out, _macro_features(df, prefix=idx["name"].lower()))

    for etf in macro_cfg.get("overseas_etf_fallback", []):
        df = storage.load_kind(_MACRO_ETF_KIND, codes=[etf["code"]])
        if df.empty:
            log.warning("ETF %s(%s) 없음 — 건너뛴다", etf["name"], etf["code"])
            continue
        prefix = f"etf{etf['code']}"
        out = _merge_macro(out, _macro_features(df, prefix=prefix), how="outer")

    if out is None:
        raise RuntimeError("매크로 자산을 하나도 못 읽었다")

    out = out.sort_values("date").reset_index(drop=True)

    # ETF 는 상장이 늦어 앞 구간이 비어 있다(2021-06 상장, 커버리지 44%).
    # 0 으로 채우되 **가용성 플래그를 함께 준다** — 플래그가 없으면 모델이 0 을 신호로 오해한다.
    for prefix in {c.split("_")[0] for c in out.columns if c.startswith("etf")}:
        cols = [c for c in out.columns if c.startswith(f"{prefix}_")]
        avail = out[cols].notna().all(axis=1)
        out[f"{prefix}_available"] = avail.astype("float32")
        out[cols] = out[cols].fillna(0.0)
        log.info("%s 커버리지 %.0f%% — 결측은 0 + available 플래그로 처리",
                 prefix, 100 * avail.mean())

    return out


def _macro_features(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """지수/ETF 일봉 → 정상성 있는 매크로 피처.

    지수값 자체는 비정상(KOSPI 는 100배 스케일로 오기도 한다)이라 절대 그대로 쓰지 않는다.
    """
    d = df.sort_values("date").reset_index(drop=True)
    close, vol = d["close"], d["volume"]
    out = pd.DataFrame({"date": d["date"]})
    out[f"{prefix}_ret_1d"] = log_return(close, 1)
    out[f"{prefix}_ret_5d"] = log_return(close, 5)
    out[f"{prefix}_rvol_20"] = realized_vol(close, 20)
    out[f"{prefix}_vol_ratio"] = volume_features(vol)["vol_ratio"]
    return out


def _merge_macro(base, new, how: str = "outer"):
    return new if base is None else base.merge(new, on="date", how=how)


# --------------------------------------------------------------- static
def build_static(cfg: dict, train_end) -> pd.DataFrame:
    """종목별 고정 특성. TFT 변수선택망의 static 입력.

    시가총액 분위는 **train 구간 기준**으로 자른다 — 전체 기간으로 자르면
    미래 정보가 범주 경계에 새어든다.
    """
    uni = pd.DataFrame(cfg["data"]["universe"])
    uni = uni.rename(columns={"size": "size_class"})
    keep = [c for c in ("code", "name", "sector", "size_class", "market") if c in uni.columns]
    out = uni[keep].copy()

    info = storage.load_kind("stock_info", codes=out["code"].tolist())
    if not info.empty:
        cols = [c for c in ("code", "market_cap", "per", "pbr", "roe") if c in info.columns]
        out = out.merge(info[cols].drop_duplicates("code"), on="code", how="left")

    if "market_cap" in out.columns and out["market_cap"].notna().any():
        out["market_cap_bucket"] = pd.qcut(
            out["market_cap"].rank(method="first"), 5, labels=False, duplicates="drop"
        ).astype("Int64")
    else:
        out["market_cap_bucket"] = pd.NA

    out["sector"] = out["sector"].fillna("미분류")
    log.info("static: %d종목, 섹터 %d종, 시총구간 %s (train_end=%s 기준)",
             len(out), out["sector"].nunique(),
             out["market_cap_bucket"].nunique(), train_end)
    return out.reset_index(drop=True)
