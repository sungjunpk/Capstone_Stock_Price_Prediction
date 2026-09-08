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
# 분봉 트랙은 config 프로파일이 이 값들을 덮어쓴다(profiles.intraday).
_MACRO_INDEX_KIND = "index_daily"
_MACRO_ETF_KIND = "daily_chart"
_CHART_KIND = "daily_chart"

# 수급이 일봉보다 이만큼 넘게 뒤처지면 경고한다. 자동수집이 평일마다 도니 정상은 0 이고,
# 연휴를 한 번 건너뛰어도 안 울리도록 여유를 뒀다.
_FLOW_STALE_BDAYS = 3


def _load_bars(kind: str, codes: list[str] | None) -> pd.DataFrame:
    """가격 봉을 읽고 **시간 컬럼 이름을 'date' 로 통일**한다.

    분봉 raw 의 키는 'datetime' 이지만 여기서 이름을 맞춰두면 하류가
    (dataset / split / backtest / inference) 봉 종류를 몰라도 된다 —
    전부 'date 컬럼의 순서'만 쓰기 때문이다. 봉 단위를 아는 건 이 파일까지다.
    """
    df = storage.load_kind(kind, codes=codes)
    if not df.empty and "date" not in df.columns and "datetime" in df.columns:
        df = df.rename(columns={"datetime": "date"})
    return df


# --------------------------------------------------------------- panel
def build_panel(cfg: dict) -> pd.DataFrame:
    """유니버스 종목별 동적 피처 + 타깃. (code, date) 키."""
    feat_cfg = cfg["features"]
    horizon = int(feat_cfg["return_horizon"])
    codes = [u["code"] for u in cfg["data"]["universe"]]
    kind = cfg["data"].get("chart_kind", _CHART_KIND)

    raw = _load_bars(kind, codes)
    if raw.empty:
        raise RuntimeError(f"data/raw/{kind} 가 비었다. 수집 스크립트를 먼저 실행할 것.")

    # 수급(ka10059)은 **일 단위**다. 봉 단위 패널에 날짜로 조인하면 조용히 어긋난다
    # (키 타입이 date vs datetime 이라 매칭이 0건이 되거나 예외가 난다).
    if kind == _CHART_KIND:
        flow = _load_flow(codes, bars_last=raw["date"].max())
    else:
        flow = None
        log.info("%s 패널이라 수급 피처를 쓰지 않는다 — 수급은 일 단위 데이터다", kind)

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
    panel = _add_cross_sectional(panel, feat_cfg)
    panel = _apply_target_mode(panel, feat_cfg)
    return panel.reset_index(drop=True)


def _add_cross_sectional(panel: pd.DataFrame, feat_cfg: dict) -> pd.DataFrame:
    """같은 날 전 종목 대비 **상대 위치**를 피처로 추가한다.

    왜 필요한가 — 기존 피처 17개는 전부 그 종목 혼자의 시계열에서 나온다.
    게다가 RevIN 이 종목별 윈도우 안에서 다시 표준화하므로, 모델은 "이 종목의
    모멘텀이 오늘 시장에서 상위인가"를 **구조적으로 알 수 없다.**
    그런데 매매 규칙(cross_sectional)은 정확히 그 상대 순위로 종목을 고른다 —
    모델이 전략에 필요한 정보를 못 보고 있었다.

    실측 증상: in-sample 랭크 IC 조차 0.0406 (test 0.0241) 로 낮고,
    십분위 스프레드가 t=1.10 으로 무의미했다. 과적합이 아니라 언더피팅이다.

    ⚠️ look-ahead 아니다. t 시점 값들만 t 시점 안에서 비교한다.
    ⚠️ 이름은 반드시 `xs_` 로 시작해야 한다 — 모델이 이 접두어로 RevIN 을 건너뛴다.
    """
    cols = list(feat_cfg.get("cross_sectional", []))
    if not cols:
        return panel

    missing = [c for c in cols if c not in panel.columns]
    if missing:
        raise ValueError(f"횡단면 피처로 지정된 컬럼이 패널에 없다: {missing}")

    g = panel.groupby("date")
    for c in cols:
        # 백분위 순위를 -0.5~+0.5 로. 순위라서 이상치에 둔감하고 날짜 간 스케일이 같다.
        panel[f"xs_{c}"] = g[c].rank(pct=True) - 0.5
    n_per_date = g.size()
    log.info("횡단면 피처 %d개 추가 (%s) — 날짜당 종목 수 중앙값 %d",
             len(cols), ", ".join(cols), int(n_per_date.median()))
    return panel


def _apply_target_mode(panel: pd.DataFrame, feat_cfg: dict) -> pd.DataFrame:
    """타깃을 원시 수익률로 둘지, 시장 대비 초과수익으로 바꿀지.

    `market_relative` 는 같은 날 전 종목의 평균 forward 수익률을 뺀다.

    왜 이게 필요한가 — 원시 수익률은 **시장 공통 성분이 압도적**이라(한국 대형주는
    지수와 상관이 0.7~0.9) 모델이 "시장이 오를까"를 맞히는 데 용량을 쓴다.
    그런데 우리 매매 규칙은 cross_sectional 순위라 그 공통 성분은 어차피 상쇄된다 —
    즉 학습이 매매에 안 쓰이는 것을 배우고 있었다.
    실측 증상이 정확히 이 모양이었다: 랭크 IC 는 유의(t=3.89)한데 십분위
    스프레드는 t=1.10 으로 무의미 — 순위는 맞히지만 상위가 실제로 더 오르지 않는다.

    ⚠️ look-ahead 아니다. 빼는 값은 **같은 미래 창의 횡단면 평균**이라 라벨 정의의
    일부다. 피처에는 들어가지 않고, 추론 시점에 알 필요도 없다.
    """
    mode = str(feat_cfg.get("target_mode", "raw"))
    if mode == "raw":
        return panel
    if mode != "market_relative":
        raise ValueError(f"알 수 없는 features.target_mode: {mode}")

    mkt = panel.groupby("date")["target"].transform("mean")
    before = panel["target"].std()
    panel["target"] = panel["target"] - mkt
    log.info("타깃을 시장 대비 초과수익으로 변환 — 표준편차 %.4f → %.4f (공통성분 %.0f%% 제거)",
             before, panel["target"].std(), 100 * (1 - panel["target"].std() / before))
    return panel


def _load_flow(codes: list[str], *, bars_last=None) -> pd.DataFrame | None:
    """수급 데이터. **유니버스 전 종목이 갖춰졌을 때만** 사용한다.

    일부 종목에만 있으면 종목마다 피처 차원이 달라져 패널이 깨진다.
    부분 수집 상태에서는 아예 안 쓰는 편이 낫다 — 수집이 끝나면 자동으로 켜진다.

    bars_last: 일봉의 마지막 날짜. 수급이 그보다 한참 뒤처졌는지 본다.
        ⚠️ 종목 유무만 검사하면 **날짜가 멈춘 건 안 잡힌다.** 실제로 2026-08-25 에
        멈춘 수급으로 9거래일을 돌았는데 경고가 한 줄도 안 나왔다
        (자동수집이 `--tr chart` 만 돌고 있었다 — `scripts/daily_collect.sh`).
        여기서 막지는 않는다. 과거는 멀쩡한 데이터라 학습에는 쓸 수 있고,
        멈춘 걸 **보이게** 하는 것이 목적이다.
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

    if bars_last is not None:
        flow_last = flow["date"].max()
        lag = len(pd.bdate_range(flow_last, bars_last)) - 1
        if lag > _FLOW_STALE_BDAYS:
            log.warning(
                "⚠️ 수급이 %s 에 멈춰 있다 — 일봉(%s)보다 %d거래일 뒤처졌다. "
                "최근 구간의 수급 피처가 비어 학습가능 행이 줄어든다. "
                "`scripts/collect.py --tr flow` 로 따라잡을 것.",
                pd.Timestamp(flow_last).date(), pd.Timestamp(bars_last).date(), lag,
            )

    log.info("수급 데이터 %d종목 사용 (최종일 %s)",
             len(have), pd.Timestamp(flow["date"].max()).date())
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
    index_kind = cfg["data"].get("macro_index_kind", _MACRO_INDEX_KIND)
    etf_kind = cfg["data"].get("macro_etf_kind", _MACRO_ETF_KIND)
    out: pd.DataFrame | None = None

    for idx in macro_cfg["indices"]:
        df = _load_bars(index_kind, [idx["code"]])
        if df.empty:
            log.warning("지수 %s(%s) 없음 — 건너뛴다", idx["name"], idx["code"])
            continue
        out = _merge_macro(out, _macro_features(df, prefix=idx["name"].lower()))

    for etf in macro_cfg.get("overseas_etf_fallback", []):
        df = _load_bars(etf_kind, [etf["code"]])
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


def _market_cap_at(info: pd.DataFrame, panel: pd.DataFrame, train_end) -> pd.Series:
    """train_end 시점의 시가총액. 코드 → 시총.

    `stock_info` 는 **조회시점 스냅샷 1행**이라 과거 시총이 없다. 그래서 수정주가
    비율로 되돌린다 — `시총(train_end) = 시총(현재) x 종가(train_end) / 종가(최근)`.
    수정주가는 액면분할·병합을 이미 반영하므로, 주식수가 그 사이 크게 변하지
    않았다는 가정 아래 성립하는 근사다.

    ⚠️ **이 근사가 필요한 이유가 look-ahead 다.** 현재 시총으로 구간을 자르면
    2015년 행이 "이 종목이 나중에 커진다"를 안다. 실측(2026-09-08): 가격비가
    0.03~4.91 로 벌어져 순위가 실제로 바뀐다 — 무해한 차이가 아니다.

    train_end 이전 가격이 없는 종목(그때 미상장)은 NA 로 남긴다. 그 시점에
    존재하지 않았으므로 '규모 미상'이 정직한 값이고, 코드북에서 0번(미등록)이 된다.
    """
    px = panel[["code", "date", "close"]].dropna(subset=["close"])
    upto = px[pd.to_datetime(px["date"]).dt.date <= train_end]
    at = upto.sort_values("date").groupby("code")["close"].last()
    latest = px.sort_values("date").groupby("code")["close"].last()

    cap_now = info.set_index("code")["market_cap"]
    return cap_now * (at / latest)


# --------------------------------------------------------------- static
def build_static(cfg: dict, train_end, panel: pd.DataFrame) -> pd.DataFrame:
    """종목별 고정 특성. TFT 변수선택망의 static 입력.

    시가총액 분위는 **train 구간 기준**으로 자른다 — 전체 기간으로 자르면
    미래 정보가 범주 경계에 새어든다. 정규화 통계를 train 에서만 뽑는 것과 같은 규칙이다.

    ⚠️ `size_class`(대형주/중형주/소형주)는 **static covariate 에서 뺐다**(2026-09-08).
    출처가 ka10099 의 **현재** 분류뿐이라 과거 시점 값을 구할 방법이 없어 그대로
    쓰면 look-ahead 였고, 애초에 `market_cap_bucket` 이 같은 정보를 더 세밀하게 담는다.
    참고용으로 `configs/universe.yaml` 에는 남아 있다.
    """
    uni = pd.DataFrame(cfg["data"]["universe"])
    keep = [c for c in ("code", "name", "sector", "market") if c in uni.columns]
    out = uni[keep].copy()

    info = storage.load_kind("stock_info", codes=out["code"].tolist())
    if not info.empty:
        cols = [c for c in ("code", "market_cap", "per", "pbr", "roe") if c in info.columns]
        out = out.merge(info[cols].drop_duplicates("code"), on="code", how="left")

    if "market_cap" in out.columns and out["market_cap"].notna().any():
        cap = _market_cap_at(info, panel, train_end)
        out["market_cap_train_end"] = out["code"].map(cap)
        out["market_cap_bucket"] = pd.qcut(
            out["market_cap_train_end"].rank(method="first"), 5,
            labels=False, duplicates="drop",
        ).astype("Int64")
    else:
        out["market_cap_train_end"] = pd.NA
        out["market_cap_bucket"] = pd.NA

    out["sector"] = out["sector"].fillna("미분류")
    unknown = int(out["market_cap_bucket"].isna().sum())
    log.info("static: %d종목, 섹터 %d종, 시총구간 %s (train_end=%s 시점 시총 기준"
             "%s)", len(out), out["sector"].nunique(),
             out["market_cap_bucket"].nunique(), train_end,
             f", 그때 미상장 {unknown}종목은 미등록" if unknown else "")
    return out.reset_index(drop=True)
