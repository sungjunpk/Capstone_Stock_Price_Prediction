"""학습된 체크포인트 → 분위 예측. **백테스트와 모의투자가 공유하는 단일 추론 경로.**

CLAUDE.md 절대 규칙 6·7 의 연장선이다:
    - 정규화 통계는 언제나 **train 구간에서만** 계산한다. 모의투자도 예외가 아니다.
      오늘 데이터로 통계를 다시 잡으면 학습 때와 다른 스케일이 모델에 들어간다.
    - 백테스트가 쓰는 추론과 모의투자가 쓰는 추론이 갈라지면, 백테스트에서 검증한
      숫자가 실거래에서 재현되지 않는다. 그래서 여기 하나만 둔다.

백테스트(`predict_split`)와 모의투자(`predict_recent`)의 유일한 차이는
**어느 윈도우를 뽑느냐**뿐이다. 모델·정규화·윈도우 구성은 완전히 같은 코드다.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from src.data.storage import PROCESSED_DIR
from src.models.phase1 import Phase1Config, Phase1Model
from src.training.dataset import StaticVocab, WindowDataset
from src.training.split import (
    SplitSpec,
    apply_normalizer,
    fit_normalizer,
    split_by_date,
)
from src.utils.logging import get_logger
from src.utils.seed import get_device

log = get_logger(__name__)

QUANTILE_COLS = ["q10", "q50", "q90"]


@dataclass
class LoadedModel:
    model: Phase1Model
    meta: dict
    device: torch.device
    path: Path
    val_loss: float = float("nan")
    epoch: int = -1

    @property
    def feature_cols(self) -> list[str]:
        return list(self.meta["feature_cols"])

    @property
    def macro_cols(self) -> list[str]:
        return list(self.meta["macro_cols"])


@dataclass
class FeatureBundle:
    """정규화까지 끝난 모델 입력 3종. train 통계로만 정규화되어 있다."""

    panel: pd.DataFrame          # 전 구간(정규화 완료)
    raw_panel: pd.DataFrame      # 원본 — 가격/날짜는 여기서 읽는다
    macro: pd.DataFrame
    static: pd.DataFrame
    vocab: StaticVocab
    spec: SplitSpec

    @property
    def last_date(self):
        return pd.to_datetime(self.raw_panel["date"]).max().date()


def load_model(ckpt_path: str | Path, *, device: torch.device | None = None) -> LoadedModel:
    """체크포인트 로드. 캐글(CUDA)에서 저장한 것을 맥에서 열 수 있어야 한다."""
    path = Path(ckpt_path)
    device = device or get_device()
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    meta = ckpt["meta"]

    mcfg = Phase1Config(**{**ckpt["config"], "static_vocab": meta["vocab_sizes"]})
    model = Phase1Model(mcfg).to(device).eval()
    model.load_state_dict(ckpt["model"])

    log.info(
        "모델 로드: %s (val loss %.6f, epoch %s)",
        path.name, ckpt.get("val_loss", float("nan")), ckpt.get("epoch", -1),
    )
    return LoadedModel(
        model=model, meta=meta, device=device, path=path,
        val_loss=float(ckpt.get("val_loss", float("nan"))),
        epoch=int(ckpt.get("epoch", -1)),
    )


def load_features(cfg: dict, loaded: LoadedModel) -> FeatureBundle:
    """panel/macro/static 을 읽고 **train 구간 통계로만** 정규화한다."""
    # 트랙마다 산출물이 다르다 (일봉: panel.parquet / 60분봉: panel_60m.parquet)
    sfx = cfg["data"].get("processed_suffix", "")
    panel = pd.read_parquet(PROCESSED_DIR / f"panel{sfx}.parquet")
    macro = pd.read_parquet(PROCESSED_DIR / f"macro{sfx}.parquet")
    static = pd.read_parquet(PROCESSED_DIR / f"static{sfx}.parquet")

    feature_cols = loaded.feature_cols
    missing = set(feature_cols) - set(panel.columns)
    if missing:
        raise RuntimeError(
            f"체크포인트가 기대하는 피처가 패널에 없다: {sorted(missing)}\n"
            "  학습 때와 다른 데이터다 — build_features.py 를 다시 돌렸는지 확인할 것."
        )

    spec = SplitSpec.from_config(cfg)
    parts = split_by_date(panel, spec)
    stats = fit_normalizer(
        parts["train"].dropna(subset=feature_cols + ["target"]), feature_cols
    )

    macro_cols = loaded.macro_cols
    macro_train = macro[pd.to_datetime(macro["date"]).dt.date <= spec.train_end]
    macro_stats = fit_normalizer(macro_train.dropna(subset=macro_cols), macro_cols)

    return FeatureBundle(
        panel=apply_normalizer(panel, stats),
        raw_panel=panel,
        macro=apply_normalizer(macro.fillna(0.0), macro_stats),
        static=static,
        vocab=StaticVocab.build(static),
        spec=spec,
    )


def _make_dataset(
    part: pd.DataFrame, bundle: FeatureBundle, cfg: dict, loaded: LoadedModel,
    *, require_target: bool,
) -> WindowDataset:
    return WindowDataset(
        part, bundle.macro, bundle.static,
        lookback=int(cfg["features"]["lookback"]),
        feature_cols=loaded.feature_cols,
        vocab=bundle.vocab,
        require_target=require_target,
    )


@torch.no_grad()
def _run(loaded: LoadedModel, ds, rows: list[int] | None, batch_size: int) -> np.ndarray:
    data = ds if rows is None else Subset(ds, rows)
    loader = DataLoader(data, batch_size=batch_size, shuffle=False)
    out = []
    for dyn, mac, stat, _ in loader:
        q = loaded.model(
            dyn.to(loaded.device), mac.to(loaded.device), stat.to(loaded.device)
        ).quantiles
        out.append(q.float().cpu())
    if not out:
        return np.empty((0, 3), dtype=np.float32)
    return torch.cat(out).numpy()


def predict_split(
    loaded: LoadedModel, bundle: FeatureBundle, cfg: dict, split: str,
    *, batch_size: int = 1024,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """백테스트용. (예측 q10/q50/q90 + target, 주가) 를 낸다."""
    part = split_by_date(bundle.panel, bundle.spec)[split]
    ds = _make_dataset(part, bundle, cfg, loaded, require_target=True)

    preds = ds.sample_keys()
    preds[QUANTILE_COLS] = _run(loaded, ds, None, batch_size)

    raw_part = split_by_date(bundle.raw_panel, bundle.spec)[split]
    prices = raw_part[["code", "date", "close"]].copy()
    return preds, prices


def predict_recent(
    loaded: LoadedModel, bundle: FeatureBundle, cfg: dict,
    *, days: int = 90, batch_size: int = 1024,
) -> pd.DataFrame:
    """모의투자용. **최근 `days` 일치 윈도우**를 예측한다.

    왜 최신 하루가 아니라 최근 구간인가:
      기권 임계값은 절대값이 아니라 **예측 폭 분포의 백분위**로 잡는다
      (`resolve_abstain_threshold`). 오늘 하루치 폭만으로 백분위를 잡으면
      시장 전체가 불확실한 날에도 "그날 기준 상위 30%"가 그대로 통과해버려,
      기권 로직이 '언제 쉬는가'를 판단하지 못한다.
      최근 구간의 분포를 기준으로 삼아야 오늘이 평소보다 불확실한지 알 수 있다.

    타깃이 없는 최신 구간(t+h 가 아직 안 온 날들)도 포함한다 —
    `require_target=False` 가 그 행들을 살린다.
    """
    panel = bundle.panel
    last = pd.to_datetime(panel["date"]).max()
    # 윈도우 구성에는 lookback 만큼의 과거가 필요하므로 행은 자르지 않는다.
    # 대신 만들어진 윈도우 중 최근 것만 고른다.
    ds = _make_dataset(panel, bundle, cfg, loaded, require_target=False)

    keys = ds.sample_keys()
    cutoff = last - timedelta(days=int(days))
    rows = keys.index[pd.to_datetime(keys["date"]) >= cutoff].tolist()
    if not rows:
        raise RuntimeError(f"최근 {days}일 안에 예측 가능한 윈도우가 없다 (마지막 {last.date()})")

    out = keys.loc[rows].reset_index(drop=True)
    out[QUANTILE_COLS] = _run(loaded, ds, rows, batch_size)
    # target 은 이 경로에서 의미가 없다(미래가 안 왔다). 실수로 쓰이지 않게 버린다.
    out = out.drop(columns=["target"], errors="ignore")
    log.info("최근 예측 %d건 | 종목 %d | %s ~ %s",
             len(out), out["code"].nunique(),
             pd.to_datetime(out["date"]).min().date(),
             pd.to_datetime(out["date"]).max().date())
    return out


@torch.no_grad()
def vsn_weights_split(
    loaded: LoadedModel, bundle: FeatureBundle, cfg: dict, split: str,
    *, batch_size: int = 1024,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """**해석가능성 진단 전용** — VSN 변수선택 가중치를 (code, date) 에 붙여 낸다.

    매매 경로가 아니다. 매매에 쓰는 예측은 `predict_split` / `predict_recent`
    하나뿐이라는 규칙(CLAUDE.md 7)은 그대로다 — 여기서는 분위를 내지 않는다.

    반환: (dynamic, static)
        dynamic: code, date, <피처별 가중치>   — 패치축(N)은 평균으로 접는다
        static:  code, date, <static 변수별 가중치>
    """
    part = split_by_date(bundle.panel, bundle.spec)[split]
    ds = _make_dataset(part, bundle, cfg, loaded, require_target=True)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    dyn_w, stat_w = [], []
    for dyn, mac, stat, _ in loader:
        out = loaded.model(
            dyn.to(loaded.device), mac.to(loaded.device), stat.to(loaded.device)
        )
        # (B,N,C) → (B,C). 시점별 가중치의 평균이 그 윈도우의 채널 중요도다.
        dyn_w.append(out.dynamic_weights.float().mean(dim=1).cpu())
        stat_w.append(out.static_weights.float().cpu())

    keys = ds.sample_keys()[["code", "date"]].reset_index(drop=True)
    static_names = list(loaded.meta["vocab_sizes"])

    dynamic = pd.concat(
        [keys, pd.DataFrame(torch.cat(dyn_w).numpy(), columns=loaded.feature_cols)],
        axis=1,
    )
    static = pd.concat(
        [keys, pd.DataFrame(torch.cat(stat_w).numpy(), columns=static_names)],
        axis=1,
    )
    return dynamic, static


def latest_slice(preds: pd.DataFrame) -> pd.DataFrame:
    """가장 최근 날짜의 예측만. 오늘의 매매 판단 대상이다."""
    d = pd.to_datetime(preds["date"])
    return preds[d == d.max()].reset_index(drop=True)


def latest_prices(bundle: FeatureBundle) -> dict[str, float]:
    """패널 마지막 종가. 브로커 현재가를 못 받았을 때의 대체값이다."""
    raw = bundle.raw_panel
    idx = raw.groupby("code")["date"].idxmax()
    last = raw.loc[idx, ["code", "close"]]
    return {str(r.code): float(r.close) for r in last.itertuples() if pd.notna(r.close)}
