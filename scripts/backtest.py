#!/usr/bin/env python
"""학습된 모델 → test 구간 예측 → 거래비용 반영 백테스트.

필요한 것:
  1) 체크포인트 (outputs/checkpoints/phase1_*.pt)  ← 학습이 만든다
  2) panel/macro/static parquet                    ← build_features.py 가 만든다

캐글에서 학습했다면 체크포인트를 로컬로 내려받아 outputs/checkpoints/ 에 두거나,
캐글에서 그대로 이 스크립트를 돌려도 된다.

사용:
    python scripts/backtest.py                          # 최신 체크포인트 자동 선택
    python scripts/backtest.py --checkpoint outputs/checkpoints/phase1_abcd1234.pt
    python scripts/backtest.py --split test             # 기본 test (val 도 가능)
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.data.storage import PROCESSED_DIR  # noqa: E402
from src.evaluation.backtest import buy_and_hold, run_backtest  # noqa: E402
from src.evaluation.metrics import summarize  # noqa: E402
from src.models.phase1 import Phase1Config, Phase1Model  # noqa: E402
from src.training.dataset import StaticVocab, WindowDataset  # noqa: E402
from src.training.split import (  # noqa: E402
    SplitSpec,
    apply_normalizer,
    fit_normalizer,
    split_by_date,
)
from src.utils.config import PROJECT_ROOT, load_config  # noqa: E402
from src.utils.logging import get_logger, setup_logging  # noqa: E402
from src.utils.seed import get_device  # noqa: E402

log = get_logger("backtest")
CKPT_DIR = PROJECT_ROOT / "outputs" / "checkpoints"


def find_checkpoint(explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if not p.exists():
            raise SystemExit(f"체크포인트가 없다: {p}")
        return p

    cands = [p for p in CKPT_DIR.glob("phase1_*.pt") if "smoke" not in p.name]
    if not cands:
        raise SystemExit(
            "체크포인트가 없다. 학습을 먼저 하거나, 캐글에서 받은 .pt 를\n"
            "  outputs/checkpoints/ 에 두고 --checkpoint 로 지정할 것."
        )
    latest = max(cands, key=lambda p: p.stat().st_mtime)
    log.info("체크포인트 자동 선택: %s", latest.name)
    return latest


@torch.no_grad()
def predict(ckpt_path: Path, cfg: dict, split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """지정 구간에 대해 (예측 q10/q50/q90, 주가) 를 만든다."""
    device = get_device()
    # 캐글(CUDA)에서 저장한 것을 맥에서 열 수 있어야 한다
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    meta = ckpt["meta"]

    panel = pd.read_parquet(PROCESSED_DIR / "panel.parquet")
    macro = pd.read_parquet(PROCESSED_DIR / "macro.parquet")
    static = pd.read_parquet(PROCESSED_DIR / "static.parquet")

    feature_cols = meta["feature_cols"]
    missing = set(feature_cols) - set(panel.columns)
    if missing:
        raise SystemExit(
            f"체크포인트가 기대하는 피처가 패널에 없다: {sorted(missing)}\n"
            "  학습 때와 다른 데이터다 — build_features.py 를 다시 돌렸는지 확인할 것."
        )

    spec = SplitSpec.from_config(cfg)
    parts = split_by_date(panel, spec)
    # 정규화 통계는 학습 때와 동일하게 **train 구간에서만** 계산한다
    stats = fit_normalizer(
        parts["train"].dropna(subset=feature_cols + ["target"]), feature_cols
    )
    macro_cols = meta["macro_cols"]
    macro_train = macro[pd.to_datetime(macro["date"]).dt.date <= spec.train_end]
    macro_stats = fit_normalizer(macro_train.dropna(subset=macro_cols), macro_cols)
    macro_n = apply_normalizer(macro.fillna(0.0), macro_stats)

    part = parts[split]
    ds = WindowDataset(
        apply_normalizer(part, stats), macro_n, static,
        lookback=int(cfg["features"]["lookback"]),
        feature_cols=feature_cols, vocab=StaticVocab.build(static),
    )

    mcfg = Phase1Config(**{**ckpt["config"], "static_vocab": meta["vocab_sizes"]})
    model = Phase1Model(mcfg).to(device).eval()
    model.load_state_dict(ckpt["model"])
    log.info("모델 로드 완료 (val loss %.6f, epoch %d)",
             ckpt.get("val_loss", float("nan")), ckpt.get("epoch", -1))

    loader = DataLoader(ds, batch_size=1024, shuffle=False)
    out = []
    for dyn, mac, stat, _ in loader:
        q = model(dyn.to(device), mac.to(device), stat.to(device)).quantiles
        out.append(q.float().cpu())
    q = torch.cat(out).numpy()

    # 종목/날짜 복원 — Dataset 이 윈도우를 만든 순서와 **정확히 같게** 재구성해야
    # 예측이 엉뚱한 종목/날짜에 붙는다. 같은 필터·정렬을 그대로 반복한다.
    usable_by_stock = []
    for code, g in apply_normalizer(part, stats).groupby("code", sort=True):
        g = g.sort_values("date").dropna(subset=feature_cols + ["target"])
        if len(g) > ds.lookback:
            usable_by_stock.append((code, g))
    rows = [
        {"code": usable_by_stock[si][0],
         "date": usable_by_stock[si][1]["date"].iloc[end]}
        for si, end in ds._index
    ]
    preds = pd.DataFrame(rows)
    preds[["q10", "q50", "q90"]] = q
    prices = part[["code", "date", "close"]].copy()
    return preds, prices


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--allow-short", action="store_true",
                    help="공매도 허용 (기본은 매수 전용 — 모의투자 현실 반영)")
    args = ap.parse_args()

    setup_logging(run_name="backtest")
    cfg = load_config().raw
    ckpt_path = find_checkpoint(args.checkpoint)

    preds, prices = predict(ckpt_path, cfg, args.split)
    log.info("예측 %d건 | 종목 %d | %s ~ %s",
             len(preds), preds["code"].nunique(), preds["date"].min(), preds["date"].max())

    result = run_backtest(preds, prices, cfg, allow_short=args.allow_short)
    bh = summarize(buy_and_hold(prices))

    print("\n" + "=" * 66)
    print(f"백테스트 결과 ({args.split} 구간, 거래비용 반영)")
    print("=" * 66)
    print(f"{'지표':<16}{'전략':>14}{'매수후보유':>14}")
    for k in ("cagr", "volatility", "sharpe", "sortino", "calmar",
              "max_drawdown", "hit_rate", "total_return"):
        print(f"  {k:<14}{result.metrics[k]:>14.4f}{bh[k]:>14.4f}")

    s = result.signal_stats
    print("\n신호 통계 — 기권 로직이 이 프로젝트의 핵심 차별점")
    print(f"  총 판단      {s['decisions']:,}회")
    print(f"  기권률       {s['abstain_rate']:.1%}   (신뢰구간이 넓어 관망)")
    print(f"  실거래율     {s['trade_rate']:.1%}")
    print(f"  체결 건수    {s['n_trades']:,}")
    print(f"  리스크 차단  {s['blocked']:,}회")
    print(f"  왕복비용     {s['round_trip_cost']:.2%}")

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "checkpoint": ckpt_path.name, "split": args.split,
        "allow_short": args.allow_short,
        "strategy": result.metrics, "buy_and_hold": bh,
        "signal_stats": s,
    }
    out = PROJECT_ROOT / "outputs" / "reports" / \
        f"backtest_{datetime.now():%Y%m%d_%H%M%S}_{ckpt_path.stem}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {out.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
