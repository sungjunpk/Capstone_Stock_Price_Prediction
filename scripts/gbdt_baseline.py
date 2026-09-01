#!/usr/bin/env python
"""GBDT 베이스라인 — Phase1 트랜스포머와 같은 조건에서 맞붙인다.

왜 필요한가:
    Qlib 공식 벤치마크(CSI300/Alpha158)에서 LightGBM·XGBoost 가 Transformer·LSTM 을
    모두 이긴다(Rank IC 0.047~0.051 vs 0.041~0.044). 우리 자체 진단도 같은 방향이었다 —
    `rvol_20` 단일 피처의 test 랭크 IC(0.0411)가 학습된 모델 전체(0.0237)보다 컸다.
    **잘 최적화된 단순 베이스라인과의 공정한 비교가 없다**는 건 이 분야 문헌의
    공통 약점이고, 우리도 아직 그게 없다.

무엇을 고정하는가 (공정성의 전부):
    - 같은 패널 / 같은 날짜 기준 split (SplitSpec) / 같은 embargo
    - 같은 타깃 (t+1~t+h 누적 로그수익률)
    - 같은 평가 함수 (evaluation.metrics 의 rank_ic / decile_spread / summarize)
    - 같은 백테스트 (evaluation.backtest.run_backtest, 같은 trading 설정)
    - 같은 (code, date) 예측 대상 — Phase1 은 lookback 120일이 차야 첫 예측을 내므로
      교집합으로 맞춘다. 안 맞추면 GBDT 가 더 이른 구간을 덤으로 먹는다.
    - early stopping 도 **날짜 기준 val** 로 한다. sklearn 내장 early stopping 은
      train 을 무작위로 쪼개는데, 인접일이 상관돼 있어 낙관적으로 멈춘다.

무엇이 다른가 (구조상 어쩔 수 없는 것):
    Phase1 은 (120일 × 17채널) 시퀀스를 보고, GBDT 는 시점 t 의 피처 + 지정한 lag 만
    본다. 트리는 시퀀스를 못 먹기 때문이다. Qlib 의 Alpha158 도 같은 방식이다 —
    lookback 정보를 **피처로 미리 접어 넣고** 트리에 준다.

사용:
    python scripts/gbdt_baseline.py                     # 기본 (lag 1,5,20)
    python scripts/gbdt_baseline.py --lags 0            # 시점 t 만
    python scripts/gbdt_baseline.py --no-phase1         # GBDT 만 (체크포인트 없이)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402

from src.data.storage import PROCESSED_DIR  # noqa: E402
from src.evaluation.backtest import buy_and_hold, run_backtest  # noqa: E402
from src.evaluation.metrics import decile_spread, rank_ic, summarize  # noqa: E402
from src.training.dataset import dynamic_feature_columns  # noqa: E402
from src.training.split import SplitSpec, split_by_date  # noqa: E402
from src.utils.config import PROJECT_ROOT, load_config  # noqa: E402
from src.utils.logging import get_logger, setup_logging  # noqa: E402

log = get_logger("gbdt_baseline")
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
CKPT_DIR = PROJECT_ROOT / "outputs" / "checkpoints"

QUANTILES = (0.1, 0.5, 0.9)


# --------------------------------------------------------------- 피처 조립
def build_table(cfg: dict, lags: list[int]) -> tuple[pd.DataFrame, list[str]]:
    """패널 + 매크로 + static → 트리가 먹을 수 있는 평평한 표.

    lag 는 **종목별로** 민다. groupby 없이 shift 하면 종목 경계를 넘어
    앞 종목의 꼬리가 다음 종목의 머리에 붙는다 (조용히 틀리는 유형).
    """
    sfx = cfg["data"].get("processed_suffix", "")
    panel = pd.read_parquet(PROCESSED_DIR / f"panel{sfx}.parquet")
    macro = pd.read_parquet(PROCESSED_DIR / f"macro{sfx}.parquet")
    static = pd.read_parquet(PROCESSED_DIR / f"static{sfx}.parquet")

    panel["date"] = pd.to_datetime(panel["date"])
    macro["date"] = pd.to_datetime(macro["date"])
    panel = panel.sort_values(["code", "date"]).reset_index(drop=True)

    dyn_cols = dynamic_feature_columns(panel)
    feat_cols = list(dyn_cols)

    # --- lag: 시퀀스 정보를 피처로 접어 넣는다 (Alpha158 방식)
    if lags:
        g = panel.groupby("code", sort=False)[dyn_cols]
        blocks = []
        for lag in lags:
            s = g.shift(lag)
            s.columns = [f"{c}_lag{lag}" for c in dyn_cols]
            blocks.append(s)
            feat_cols += list(s.columns)
        panel = pd.concat([panel] + blocks, axis=1)

    # --- 매크로: 날짜로 붙인다 (전 종목 공통)
    macro_cols = [c for c in macro.columns if c != "date"]
    panel = panel.merge(macro[["date"] + macro_cols], on="date", how="left")
    feat_cols += macro_cols

    # --- static: 범주형을 정수 코드로. 트리는 임베딩이 필요 없다
    st_cols = [c for c in ("sector", "size_class", "market_cap_bucket")
               if c in static.columns]
    panel = panel.merge(static[["code"] + st_cols], on="code", how="left")
    for c in st_cols:
        panel[c] = panel[c].astype("category").cat.codes.astype("float64")
    feat_cols += st_cols

    panel["day_of_week"] = panel["date"].dt.dayofweek.astype("float64")
    feat_cols.append("day_of_week")

    log.info("표 %d행 × 피처 %d개 (동적 %d × (1+lag %d) + 매크로 %d + static %d)",
             len(panel), len(feat_cols), len(dyn_cols), len(lags),
             len(macro_cols), len(st_cols) + 1)
    return panel, feat_cols


# --------------------------------------------------------------- 손실
def pinball(y: np.ndarray, preds: np.ndarray, quantiles=QUANTILES) -> float:
    """train.py 의 pinball_loss 와 같은 정의 — 분위 축까지 평균."""
    e = y[:, None] - preds
    q = np.asarray(quantiles)[None, :]
    return float(np.maximum(q * e, (q - 1.0) * e).mean())


def _pinball_1q(y: np.ndarray, pred: np.ndarray, q: float) -> float:
    e = y - pred
    return float(np.maximum(q * e, (q - 1.0) * e).mean())


def quantile_breakdown(y: np.ndarray, preds: np.ndarray, base_q: np.ndarray) -> dict:
    """**분위별로** 기준선을 이겼는지 따로 본다.

    합쳐 놓으면 안 보이는 게 있다: 폭(q10/q90)만 배우고 방향(q50)은 못 배워도
    총 pinball 은 개선된 것처럼 나온다. 우리 전략은 q50 순위로만 사므로
    q50 이 기준선을 못 이기면 성과 지표를 읽을 이유가 없다.
    """
    out = {}
    for i, q in enumerate(QUANTILES):
        bl = _pinball_1q(y, np.full_like(y, base_q[i]), q)
        md = _pinball_1q(y, preds[:, i], q)
        out[f"q{int(q * 100)}"] = {
            "baseline": round(bl, 6), "model": round(md, 6),
            "improvement_pct": round(100 * (bl - md) / bl, 3),
        }
    return out


# --------------------------------------------------------------- 학습
def fit_quantile(
    q: float, Xtr, ytr, Xva, yva, Xte, *, chunk: int, max_iter: int,
    patience: int, seed: int, learning_rate: float, max_leaf: int, l2: float,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    """분위 하나. **날짜 기준 val 로 early stopping** 하고 최적 지점의 예측을 남긴다.

    warm_start 로 트리를 조금씩 늘리면서 val 손실을 보고, 개선될 때마다
    val/test 예측을 스냅샷한다. 재학습 없이 최적 지점의 예측을 얻는 방법이다
    (HistGradientBoosting 은 staged_predict 를 주지 않는다).
    """
    m = HistGradientBoostingRegressor(
        loss="quantile", quantile=q,
        learning_rate=learning_rate, max_leaf_nodes=max_leaf,
        l2_regularization=l2, early_stopping=False,
        warm_start=True, max_iter=chunk, random_state=seed,
    )

    best = (np.inf, 0, None, None)      # (val loss, n_iter, val pred, test pred)
    stale = 0
    for n in range(chunk, max_iter + 1, chunk):
        m.set_params(max_iter=n)
        m.fit(Xtr, ytr)
        pv = m.predict(Xva)
        loss = _pinball_1q(yva, pv, q)
        if loss < best[0] - 1e-9:
            best = (loss, n, pv, m.predict(Xte))
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    log.info("  q%02d — 트리 %d개에서 최적 (val pinball %.6f)",
             int(q * 100), best[1], best[0])
    return best[2], best[3], best[1], best[0]


def _sort_quantiles(raw: np.ndarray) -> tuple[np.ndarray, int]:
    """**분위 교차를 정렬로 강제한다.**

    Phase1 은 누적 softplus 로 단조성을 구조적으로 보장하지만, 분위별로 따로 학습한
    GBDT 는 교차가 난다. 교차가 남으면 `trading/signal.py` 의 QuantilePrediction 이
    예외를 던져 백테스트가 통째로 죽는다.
    """
    crossed = int((np.diff(raw, axis=1) < 0).any(axis=1).sum())
    return np.sort(raw, axis=1), crossed


# --------------------------------------------------------------- Phase1 쪽
def phase1_predictions(cfg: dict, ckpt: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """비교 대상. 기존 추론 경로를 그대로 쓴다 — 재구현하지 않는다.

    val 도 같이 낸다. Phase1 의 val 표본은 lookback 120일이 차는 윈도우뿐이라
    GBDT 의 val(전체 행)과 **집합이 다르다.** 같은 기준선 위에 놓으려면
    각자의 표본 위에서 각자의 기준선과 비교해야 한다.
    """
    from src.models.inference import load_features, load_model, predict_split

    loaded = load_model(ckpt)
    bundle = load_features(cfg, loaded)
    te, _ = predict_split(loaded, bundle, cfg, "test")
    va, _ = predict_split(loaded, bundle, cfg, "val")
    for d in (te, va):
        d["date"] = pd.to_datetime(d["date"])
    return te, va


def find_daily_checkpoint() -> Path | None:
    daily = re.compile(r"^phase1_[0-9a-f]{8}\.pt$")
    cands = [p for p in CKPT_DIR.glob("phase1_*.pt") if daily.match(p.name)]
    return max(cands, key=lambda p: p.stat().st_mtime) if cands else None


# --------------------------------------------------------------- 평가
def evaluate(name: str, preds: pd.DataFrame, prices: pd.DataFrame, cfg: dict) -> dict:
    """예측 → 진단 + 백테스트. 두 모델에 **같은 함수**를 쓴다."""
    ic = rank_ic(preds)
    spread = decile_spread(preds)
    result = run_backtest(preds[["code", "date", "q10", "q50", "q90"]], prices, cfg)

    log.info("[%s] 랭크 IC %.4f (t=%.2f) | 십분위 스프레드 %.4f (t=%.2f)",
             name, ic["ic_mean"], ic["t_stat"],
             spread["spread_mean"], spread["t_stat"])
    log.info("[%s] Sharpe %.2f | CAGR %.1f%% | MDD %.1f%% | 노출 %.1f%%",
             name, result.metrics["sharpe"], 100 * result.metrics["cagr"],
             100 * result.metrics["max_drawdown"],
             100 * result.signal_stats["avg_gross_exposure"])

    return {
        "rank_ic": ic,
        "decile_spread": spread,
        "backtest": result.metrics,
        "signal_stats": result.signal_stats,
        "n_predictions": int(len(preds)),
    }


def permutation_test(
    preds: pd.DataFrame, prices: pd.DataFrame, cfg: dict, n: int, real: dict,
) -> dict:
    """음성 대조군 — **q50 순위만** 날짜별로 섞고 나머지는 전부 그대로 둔다.

    폭(q90-q10)은 종목에 붙어 있으므로 기권 판단이 바뀌지 않는다. 바뀌는 건
    "어느 종목을 살 것인가" 하나뿐이다. 따라서 셔플 성과는 **신호를 뺀
    파이프라인(기권 필터 + 사이징 + 리스크 오버레이 + 시장 베타)의 성과**다.

    실제 성과가 이 분포 안에 있으면, 백테스트 지표는 모델의 방향 예측력을
    입증하지 못한다 — 규칙과 국면의 산물이다.
    """
    sh, cg, md = [], [], []
    for seed in range(n):
        rng = np.random.default_rng(seed)
        p = preds.copy()
        p["q50"] = p.groupby("date")["q50"].transform(
            lambda s: rng.permutation(s.to_numpy())
        )
        # 섞인 q50 이 그 종목의 [q10, q90] 밖으로 나가면 분위 교차가 되고,
        # QuantilePrediction 이 그 행을 통째로 버린다(후보 집합이 달라진다).
        # 자기 구간 안으로 되돌려 **순위만** 바뀌게 한다.
        p["q50"] = np.clip(p["q50"], p["q10"], p["q90"])
        r = run_backtest(p[["code", "date", "q10", "q50", "q90"]], prices, cfg)
        sh.append(r.metrics["sharpe"])
        cg.append(r.metrics["cagr"])
        md.append(r.metrics["max_drawdown"])

    sh_a, cg_a = np.array(sh), np.array(cg)
    out = {
        "n_permutations": n,
        "real": {"sharpe": real["sharpe"], "cagr": real["cagr"]},
        "shuffled_mean": {"sharpe": round(float(sh_a.mean()), 4),
                          "cagr": round(float(cg_a.mean()), 4),
                          "max_drawdown": round(float(np.mean(md)), 4)},
        "shuffled_sharpe_range": [round(float(sh_a.min()), 3), round(float(sh_a.max()), 3)],
        "shuffled_sharpe_std": round(float(sh_a.std(ddof=1)), 4),
        "p_value_sharpe": round(float((sh_a >= real["sharpe"]).mean()), 4),
        "p_value_cagr": round(float((cg_a >= real["cagr"]).mean()), 4),
        "signal_contribution": {
            "sharpe": round(float(real["sharpe"] - sh_a.mean()), 4),
            "cagr": round(float(real["cagr"] - cg_a.mean()), 4),
        },
    }
    log.info("순열검정 %d회 — 실제 Sharpe %.2f vs 셔플 평균 %.2f [%.2f, %.2f], p=%.3f",
             n, real["sharpe"], sh_a.mean(), sh_a.min(), sh_a.max(),
             out["p_value_sharpe"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lags", type=int, nargs="*", default=[1, 5, 20],
                    help="종목별로 미는 lag 목록. 비우면 시점 t 만 쓴다")
    ap.add_argument("--max-iter", type=int, default=600)
    ap.add_argument("--chunk", type=int, default=25,
                    help="early stopping 평가 간격(트리 수)")
    ap.add_argument("--patience", type=int, default=4, help="chunk 단위 인내")
    ap.add_argument("--learning-rate", type=float, default=0.03)
    ap.add_argument("--max-leaf", type=int, default=31)
    ap.add_argument("--l2", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--checkpoint")
    ap.add_argument("--no-phase1", action="store_true", help="Phase1 비교를 건너뛴다")
    ap.add_argument("--permutations", type=int, default=20,
                    help="음성 대조군 횟수. 0 이면 건너뛴다")
    args = ap.parse_args()

    setup_logging(run_name="gbdt_baseline")
    cfg = load_config().raw
    lags = sorted({x for x in args.lags if x > 0})

    # --- 1) 표 만들기
    table, feat_cols = build_table(cfg, lags)
    spec = SplitSpec.from_config(cfg)
    parts = split_by_date(table, spec)

    usable = {k: v.dropna(subset=feat_cols + ["target"]) for k, v in parts.items()}
    for k, v in usable.items():
        log.info("%-5s %7d행 (결측 제거 전 %d)", k, len(v), len(parts[k]))
    if not len(usable["train"]):
        raise SystemExit("train 이 비었다 — 피처/타깃 결측을 확인할 것")

    # --- 2) 무조건부 분위수 기준선. 이걸 못 이기면 학습이 의미 없다
    base_q = np.quantile(usable["train"]["target"].to_numpy(), QUANTILES)
    yva = usable["val"]["target"].to_numpy()
    baseline_val = pinball(yva, np.tile(base_q, (len(yva), 1)))
    log.info("기준선(무조건부 분위수) val pinball = %.6f  분위=%s",
             baseline_val, np.round(base_q, 5).tolist())

    # --- 3) GBDT 학습 — 분위별로 하나씩, 날짜 기준 val 로 early stopping
    Xtr = usable["train"][feat_cols].to_numpy("float32")
    ytr = usable["train"]["target"].to_numpy("float64")
    Xva = usable["val"][feat_cols].to_numpy("float32")
    Xte = usable["test"][feat_cols].to_numpy("float32")
    log.info("GBDT 학습 시작 — 분위 3개, 피처 %d개, train %d행", len(feat_cols), len(Xtr))

    va_cols, te_cols, fit_info = [], [], {}
    for q in QUANTILES:
        pv, pt, n_iter, loss = fit_quantile(
            q, Xtr, ytr, Xva, yva, Xte,
            chunk=args.chunk, max_iter=args.max_iter, patience=args.patience,
            seed=args.seed, learning_rate=args.learning_rate,
            max_leaf=args.max_leaf, l2=args.l2,
        )
        va_cols.append(pv)
        te_cols.append(pt)
        fit_info[f"q{int(q*100)}"] = {"n_trees": n_iter, "val_pinball": round(loss, 6)}

    qva, va_crossed = _sort_quantiles(np.column_stack(va_cols))
    gbdt_val = pinball(yva, qva)
    improvement = 100 * (baseline_val - gbdt_val) / baseline_val
    log.info("GBDT val pinball = %.6f — 기준선 대비 %+.2f%%", gbdt_val, improvement)

    gbdt_breakdown = quantile_breakdown(yva, qva, base_q)
    for k, v in gbdt_breakdown.items():
        log.info("  [GBDT] %s 기준선 %.6f → %.6f (%+.2f%%)",
                 k, v["baseline"], v["model"], v["improvement_pct"])

    # --- 4) test 예측
    qte, crossed = _sort_quantiles(np.column_stack(te_cols))
    te = usable["test"].copy()
    te[["q10", "q50", "q90"]] = qte
    if crossed:
        log.warning("test 분위 교차 %d건 (%.2f%%) — 정렬로 강제했다",
                    crossed, 100 * crossed / max(len(te), 1))

    gbdt_preds = te[["code", "date", "q10", "q50", "q90", "target"]].reset_index(drop=True)
    # 가격은 **test 구간 원본**. predict_split 이 내는 것과 같은 범위여야 한다.
    prices = parts["test"][["code", "date", "close"]].dropna().reset_index(drop=True)

    # --- 5) Phase1 과 예측 대상을 맞춘다
    ckpt = Path(args.checkpoint) if args.checkpoint else find_daily_checkpoint()
    phase1_preds = None
    phase1_breakdown: dict = {}
    phase1_val: dict = {}
    if not args.no_phase1 and ckpt and ckpt.exists():
        log.info("Phase1 비교 — %s", ckpt.name)
        phase1_preds, p1va = phase1_predictions(cfg, ckpt)

        # Phase1 의 val 표본 위에서 **그 표본의** 기준선과 비교한다
        y1 = p1va["target"].to_numpy()
        q1 = p1va[["q10", "q50", "q90"]].to_numpy()
        b1 = pinball(y1, np.tile(base_q, (len(y1), 1)))
        m1 = pinball(y1, q1)
        phase1_breakdown = quantile_breakdown(y1, q1, base_q)
        phase1_val = {
            "n_samples": int(len(p1va)), "baseline_val": round(b1, 6),
            "model_val": round(m1, 6),
            "improvement_vs_baseline_pct": round(100 * (b1 - m1) / b1, 3),
            "val_rank_ic": rank_ic(p1va),
        }
        log.info("Phase1 val pinball = %.6f — 기준선(같은 표본 %d개) %.6f 대비 %+.2f%%",
                 m1, len(p1va), b1, phase1_val["improvement_vs_baseline_pct"])
        for k, v in phase1_breakdown.items():
            log.info("  [Phase1] %s 기준선 %.6f → %.6f (%+.2f%%)",
                     k, v["baseline"], v["model"], v["improvement_pct"])

        keys = phase1_preds[["code", "date"]].drop_duplicates()
        before = len(gbdt_preds)
        gbdt_preds = gbdt_preds.merge(keys, on=["code", "date"], how="inner")
        log.info("예측 대상 교집합으로 정렬: GBDT %d → %d행 (Phase1 %d행)",
                 before, len(gbdt_preds), len(phase1_preds))
    elif not args.no_phase1:
        log.warning("일봉 체크포인트를 못 찾았다 — GBDT 만 평가한다")

    # --- 6) 평가
    results = {"gbdt": evaluate("GBDT", gbdt_preds, prices, cfg)}
    if phase1_preds is not None:
        results["phase1"] = evaluate("Phase1", phase1_preds, prices, cfg)

    # --- 음성 대조군. 신호를 뺀 파이프라인만의 성과를 잰다
    perm: dict = {}
    if args.permutations > 0:
        for key, src in (("phase1", phase1_preds), ("gbdt", gbdt_preds)):
            if src is None:
                continue
            perm[key] = permutation_test(
                src, prices, cfg, args.permutations, results[key]["backtest"]
            )

    bench_px = prices[prices["date"] >= gbdt_preds["date"].min()]
    results["buy_and_hold"] = {"backtest": summarize(buy_and_hold(bench_px))}
    log.info("[매수후보유] Sharpe %.2f | CAGR %.1f%%",
             results["buy_and_hold"]["backtest"]["sharpe"],
             100 * results["buy_and_hold"]["backtest"]["cagr"])

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": "GBDT(HistGradientBoosting quantile) vs Phase1 트랜스포머 공정 비교",
        "setup": {
            "lags": lags,
            "n_features": len(feat_cols),
            "split": str(spec),
            "rows": {k: int(len(v)) for k, v in usable.items()},
            "hyperparams": {
                "max_iter": args.max_iter, "chunk": args.chunk,
                "patience": args.patience, "learning_rate": args.learning_rate,
                "max_leaf_nodes": args.max_leaf, "l2": args.l2, "seed": args.seed,
            },
            "trees_per_quantile": fit_info,
            "checkpoint": ckpt.name if ckpt and phase1_preds is not None else None,
        },
        "quantile_loss": {
            "note": "두 모델의 val 표본 집합이 다르다(Phase1 은 lookback 120일 윈도우만). "
                    "각자의 표본 위에서 같은 train 무조건부 분위수와 비교한다.",
            "gbdt": {
                "n_samples": int(len(yva)),
                "baseline_val": round(baseline_val, 6),
                "model_val": round(gbdt_val, 6),
                "improvement_vs_baseline_pct": round(improvement, 3),
                "by_quantile": gbdt_breakdown,
                "val_crossings": va_crossed,
                "test_crossings": crossed,
            },
            "phase1": {**phase1_val, "by_quantile": phase1_breakdown} if phase1_val else None,
        },
        "results": results,
        "permutation_test": perm or None,
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = REPORTS_DIR / f"gbdt_baseline_{stamp}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str),
                   encoding="utf-8")
    log.info("리포트: %s", out.relative_to(PROJECT_ROOT))

    # --- 요약 출력
    print("\n" + "=" * 78)
    print("GBDT 베이스라인 vs Phase1 — 같은 split / 같은 백테스트 / 같은 예측 대상")
    print("=" * 78)
    print(f"  {'':<14}{'랭크 IC':>10}{'t':>7}{'스프레드':>11}"
          f"{'Sharpe':>9}{'CAGR':>9}{'MDD':>9}{'노출':>8}")
    for key, label in (("phase1", "Phase1"), ("gbdt", "GBDT"),
                       ("buy_and_hold", "매수후보유")):
        r = results.get(key)
        if not r:
            continue
        bt = r["backtest"]
        if "rank_ic" in r:
            print(f"  {label:<14}{r['rank_ic']['ic_mean']:>10.4f}"
                  f"{r['rank_ic']['t_stat']:>7.2f}"
                  f"{r['decile_spread']['spread_mean']:>11.4f}"
                  f"{bt['sharpe']:>9.2f}{100*bt['cagr']:>8.1f}%"
                  f"{100*bt['max_drawdown']:>8.1f}%"
                  f"{100*r['signal_stats']['avg_gross_exposure']:>7.1f}%")
        else:
            print(f"  {label:<14}{'—':>10}{'—':>7}{'—':>11}"
                  f"{bt['sharpe']:>9.2f}{100*bt['cagr']:>8.1f}%"
                  f"{100*bt['max_drawdown']:>8.1f}%{'100.0%':>8}")
    # 분위별로 갈라 보여준다 — 합친 pinball 은 "폭만 배웠다"를 가린다
    print(f"\n  val pinball 분위별 (기준선 대비 개선%)   ⚠️ 표본 집합이 서로 다르다")
    print(f"  {'':<14}{'q10':>10}{'q50':>10}{'q90':>10}{'전체':>10}{'표본':>9}")
    if phase1_breakdown:
        b = phase1_breakdown
        print(f"  {'Phase1':<14}{b['q10']['improvement_pct']:>9.2f}%"
              f"{b['q50']['improvement_pct']:>9.2f}%{b['q90']['improvement_pct']:>9.2f}%"
              f"{phase1_val['improvement_vs_baseline_pct']:>9.2f}%"
              f"{phase1_val['n_samples']:>9,}")
    b = gbdt_breakdown
    print(f"  {'GBDT':<14}{b['q10']['improvement_pct']:>9.2f}%"
          f"{b['q50']['improvement_pct']:>9.2f}%{b['q90']['improvement_pct']:>9.2f}%"
          f"{improvement:>9.2f}%{len(yva):>9,}")
    print("\n  ⚠️ q50 이 방향(순위)을 담당한다. 여기서 기준선을 못 이기면")
    print("     전체 pinball 개선은 폭(변동성)만 배운 것이다.")

    if perm:
        print(f"\n  음성 대조군 — q50 순위만 무작위화 ({args.permutations}회)")
        print(f"  {'':<14}{'실제':>8}{'셔플평균':>11}{'셔플범위':>18}{'p값':>8}{'신호기여':>10}")
        for key, label in (("phase1", "Phase1"), ("gbdt", "GBDT")):
            if key not in perm:
                continue
            t = perm[key]
            lo, hi = t["shuffled_sharpe_range"]
            print(f"  {label:<14}{t['real']['sharpe']:>8.2f}"
                  f"{t['shuffled_mean']['sharpe']:>11.2f}"
                  f"{f'[{lo:.2f}, {hi:.2f}]':>18}"
                  f"{t['p_value_sharpe']:>8.3f}"
                  f"{t['signal_contribution']['sharpe']:>+10.2f}")
        print("     (p 값이 크면 백테스트 Sharpe 는 신호가 아니라 파이프라인의 산물이다)")

    print(f"\n  리포트: {out.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
