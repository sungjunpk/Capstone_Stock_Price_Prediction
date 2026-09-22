#!/usr/bin/env python
"""원본 PatchTST · TFT vs 우리 모델 — 같은 조건에서 정면 비교한다.

왜: 구조도는 무엇이 다른지 보여주지만 **그 차이가 이득이었는지는 말해주지 않는다.**
같은 데이터·같은 split·같은 학습 루프·같은 seed 로 원본 둘을 직접 학습시켜
숫자로 답한다.

사다리 — 각 단계가 정확히 하나를 더한다:
    patchtst    PatchTST 원본(MSE·점예측)  → 불확실성이 없어 **기권이 불가능**
    patchtst_q  + 분위 헤드                → 여기서부터 기권 로직이 성립 (차별점 1)
    tft         TFT 원본(VSN+LSTM, 패치 없음) → 변수선택·정적문맥 (차별점 2)
    ours        우리 모델 = 위 둘의 결합 + 매크로 크로스어텐션

사용:
    python scripts/arch_compare.py --smoke          # 배관 점검 (6종목 2epoch, 수 분)
    python scripts/arch_compare.py --train patchtst # 하나씩 학습한다 (결과 보고 다음으로)
    python scripts/arch_compare.py --eval           # 학습된 것 전부 평가 → 비교 JSON

⚠️ 체크포인트에 `_patchtst` 등 태그가 붙는다. `scripts/paper_trade.py` 는 무태그
   최신만 집으므로 대조군이 실주문 경로로 샐 수 없다 (2026-09-14 에 실제로 났던 사고).
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.evaluation.backtest import run_backtest  # noqa: E402
from src.evaluation.benchmark import index_returns  # noqa: E402
from src.evaluation.metrics import equity_curve, summarize  # noqa: E402
from src.models.inference import load_features, load_model, predict_split  # noqa: E402
from src.training.train import train  # noqa: E402
from src.utils.config import PROJECT_ROOT, load_config  # noqa: E402
from src.utils.logging import get_logger, setup_logging  # noqa: E402

log = get_logger("arch_compare")
CKPT_DIR = PROJECT_ROOT / "outputs" / "checkpoints"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"

# 우리 모델은 **지금 실거래에 쓰는 바로 그 체크포인트**다. 재학습하지 않는다.
OURS_CKPT = "phase1_eadf265f.pt"

ARCHS: dict[str, dict] = {
    "patchtst": {
        "label": "PatchTST 원본",
        "loss": "mse",
        "note": "점 예측. 매크로·static·변수선택 없음",
    },
    "patchtst_q": {
        "label": "PatchTST + 분위헤드",
        "loss": "pinball",
        "note": "불확실성 추가 — 기권이 성립한다",
    },
    "tft": {
        "label": "TFT 원본",
        "loss": "pinball",
        "note": "변수선택·정적문맥. 패치 없이 120시점, 매크로는 관측입력으로 concat",
    },
    # 아래 둘은 원본 재구현이 아니라 **우리 모델의 모듈 교체판**이다
    # (`src/models/baselines.py` 의 VARIANT_ARCHS 참고).
    "itrans": {
        "label": "iTransformer (백본 교체)",
        "loss": "pinball",
        "note": "종목 경로를 변수 토큰으로. VSN·정적문맥·매크로 크로스어텐션은 우리 모델 그대로",
    },
    "timexer": {
        "label": "TimeXer (외생처리 교체)",
        "loss": "pinball",
        "note": "매크로를 채널당 토큰 하나로. 종목 경로는 패치 그대로",
    },
}


def arch_config(base: dict, arch: str, *, batch_size: int | None) -> dict:
    """base 설정에서 **아키텍처와 손실만** 바꾼다. 나머지는 전부 고정 — 그게 통제다."""
    c = copy.deepcopy(base)
    c["model"]["arch"] = arch
    c["training"]["loss"] = ARCHS[arch]["loss"]
    # 실험 산출물임을 파일명에 박는다. 무태그로 떨어지면 실주문 경로가 집어간다.
    c["data"]["checkpoint_suffix"] = (
        c["data"].get("processed_suffix", "") + f"_{arch}"
    )
    if batch_size:
        c["training"]["batch_size"] = batch_size
    return c


def find_checkpoint(arch: str, base: dict, *, smoke: bool) -> Path | None:
    """해당 아키텍처의 최신 체크포인트. 태그로 고른다."""
    if arch == "ours":
        p = CKPT_DIR / OURS_CKPT
        return p if p.exists() else None
    sfx = base["data"].get("processed_suffix", "")
    pattern = f"phase1_*{sfx}_{arch}{'_smoke' if smoke else ''}.pt"
    hits = sorted(CKPT_DIR.glob(pattern), key=lambda p: p.stat().st_mtime)
    # smoke 가 아닐 때 _smoke 파일이 섞이면 안 된다
    if not smoke:
        hits = [h for h in hits if not h.stem.endswith("_smoke")]
    return hits[-1] if hits else None


# --------------------------------------------------------------------- 학습
def do_train(names: list[str], base: dict, *, smoke: bool, epochs: int | None,
             batch_size: int | None) -> None:
    for i, arch in enumerate(names, 1):
        log.info("=" * 66)
        log.info("[%d/%d] %s — %s", i, len(names), arch, ARCHS[arch]["label"])
        log.info("=" * 66)
        cfg = arch_config(base, arch, batch_size=batch_size)
        t0 = time.time()
        r = train(cfg, smoke=smoke, max_epochs=epochs)
        log.info(
            "%s 완료 | %.2fM | val pinball %.6f (기준선 대비 %+.2f%%) | best epoch %d | %.1f분",
            arch, r["n_params"] / 1e6, r["best_val_pinball"],
            r["improvement_vs_baseline_pct"], r["best_epoch"],
            (time.time() - t0) / 60,
        )


# --------------------------------------------------------------------- 평가
def latest_train_report(arch: str) -> dict | None:
    """해당 아키텍처의 가장 최근 학습 리포트. 파라미터 수·epoch 시간을 여기서 가져온다.

    우리 모델은 재학습하지 않으므로 `arch` 키가 없던 시절(2026-09-09, 캐글)의
    리포트를 **체크포인트 이름으로** 찾는다. 실거래 사본 `phase1_eadf265f.pt` 는
    실험 트랙 `..._idxrel2.pt` 의 무태그 복사본이라 리포트는 후자를 가리킨다.
    """
    ours_stem = OURS_CKPT.removesuffix(".pt")
    best = None
    for p in REPORT_DIR.glob("2*.json"):
        try:
            r = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(r, dict) or r.get("smoke"):
            continue
        if arch == "ours":
            if not Path(str(r.get("checkpoint", ""))).stem.startswith(ours_stem):
                continue
        elif r.get("arch") != arch:
            continue
        if best is None or r["timestamp"] > best["timestamp"]:
            best = r
    return best


def evaluate_checkpoint(ckpt: Path, cfg: dict, *, months: int, end: str | None = None) -> dict:
    """체크포인트 하나 → 예측력 진단 + 매매 성과. **모든 모델이 이 함수를 탄다.**

    `end` 를 주면 그 날짜를 구간 끝으로 못박는다. 나중에 모델 하나를 추가로 평가할 때
    **먼저 평가한 모델들과 같은 구간**에 세우기 위해서다 — 기본값(패널 마지막 날)으로
    두면 수집이 며칠 더 쌓인 만큼 구간이 밀려 이전 숫자와 비교가 안 된다.
    """
    loaded = load_model(ckpt)
    bundle = load_features(cfg, loaded)
    preds, prices = predict_split(loaded, bundle, cfg, "test")

    end = pd.Timestamp(end).date() if end else max(prices["date"])
    start = (pd.Timestamp(end) - pd.DateOffset(months=months)).date()
    w_preds = preds[(preds["date"] >= start) & (preds["date"] <= end)]
    pad = (pd.Timestamp(start) - pd.DateOffset(days=20)).date()
    w_prices = prices[(prices["date"] >= pad) & (prices["date"] <= end)]

    res = run_backtest(w_preds, w_prices, cfg, width_history=preds)
    dates = list(res.returns.index)
    bench = index_returns(str(cfg["backtest"]["benchmark_index"]), pd.Index(dates))

    width = (w_preds["q90"] - w_preds["q10"])
    return {
        "checkpoint": ckpt.name,
        "window": {"start": str(dates[0]), "end": str(dates[-1]), "n_days": len(dates)},
        "metrics": res.metrics,
        "benchmark": summarize(bench) if bench is not None else None,
        # 누적수익 곡선 — 모델끼리 한 차트에 겹쳐 그린다. 날짜축은 네 모델이 공유한다
        # (같은 구간·같은 가격 데이터라 run_backtest 가 같은 인덱스를 낸다).
        "dates": [str(d) for d in dates],
        "curve": [round(float(v), 6) for v in equity_curve(res.returns).tolist()],
        "benchmark_curve": (
            [round(float(v), 6) for v in equity_curve(bench).tolist()]
            if bench is not None else None),
        "diagnostics": res.diagnostics,
        "signal_stats": res.signal_stats,
        # 차별점 1 의 직접 증거 — 폭이 0 이면 기권 판정 자체가 성립하지 않는다
        "interval": {
            "width_mean": round(float(width.mean()), 6),
            "width_p50": round(float(width.median()), 6),
            "width_zero_rate": round(float((width.abs() < 1e-9).mean()), 4),
        },
    }


def do_eval(names: list[str], base: dict, *, months: int, smoke: bool,
            end: str | None = None) -> Path:
    rows = []
    for arch in names:
        ckpt = find_checkpoint(arch, base, smoke=smoke)
        if ckpt is None:
            log.warning("%s — 체크포인트 없음, 건너뛴다", arch)
            continue
        log.info("─" * 66)
        log.info("%s 평가 — %s", arch, ckpt.name)
        ev = evaluate_checkpoint(ckpt, base, months=months, end=end)
        tr = latest_train_report(arch)
        meta = ARCHS.get(arch, {"label": "우리 모델 (Phase 1)", "loss": "pinball",
                                "note": "PatchTST 백본 + TFT 변수선택 + 매크로 크로스어텐션"})
        rows.append({
            "arch": arch, "label": meta["label"], "loss": meta["loss"],
            "note": meta["note"],
            "train": {
                "n_params": tr["n_params"] if tr else None,
                # 옛 리포트(우리 모델, 2026-09-09)에는 이 키가 없다 — 당시엔 pinball 로만
                # 학습했으므로 감시 손실이 곧 pinball 이다.
                "best_val_pinball": (
                    tr.get("best_val_pinball", tr.get("best_val_loss")) if tr else None),
                "baseline_val_loss": tr["baseline_val_loss"] if tr else None,
                "improvement_vs_baseline_pct": (
                    tr["improvement_vs_baseline_pct"] if tr else None),
                "best_epoch": tr["best_epoch"] if tr else None,
                "sec_per_epoch": (
                    round(sum(h["sec"] for h in tr["history"]) / len(tr["history"]), 1)
                    if tr and tr.get("history") else None),
                "feature_importance": (
                    dict(list(tr["feature_importance"].items())[:5]) if tr else None),
                "device": tr["device"] if tr else None,
            },
            **ev,
        })
        m, d = ev["metrics"], ev["diagnostics"]["rank_ic"]
        log.info(
            "  누적 %+.1f%% | Sharpe %.2f | MDD %+.1f%% | IC %+.4f (t %+.2f) | "
            "기권률 %.1f%% | 폭 중앙 %.4f",
            100 * m["total_return"], m["sharpe"], 100 * m["max_drawdown"],
            d["ic_mean"], d["t_stat"],
            100 * ev["signal_stats"]["abstain_rate"], ev["interval"]["width_p50"],
        )

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "window_months": months,
        "benchmark": "코스피200",
        "controls": {
            "same": ["패널 데이터", "train/val/test split", "정규화 통계",
                     "seed", "학습 루프", "매매 규칙·비용", "백테스트 구간"],
            "differs": ["아키텍처", "손실 함수", "모델이 보는 입력(아래 note)"],
            "caveats": [
                "우리 모델만 Kaggle(CUDA)에서 학습됐다 — 대조군은 로컬 MPS. "
                "train/val 구간 데이터는 동일하다",
                "PatchTST 계열은 매크로·static 을 안 본다. 원본에 그 개념이 없어서이고, "
                "따라서 순수 아키텍처 비교가 아니라 '원본을 그대로 쓰면 무엇을 못 보는가'의 비교다",
                "하이퍼파라미터(d_model 32 · 1층 · dropout 0.5)는 우리 구조에 맞춰 "
                "튜닝된 값이다. 대조군에도 같은 값을 줬지만 그것이 대조군의 최적은 아니다",
            ],
        },
        "models": rows,
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / f"arch_compare_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("=" * 66)
    log.info("비교 리포트: %s", out.relative_to(PROJECT_ROOT))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="*", metavar="ARCH",
                    help=f"학습할 아키텍처. 비우면 전부: {list(ARCHS)}")
    ap.add_argument("--eval", nargs="*", metavar="ARCH",
                    help="학습된 체크포인트를 평가한다. 아키텍처를 적으면 그것만 — "
                         "이미 평가한 모델의 숫자를 다시 건드리지 않을 때 쓴다")
    ap.add_argument("--end", metavar="YYYY-MM-DD",
                    help="백테스트 구간의 끝을 못박는다 (기본: 패널 마지막 날). "
                         "나중에 추가한 모델을 기존 모델과 같은 구간에 세울 때 필요하다")
    ap.add_argument("--smoke", action="store_true", help="6종목 2epoch 배관 점검")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--months", type=int, default=24, help="백테스트 구간 (기본 최근 2년)")
    args = ap.parse_args()

    setup_logging(run_name="arch_compare")
    base = load_config().raw

    if args.train is None and args.eval is None and not args.smoke:
        ap.error("--train 또는 --eval 또는 --smoke 중 하나가 필요하다")

    if args.smoke and args.train is None:
        args.train = []

    if args.train is not None:
        names = args.train or list(ARCHS)
        unknown = [n for n in names if n not in ARCHS]
        if unknown:
            log.error("모르는 아키텍처: %s (가능: %s)", unknown, list(ARCHS))
            return 1
        do_train(names, base, smoke=args.smoke,
                 epochs=args.epochs, batch_size=args.batch_size)

    if args.eval is not None:
        names = args.eval or [*ARCHS, "ours"]
        unknown = [n for n in names if n not in ARCHS and n != "ours"]
        if unknown:
            log.error("모르는 아키텍처: %s (가능: %s)", unknown, [*ARCHS, "ours"])
            return 1
        do_eval(names, base, months=args.months, smoke=args.smoke, end=args.end)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
