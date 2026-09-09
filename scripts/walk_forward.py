#!/usr/bin/env python
"""Walk-forward 재학습 — 창마다 다시 배우고, 각 창의 **직후 구간만** 예측한다.

왜 필요한가:
    고정 split 은 2015~2022 로 한 번 배운 모델을 2024~2026 에 그대로 쓴다.
    그런데 팩터 구조가 그 사이 바뀐다 — 실측(2026-09-09) `mom_12_1` 의 랭크 IC 가
    train 에서 -1.25, test 에서 +3.51 로 **부호가 뒤집힌다.** 고정 모델은 옛 국면을
    배운 채로 새 국면을 맞히려 든다.

    walk-forward 는 창을 밀면서 매번 다시 배우고, **그 시점 이후만** 예측한다.
    각 예측은 자기보다 과거만 본 모델이 낸 것이므로 out-of-sample 이 유지된다.

    ⚠️ 그래도 미래는 안 본다 — 창의 test 구간은 그 창의 train/val 이 끝난 뒤다.
       예측을 이어붙인 결과가 곧 "그때그때 재학습했다면 얻었을 성과"다.

무엇을 고정하는가:
    같은 패널·같은 모델 설정·같은 매매 규칙·같은 비용. 바뀌는 건 학습 창뿐이다.
    비교 대상은 같은 구간을 도는 고정 split 모델이다.

사용:
    python scripts/walk_forward.py --profile idxrel2
    python scripts/walk_forward.py --profile idxrel2 --max-epochs 12   # 캐글에서 빠르게
    python scripts/walk_forward.py --preds outputs/reports/walkforward_<...>.parquet
        └ 이미 만든 예측으로 백테스트만 다시 (재학습 없음)
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.evaluation.backtest import run_backtest  # noqa: E402
from src.evaluation.benchmark import excess_return, index_returns  # noqa: E402
from src.evaluation.metrics import decile_spread, rank_ic, summarize  # noqa: E402
from src.models.inference import load_features, load_model, predict_range  # noqa: E402
from src.training.split import walk_forward_windows  # noqa: E402
from src.training.train import train  # noqa: E402
from src.utils.config import PROJECT_ROOT, load_config  # noqa: E402
from src.utils.logging import get_logger, setup_logging  # noqa: E402

log = get_logger("walk_forward")
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"


def _window_cfg(base: dict, tr_start, tr_end, va_end, te_end) -> dict:
    """이 창 하나만 보는 설정. 원본은 건드리지 않는다."""
    cfg = copy.deepcopy(base)
    cfg["split"].update(
        train_start=str(tr_start), train_end=str(tr_end),
        val_end=str(va_end), test_end=str(te_end),
    )
    return cfg


def _run_windows(base: dict, args, stamp: str) -> pd.DataFrame:
    """창마다 학습 → 그 창의 test 구간만 예측 → 이어붙인다.

    ⚠️ **창을 끝낼 때마다 곧바로 디스크에 쓴다.** 8창을 돌다 7번째에서 메모리 부족으로
    죽어 30분치를 통째로 잃은 적이 있다(2026-09-09). 마지막에 한 번만 저장하면
    중간에 죽는 순간 아무것도 안 남는다.
    """
    panel_dates = _panel_date_range(base)
    wf = base["backtest"]["walk_forward"]
    windows = walk_forward_windows(
        panel_dates[0], panel_dates[1],
        train_months=int(wf["train_months"]), val_months=int(wf["val_months"]),
        test_months=int(wf["test_months"]), step_months=int(wf["step_months"]),
    )
    # 마지막 창 뒤에 남는 구간을 버리지 않는다. train/val 은 그대로 과거에 두고
    # test 만 데이터 끝까지 늘린 **꼬리 창**을 하나 붙인다. 이게 없으면 가장 최근
    # 구간(실측 2026-01~09, 8개월)이 통째로 평가에서 빠진다.
    if windows and windows[-1][3] < panel_dates[1]:
        nxt = pd.Timestamp(windows[-1][0]) + pd.DateOffset(months=int(wf["step_months"]))
        tr_e = nxt + pd.DateOffset(months=int(wf["train_months"]))
        va_e = tr_e + pd.DateOffset(months=int(wf["val_months"]))
        if va_e.date() < panel_dates[1]:
            windows.append((nxt.date(), tr_e.date(), va_e.date(), panel_dates[1]))
            log.info("꼬리 창 추가: test %s ~ %s", va_e.date(), panel_dates[1])

    if args.since:
        since = pd.Timestamp(args.since).date()
        windows = [w for w in windows if w[3] > since]
    log.info("창 %d개 | %s ~ %s", len(windows), windows[0][0], windows[-1][3])

    frames, rows = [], []
    for i, (tr_s, tr_e, va_e, te_e) in enumerate(windows, start=1):
        log.info("[%d/%d] train %s~%s | val ~%s | test ~%s",
                 i, len(windows), tr_s, tr_e, va_e, te_e)
        cfg = _window_cfg(base, tr_s, tr_e, va_e, te_e)

        report = train(cfg, max_epochs=args.max_epochs)
        loaded = load_model(PROJECT_ROOT / report["checkpoint"])
        # 창의 test 구간만 예측한다. 데이터셋은 패널 전체에서 만든다 —
        # split 안에서만 만들면 앞 120일(lookback)이 날아가 6개월 창이 거의 빈다.
        preds = predict_range(loaded, load_features(cfg, loaded), cfg,
                              va_e + pd.Timedelta(days=cfg["split"]["embargo_days"]), te_e)
        if preds.empty:
            log.warning("  창 %d 의 test 예측이 비었다 — 건너뛴다", i)
            continue

        preds = preds.assign(window=i)
        frames.append(preds)
        part_path = REPORTS_DIR / f"walkforward_preds_{stamp}_w{i}.parquet"
        preds.to_parquet(part_path, index=False)
        ic = rank_ic(preds)
        rows.append({
            "window": i, "train_start": str(tr_s), "train_end": str(tr_e),
            "val_end": str(va_e), "test_end": str(te_e),
            "n_preds": len(preds),
            "test_from": str(preds["date"].min()), "test_to": str(preds["date"].max()),
            "val_loss": report["best_val_loss"], "best_epoch": report["best_epoch"],
            "improvement_pct": report["improvement_vs_baseline_pct"],
            "rank_ic": ic["ic_mean"], "t_stat": ic["t_stat"],
        })
        log.info("  → 예측 %d건 (%s~%s) | 랭크 IC %+.4f (t=%+.2f)",
                 len(preds), preds["date"].min(), preds["date"].max(),
                 ic["ic_mean"], ic["t_stat"])

    if not frames:
        raise SystemExit("어느 창에서도 예측이 안 나왔다")

    out = pd.concat(frames, ignore_index=True).sort_values(["date", "code"])
    # 창이 겹치면 같은 (code,date) 가 두 번 나온다. **더 최근에 학습된 창**을 남긴다.
    before = len(out)
    out = out.drop_duplicates(["code", "date"], keep="last").reset_index(drop=True)
    if before != len(out):
        log.info("겹치는 예측 %d건 제거 (더 최근 창을 남긴다)", before - len(out))

    pd.DataFrame(rows).to_json(
        REPORTS_DIR / f"walkforward_windows_{datetime.now():%Y%m%d_%H%M%S}.json",
        orient="records", force_ascii=False, indent=2,
    )
    return out


def _panel_date_range(cfg: dict) -> tuple:
    from src.data.storage import PROCESSED_DIR

    sfx = cfg["data"].get("processed_suffix", "")
    d = pd.read_parquet(PROCESSED_DIR / f"panel{sfx}.parquet", columns=["date"])["date"]
    return pd.to_datetime(d).min().date(), pd.to_datetime(d).max().date()


def _prices(cfg: dict) -> pd.DataFrame:
    from src.data.storage import PROCESSED_DIR

    sfx = cfg["data"].get("processed_suffix", "")
    p = pd.read_parquet(PROCESSED_DIR / f"panel{sfx}.parquet",
                        columns=["code", "date", "close"])
    return p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", help="config 의 profiles.<이름> (예: idxrel2)")
    ap.add_argument("--max-epochs", type=int,
                    help="창마다 최대 epoch. 창이 짧아 보통 일찍 멈춘다")
    ap.add_argument("--since", help="YYYY-MM-DD. 이 날짜 이후를 예측하는 창만 돈다")
    ap.add_argument("--preds", help="이미 만든 예측 parquet 로 백테스트만 다시 한다")
    args = ap.parse_args()

    setup_logging(run_name="walk_forward")
    cfg = load_config(profile=args.profile).raw
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = f"{datetime.now():%Y%m%d_%H%M%S}"

    if args.preds:
        preds = pd.read_parquet(args.preds)
        log.info("예측 %d건 재사용: %s", len(preds), args.preds)
    else:
        preds = _run_windows(cfg, args, stamp)
        path = REPORTS_DIR / f"walkforward_preds_{stamp}.parquet"
        preds.to_parquet(path, index=False)
        log.info("예측 저장: %s (%d건)", path.name, len(preds))

    # --- 이어붙인 예측으로 진단 + 백테스트. 고정 split 과 **같은 함수**를 쓴다.
    ic, spread = rank_ic(preds), decile_spread(preds)
    log.info("이어붙인 예측 %d건 | %s ~ %s",
             len(preds), preds["date"].min(), preds["date"].max())
    log.info("랭크 IC %+.4f (t=%+.2f, %d일) | 십분위 스프레드 %+.4f (t=%+.2f)",
             ic["ic_mean"], ic["t_stat"], ic["n_dates"],
             spread["spread_mean"], spread["t_stat"])

    prices = _prices(cfg)
    variants = {
        "기본규칙": {},
        "시총가중+기권75": {"method": "cap_weighted", "max_position_pct": 1.0,
                            "percentile": 75},
    }
    results = {}
    for label, ov in variants.items():
        c = copy.deepcopy(cfg)
        if ov:
            c["trading"]["sizing"].update(
                method=ov["method"], max_position_pct=ov["max_position_pct"])
            c["trading"]["abstain"]["percentile"] = ov["percentile"]
        res = run_backtest(preds, prices, c)
        idx = index_returns("201", pd.Index(res.returns.index))
        bench = summarize(idx) if idx is not None else None
        ex = (excess_return(res.metrics["total_return"], bench["total_return"])
              if bench else None)
        results[label] = {"metrics": res.metrics, "index": bench,
                          "excess_vs_index": ex,
                          "annual_turnover": res.signal_stats["annual_turnover"]}
        log.info(
            f"{label:<16} 누적 {res.metrics['total_return']:+8.1%}  "
            f"Sharpe {res.metrics['sharpe']:5.2f}  "
            f"MDD {res.metrics['max_drawdown']:7.1%}  "
            f"회전 {res.signal_stats['annual_turnover']:5.1f}  "
            f"초과 {f'{ex:+.1%}' if ex is not None else '—'}"
        )

    out = REPORTS_DIR / f"walkforward_{stamp}.json"
    out.write_text(json.dumps(
        {"timestamp": datetime.now().isoformat(timespec="seconds"),
         "profile": args.profile, "n_preds": len(preds),
         "period": [str(preds["date"].min()), str(preds["date"].max())],
         "diagnostics": {"rank_ic": ic, "decile_spread": spread},
         "results": results},
        ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    log.info("저장: %s", out.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
