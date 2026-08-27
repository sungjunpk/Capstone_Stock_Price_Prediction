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
import copy
import json
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.evaluation.backtest import buy_and_hold, run_backtest  # noqa: E402
from src.evaluation.metrics import summarize  # noqa: E402
from src.training.train import _config_hash  # noqa: E402
from src.models.inference import (  # noqa: E402
    load_features,
    load_model,
    predict_split,
)
from src.utils.config import PROJECT_ROOT, load_config  # noqa: E402
from src.utils.logging import get_logger, setup_logging  # noqa: E402

log = get_logger("backtest")
CKPT_DIR = PROJECT_ROOT / "outputs" / "checkpoints"


def find_checkpoint(explicit: str | None, cfg: dict | None = None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if not p.exists():
            raise SystemExit(f"체크포인트가 없다: {p}")
        return p

    # 트랙이 둘(일봉/60분봉)이라 '최신'만으로 고르면 **다른 트랙 것을 집는다.**
    # 그러면 에러가 아니라 그럴듯하게 틀린 숫자가 나온다 — 가장 나쁜 실패다.
    # 학습이 이름에 트랙 태그를 붙이므로(train.py) 여기서 그 태그로 좁힌다.
    tag = (cfg or {}).get("data", {}).get("processed_suffix", "")
    pattern = re.compile(rf"^phase1_[0-9a-f]{{8}}{re.escape(tag)}\.pt$")
    cands = [p for p in CKPT_DIR.glob("phase1_*.pt") if pattern.match(p.name)]
    if not cands:
        raise SystemExit(
            f"이 트랙(태그 {tag!r})의 체크포인트가 없다. 학습을 먼저 하거나,\n"
            "  캐글에서 받은 .pt 를 outputs/checkpoints/ 에 두고 --checkpoint 로 지정할 것."
        )

    if cfg is not None:
        exact = CKPT_DIR / f"phase1_{_config_hash(cfg)}{tag}.pt"
        if exact.exists():
            log.info("체크포인트 선택(설정 해시 일치): %s", exact.name)
            return exact
        log.warning("이 설정 해시로 학습된 체크포인트가 없다 — 같은 트랙의 최신 것을 쓴다")

    latest = max(cands, key=lambda p: p.stat().st_mtime)
    log.info("체크포인트 자동 선택: %s", latest.name)
    return latest


def predict(ckpt_path: Path, cfg: dict, split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """지정 구간에 대해 (예측 q10/q50/q90, 주가) 를 만든다.

    추론 경로는 `src/models/inference.py` 하나뿐이다 — 모의투자도 같은 코드를 쓴다.
    예측을 종목·날짜에 붙이는 일도 거기(`WindowDataset.sample_keys`)서 한다.
    여기서 필터·정렬을 재현하면 조건 하나가 어긋나도 조용히 틀린다.
    """
    loaded = load_model(ckpt_path)
    bundle = load_features(cfg, loaded)
    return predict_split(loaded, bundle, cfg, split)


# 규칙 변형 — 예측을 한 번만 계산하고 규칙만 갈아끼워 한 세션에서 비교한다.
# 각 항목은 (이름, 덮어쓸 설정) 이고, 설정 경로는 trading 아래 3개 섹션이다.
# "버퍼없음" 이 변경 이전 상태이므로 비교의 기준점이 된다.
RULE_VARIANTS: list[tuple[str, dict]] = [
    ("버퍼없음", {"exit_rank": None, "min_trade_weight": 0.0}),
    ("+버퍼", {"min_trade_weight": 0.0}),
    ("+버퍼+밴드", {}),
    ("+익절해제", {"take_profit_pct": 99.0}),
    ("absolute", {"mode": "absolute", "min_trade_weight": 0.0}),
]


def _variant_cfg(base: dict, override: dict) -> dict:
    """기본 설정에 변형을 얹은 사본. 원본은 건드리지 않는다."""
    cfg = copy.deepcopy(base)
    d, s, r = cfg["trading"]["direction"], cfg["trading"]["sizing"], cfg["trading"]["risk"]

    if "mode" in override:
        d["mode"] = override["mode"]
    if "exit_rank" in override:
        # None = 버퍼 없음 → exit_rank 를 top_n 과 같게 두면 버퍼가 꺼진다
        d["exit_rank"] = override["exit_rank"] or int(d["top_n"])
    if "min_trade_weight" in override:
        s["min_trade_weight"] = override["min_trade_weight"]
    if "take_profit_pct" in override:
        r["take_profit_pct"] = override["take_profit_pct"]
    return cfg


def _print_diagnostics(result) -> None:
    """예측력 진단 — 성과보다 먼저 읽어야 한다.

    방향 예측력이 없으면 아래 성과 지표는 매매 규칙의 부산물일 뿐이다.
    매매 규칙과 무관하므로 변형끼리 값이 같다 — 비교 모드에서는 한 번만 찍는다.
    """
    if not result.diagnostics:
        return
    ic = result.diagnostics["rank_ic"]
    spread = result.diagnostics["decile_spread"]
    cost = result.signal_stats["round_trip_cost"]

    print("\n예측력 진단 — 매매 규칙을 우회해 '방향을 맞히는가'만 본다")
    print(f"  랭크 IC        {ic['ic_mean']:+.4f}  (t={ic['t_stat']:+.2f}, "
          f"{ic['n_dates']}일, 양수비율 {ic['ic_positive_rate']:.1%})")
    print(f"  십분위 스프레드 {spread['spread_mean']:+.4f}  "
          f"(t={spread['t_stat']:+.2f})   왕복비용 {cost:.4f}")

    if abs(ic["t_stat"]) < 2:
        verdict = "방향 예측력 확인 안 됨 (|t| < 2) — 매매 규칙을 손봐도 성과는 안 나온다"
    elif spread["spread_mean"] <= cost:
        verdict = "순위는 맞히지만 스프레드가 거래비용 이하 — 회전을 줄여야 한다"
    else:
        verdict = "방향 알파 있음 — 거래비용을 넘는다"
    print(f"  판정           {verdict}")


def _print_single(result, bh: dict, split: str) -> None:
    print("\n" + "=" * 66)
    print(f"백테스트 결과 ({split} 구간, 거래비용 반영)")
    print("=" * 66)
    print(f"{'지표':<16}{'전략':>14}{'매수후보유':>14}")
    for k in ("cagr", "volatility", "sharpe", "sortino", "calmar",
              "max_drawdown", "hit_rate", "total_return"):
        print(f"  {k:<14}{result.metrics[k]:>14.4f}{bh[k]:>14.4f}")

    _print_diagnostics(result)
    s = result.signal_stats

    print("\n신호 통계 — 기권 로직이 이 프로젝트의 핵심 차별점")
    print(f"  판단 방식    {s['mode']}"
          + (f" (상위 {s['top_n']}, 청산 {s['exit_rank']}위 밖)"
             if s['mode'] == 'cross_sectional' else ""))
    print(f"  총 판단      {s['decisions']:,}회")
    print(f"  기권률       {s['abstain_rate']:.1%}   (신뢰구간이 넓어 관망)")
    print(f"  실거래율     {s['trade_rate']:.1%}")
    print(f"  체결 건수    {s['n_trades']:,}")
    print(f"  왕복비용     {s['round_trip_cost']:.2%}")

    if s["blocked_by_reason"]:
        print("\n  리스크 차단 사유")
        for reason, cnt in s["blocked_by_reason"].items():
            print(f"    {reason:<10} {cnt:,}회")

    print("\n  실제로 투자했는가 — 이게 0 에 가까우면 위 성과는 읽을 의미가 없다")
    print(f"    평균 노출도    {s['avg_gross_exposure']:.1%}")
    print(f"    평균 보유종목  {s['avg_n_positions']:.1f}개")
    print(f"    평균 보유일수  {s['avg_holding_days']:.1f}일")
    print(f"    연 회전율      {s['annual_turnover']:.1f}회 "
          f"(리밸런싱 {s['n_rebalances']}회)")
    print(f"    실지불 거래비용 연 {s['annual_cost_pct']:.2%} "
          f"(누계 {s['total_cost_pct']:.2%})")

    if s.get("hold_until_exit"):
        print("\n  타점 탐지 — 드물게, 정확하게 들어갔는가")
        print(f"    진입 횟수      {s['n_entries']:,}회 (연 {s['annual_entries']:.0f}회)")
        print(f"    무포지션 비율  {s['flat_rate']:.1%}  (조건이 안 맞아 쉰 구간)")


def _print_compare(runs: list[tuple[str, object]], bh: dict, split: str) -> None:
    """규칙 변형 나란히 비교. 격차 중 비용 몫과 종목선택 몫을 가르는 게 목적이다."""
    names = [n for n, _ in runs]
    w = max(12, max(len(n) for n in names) + 2)

    print("\n" + "=" * 66)
    print(f"매매 규칙 비교 ({split} 구간, 거래비용 반영)")
    print("=" * 66)
    print(f"  {'지표':<16}" + "".join(f"{n:>{w}}" for n in names) + f"{'매수후보유':>{w}}")
    for k in ("sharpe", "sortino", "calmar", "max_drawdown",
              "volatility", "cagr", "total_return"):
        row = "".join(f"{r.metrics[k]:>{w}.4f}" for _, r in runs)
        print(f"    {k:<14}" + row + f"{bh[k]:>{w}.4f}")

    print("\n  거래 활동 — 회전율이 내려가면 비용도 같이 내려가야 한다")
    rows = [
        ("연 회전율", "annual_turnover", f">{w}.1f"),
        ("실지불비용(연)", "annual_cost_pct", f">{w}.2%"),
        ("평균 보유일수", "avg_holding_days", f">{w}.1f"),
        ("평균 노출도", "avg_gross_exposure", f">{w}.1%"),
        ("평균 보유종목", "avg_n_positions", f">{w}.1f"),
        ("체결 건수", "n_trades", f">{w},d"),
    ]
    for label, key, fmt in rows:
        print(f"    {label:<14}"
              + "".join(f"{r.signal_stats[key]:{fmt}}" for _, r in runs))

    print("\n  리스크 차단 사유")
    reasons = sorted({k for _, r in runs for k in r.signal_stats["blocked_by_reason"]})
    for reason in reasons:
        print(f"    {reason:<14}"
              + "".join(f"{r.signal_stats['blocked_by_reason'].get(reason, 0):>{w},d}"
                        for _, r in runs))


def _write_report(result, bh: dict, ckpt_path: Path, args, label: str) -> Path:
    """실험 결과는 날짜+체크포인트로 남기고 덮어쓰지 않는다 (절대 규칙 8)."""
    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "checkpoint": ckpt_path.name, "split": args.split,
        "variant": label,
        "diagnostics": result.diagnostics,
        "allow_short": args.allow_short,
        "strategy": result.metrics, "buy_and_hold": bh,
        "signal_stats": result.signal_stats,
    }
    suffix = f"_{label}" if label else ""
    out = PROJECT_ROOT / "outputs" / "reports" / (
        f"backtest_{datetime.now():%Y%m%d_%H%M%S}_{ckpt_path.stem}{suffix}.json"
    )
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--allow-short", action="store_true",
                    help="공매도 허용 (기본은 매수 전용 — 모의투자 현실 반영)")
    ap.add_argument("--mode", choices=["absolute", "cross_sectional"],
                    help="방향 판단 방식. config 의 trading.direction.mode 를 덮어쓴다")
    ap.add_argument("--exposure", choices=["scaled", "fixed"],
                    help="scaled=생존 후보 수에 비례해 노출 조절, fixed=항상 상한까지")
    ap.add_argument("--compare", action="store_true",
                    help="규칙 변형을 한 번에 비교 (예측은 한 번만 계산한다)")
    ap.add_argument("--profile", help="config 의 profiles.<이름> 을 덮어쓴다 (예: intraday)")
    args = ap.parse_args()

    setup_logging(run_name="backtest")
    cfg = load_config(profile=args.profile).raw

    # 같은 체크포인트로 여러 매매 규칙을 비교할 수 있게 설정을 덮어쓴다.
    # 신호 코드는 하나뿐이고 분기는 설정값에만 있다 (CLAUDE.md 절대 규칙 7).
    if args.mode:
        cfg["trading"]["direction"]["mode"] = args.mode
    if args.exposure:
        cfg["trading"]["sizing"]["exposure_scaling"] = args.exposure == "scaled"

    ckpt_path = find_checkpoint(args.checkpoint, cfg)
    preds, prices = predict(ckpt_path, cfg, args.split)
    log.info("예측 %d건 | 종목 %d | %s ~ %s",
             len(preds), preds["code"].nunique(), preds["date"].min(), preds["date"].max())

    periods = float(cfg.get("backtest", {}).get("bars_per_year", 252))
    bh = summarize(buy_and_hold(prices), periods_per_year=periods)

    if not args.compare:
        log.info("매매 규칙: mode=%s | exposure_scaling=%s",
                 cfg["trading"]["direction"]["mode"],
                 cfg["trading"]["sizing"]["exposure_scaling"])
        result = run_backtest(preds, prices, cfg, allow_short=args.allow_short)
        _print_single(result, bh, args.split)
        out = _write_report(result, bh, ckpt_path, args, "")
        print(f"\n저장: {out.relative_to(PROJECT_ROOT)}")
        return 0

    # --- 비교 모드: 예측은 위에서 한 번만 계산했다. 규칙만 갈아끼운다
    runs = []
    for label, override in RULE_VARIANTS:
        log.info("규칙 변형 실행: %s", label)
        result = run_backtest(preds, prices, _variant_cfg(cfg, override),
                              allow_short=args.allow_short)
        runs.append((label, result))
        _write_report(result, bh, ckpt_path, args, label)

    _print_diagnostics(runs[0][1])      # 규칙과 무관하므로 한 번만
    _print_compare(runs, bh, args.split)
    print(f"\n저장: outputs/reports/ 에 변형 {len(runs)}개")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
