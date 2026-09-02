#!/usr/bin/env python
"""피처 진단 — 단변량 팩터 검증 × VSN 학습 중요도 대조.

두 가지를 한 번에 묻는다. 둘 다 **읽기 전용**이다 — 매매 설정을 바꾸지 않는다.

(1) 팩터 사전검증 (국내 팩터투자 연구 방법론)
    "TFT 에 넣기 전에 각 피처가 독립적으로 예측력이 있는가"를 단변량 랭크 IC 로 잰다.
    ⚠️ **결정에 쓰는 IC 는 train 구간에서만 계산한다** (규칙 5·6). test IC 는
       "그 사전선별이 실제로 유지됐는가"를 사후에 보기 위해 같이 찍을 뿐이다.
    ⚠️ 이 결과로 피처를 잘라내지 않는다. 자동 변수선택(VSN)이 이 프로젝트의
       핵심 차별점이라 앞단에서 손으로 자르면 그 주장이 약해지고, IC 기준
       사전선별 자체가 선택편향이다. 여기서는 **대조표**로만 쓴다.

(2) 해석의 안정성 (KCMI 의 "안전 환상" 경고를 실측으로 옮긴 것)
    자본시장연구원은 AI 자산운용의 위험으로 "그럴듯하지만 실제 논리와 불일치하는
    설명"을 든다. VSN 중요도를 리포트 근거로 쓰려면 그 중요도가 **흔들리지 않는지**
    부터 보여야 한다. 여기서는 재학습 없이 두 축으로 잰다:
        - 연도별(국면별) 중요도 순위가 유지되는가
        - 날짜 부트스트랩에서 순위가 얼마나 흔들리는가
    (시드 축은 재학습이 필요하다 — `--help` 아래 주석 참조)

사용:
    python scripts/feature_diagnostics.py                  # test 구간 기준
    python scripts/feature_diagnostics.py --split val
    python scripts/feature_diagnostics.py --bootstrap 500
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

from src.models.inference import (  # noqa: E402
    load_features,
    load_model,
    vsn_weights_split,
)
from src.evaluation.metrics import rank_ic  # noqa: E402
from src.training.split import split_by_date  # noqa: E402
from src.utils.config import PROJECT_ROOT, load_config  # noqa: E402
from src.utils.logging import get_logger, setup_logging  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

log = get_logger("feature_diagnostics")
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
CKPT_DIR = PROJECT_ROOT / "outputs" / "checkpoints"


def find_checkpoint(explicit: str | None) -> Path:
    """일봉 트랙 체크포인트. 60분봉(`_60m`)·변형(`_xs`/`_mr`)은 제외한다."""
    if explicit:
        return Path(explicit)
    daily = re.compile(r"^phase1_[0-9a-f]{8}\.pt$")
    cands = [p for p in CKPT_DIR.glob("phase1_*.pt") if daily.match(p.name)]
    if not cands:
        raise SystemExit("일봉 체크포인트가 없다 — --checkpoint 로 지정할 것.")
    return max(cands, key=lambda p: p.stat().st_mtime)


# ------------------------------------------------------------------ (1) 단변량 IC
def univariate_ic(panel: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """피처 하나를 그대로 점수로 써서 날짜별 랭크 IC 를 낸다.

    `rank_ic` 는 q50 컬럼을 점수로 본다 — 피처를 그 자리에 넣으면 그대로 단변량
    검증이 된다. 순위 기반이라 정규화 여부가 결과를 바꾸지 않는다(단조변환).
    """
    rows = []
    for c in cols:
        sub = panel[["date", c, "target"]].rename(columns={c: "q50"})
        r = rank_ic(sub)
        rows.append({"feature": c, "ic": r["ic_mean"], "t": r["t_stat"],
                     "n_dates": r["n_dates"]})
    return pd.DataFrame(rows).set_index("feature")


# ------------------------------------------------------- (2) VSN 중요도와 그 안정성
def importance_by_year(w: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    year = pd.to_datetime(w["date"]).dt.year
    return w.groupby(year)[cols].mean().T


def bootstrap_rank_stability(
    w: pd.DataFrame, cols: list[str], n: int, rng: np.random.Generator
) -> pd.DataFrame:
    """날짜를 복원추출해 중요도 순위가 얼마나 흔들리는지 본다.

    같은 날 종목들은 시장 공통 요인으로 묶여 있으므로 **날짜 단위로** 뽑는다
    (랭크 IC 를 날짜별로 먼저 집계하는 것과 같은 이유).
    """
    by_date = w.groupby("date")[cols].mean()
    dates = by_date.index.to_numpy()
    ranks = []
    for _ in range(n):
        pick = rng.choice(len(dates), size=len(dates), replace=True)
        m = by_date.iloc[pick].mean()
        ranks.append(m.rank(ascending=False))
    r = pd.DataFrame(ranks)
    return pd.DataFrame({
        "rank_mean": r.mean(),
        "rank_std": r.std(ddof=1),
        "rank_p05": r.quantile(0.05),
        "rank_p95": r.quantile(0.95),
    })


def _spearman(a: pd.Series, b: pd.Series) -> float:
    return float(a.rank().corr(b.rank()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"],
                    help="VSN 중요도를 잴 구간 (기본 test = 학습에 안 쓴 구간)")
    ap.add_argument("--bootstrap", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=1024)
    args = ap.parse_args()

    setup_logging(run_name="feature_diagnostics")
    cfg = load_config()
    set_seed(int(cfg["project"]["seed"]))
    rng = np.random.default_rng(int(cfg["project"]["seed"]))

    ckpt = find_checkpoint(args.checkpoint)
    loaded = load_model(ckpt)
    bundle = load_features(cfg, loaded)
    cols = loaded.feature_cols

    # --- (1) 단변량 IC. 결정용은 train, 사후확인용으로 test 도 같이 낸다.
    parts = split_by_date(bundle.raw_panel, bundle.spec)
    ic_train = univariate_ic(parts["train"].dropna(subset=["target"]), cols)
    ic_test = univariate_ic(parts["test"].dropna(subset=["target"]), cols)

    # --- (2) VSN 중요도
    log.info("VSN 가중치 추출 (%s 구간)...", args.split)
    dyn, static = vsn_weights_split(loaded, bundle, cfg, args.split,
                                    batch_size=args.batch_size)
    imp = dyn[cols].mean().sort_values(ascending=False)
    by_year = importance_by_year(dyn, cols)
    boot = bootstrap_rank_stability(dyn, cols, args.bootstrap, rng)

    # VSN 은 원래 **윈도우마다 다른** 가중치를 낼 수 있다(문맥 조건화).
    # 실제로 변하는지 재지 않으면 "변수선택망이 상황에 따라 고른다"는 서술을
    # 근거 없이 하게 된다. 창별 순위가 전역 평균 순위와 얼마나 같은지로 본다.
    sample = dyn[cols].sample(min(2000, len(dyn)), random_state=int(cfg["project"]["seed"]))
    per_window = sample.rank(axis=1, ascending=False)
    global_rank = imp.rank(ascending=False)[cols]
    window_agreement = float(
        per_window.apply(lambda r: r.corr(global_rank, method="spearman"), axis=1).mean()
    )

    # 연도 간 순위 일치도 — 국면이 바뀌어도 같은 피처를 보는가
    years = list(by_year.columns)
    year_pairs = {
        f"{a}-{b}": round(_spearman(by_year[a], by_year[b]), 4)
        for i, a in enumerate(years) for b in years[i + 1:]
    }

    # --- (3) 대조: VSN 이 고른 것과 단변량으로 유의한 것이 같은가
    table = pd.DataFrame({
        "vsn_importance": imp,
        "vsn_rank": imp.rank(ascending=False),
        "rank_std": boot["rank_std"],
        "imp_std": dyn[cols].std(),
        "ic_train": ic_train["ic"],
        "t_train": ic_train["t"],
        "ic_test": ic_test["ic"],
        "t_test": ic_test["t"],
    }).sort_values("vsn_importance", ascending=False)
    table["abs_ic_train_rank"] = table["ic_train"].abs().rank(ascending=False)

    agreement = {
        "vsn_vs_abs_ic_train": round(
            _spearman(table["vsn_importance"], table["ic_train"].abs()), 4),
        "vsn_vs_abs_ic_test": round(
            _spearman(table["vsn_importance"], table["ic_test"].abs()), 4),
        "abs_ic_train_vs_test": round(
            _spearman(table["ic_train"].abs(), table["ic_test"].abs()), 4),
    }

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "checkpoint": ckpt.name,
        "split": args.split,
        "n_windows": int(len(dyn)),
        "bootstrap": args.bootstrap,
        "table": table.round(5).reset_index().rename(
            columns={"index": "feature"}).to_dict("records"),
        "static_importance": static[list(loaded.meta["vocab_sizes"])]
            .mean().round(5).to_dict(),
        "importance_by_year": by_year.round(5).to_dict(),
        "year_rank_agreement": year_pairs,
        "within_window_rank_agreement": round(window_agreement, 4),
        "vsn_vs_univariate": agreement,
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = REPORTS_DIR / f"feature_diagnostics_{stamp}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str),
                   encoding="utf-8")

    # --- 요약 출력
    pd.set_option("display.width", 140)
    print(f"\n체크포인트 {ckpt.name} | {args.split} 구간 {len(dyn):,} 윈도우\n")
    print(table.round(4).to_string())
    print("\nstatic 중요도:", {k: round(v, 4) for k, v in
                            report["static_importance"].items()})
    print("\n연도별 중요도 순위 일치도 (Spearman):", year_pairs)
    print(f"창별 순위 ↔ 전역 평균 순위 일치도: {window_agreement:.4f}"
          "  (1.0 에 가까우면 선택이 사실상 고정돼 있다는 뜻)")
    print("VSN 중요도 ↔ 단변량 |IC| 일치도:", agreement)
    print(f"\n리포트: {out.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
