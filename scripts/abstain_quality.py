#!/usr/bin/env python
"""기권이 쓸모 있는가 — 재학습 없이 기존 체크포인트의 예측으로만 재는 세 가지.

논문 5.4절은 기권이 **어디서** 일어나는지(고변동 구간)만 보였다. 기권한 판단이
실제로 나쁜 판단이었는지는 보이지 않았다. 세 질문에 답한다.

  A1  기권한 후보의 실현 오차가 남긴 후보보다 큰가 (= 보류가 옳았는가)
  A4  그 비율이 **단순 기준보다 나은가** — 핀볼 손실은 고변동 구간에서 원래 크므로,
      고변동을 피하는 규칙이면 무엇이든 A1 이 성립한다. 동일한 규칙(최근 90일 분포의
      같은 분위)에 점수만 예측 폭 → 직전 20일 실현변동성으로 바꿔 끼워 대조한다.
  A2  폭의 정보계수가 '직전 20일 실현변동성' 이라는 단순 기준보다 높은가
      — 변동성은 군집성으로 예측 가능하므로 단순 기준이 이미 잘 맞힐 수 있다
  A3  기권 임계값 분위를 바꾸면 모형 간 비교와 분위별 집중이 유지되는가
      (`percentile: 95` 는 검정구간 누적수익으로 고른 값이다 — config.yaml:196)
  A5  **기권율을 맞추면** 그 우위가 남는가 — A4 는 폭 5.99% 대 통제군 5.66% 로 기권율이
      달라 보류 집단이 꼬리로 더 들어간 효과가 섞여 있다. 날짜마다 같은 개수를 보류시켜
      (점수의 날짜별 순위 상위 r%) 두 점수를 같은 기권율에서 비교하고, 비율 차이에
      날짜 블록 부트스트랩 신뢰구간을 붙인다.

기권 판정은 `src/trading/signal.py` 를 그대로 부른다 (CLAUDE.md 규칙 7).

사용: python scripts/abstain_quality.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scripts.backtest import find_checkpoint  # noqa: E402
from src.evaluation.metrics import rank_ic  # noqa: E402
from src.models.inference import load_features, load_model, predict_split  # noqa: E402
from src.trading.signal import abstain_scores, recent_abstain_threshold  # noqa: E402
from src.utils.config import PROJECT_ROOT, load_config  # noqa: E402
from src.utils.logging import get_logger, setup_logging  # noqa: E402

log = get_logger("abstain_quality")
REPORTS = PROJECT_ROOT / "outputs" / "reports"
QUANTILES = [0.1, 0.5, 0.9]
QCOLS = ["q10", "q50", "q90"]

# tft·ours 는 <Table 4> 의 시드 42 실행을 쓴다 — 같은 GPU·같은 설정이라 장비가 섞이지
# 않는다. patchtst 계열은 단일 MPS 실행뿐이므로 그대로 두고 표 주석에 밝힌다.
MODELS = {
    "patchtst":   "outputs/checkpoints/phase1_7b693204_patchtst.pt",
    "patchtst_q": "outputs/checkpoints/phase1_5263fac6_patchtst_q.pt",
    "tft":        "outputs/checkpoints/phase1_f5e8f4e4_tft_seed42.pt",
    "ours":       "outputs/checkpoints/phase1_b113237e_seed42.pt",
}
PERCENTILES = [75, 90, 95]


def pinball(p: pd.DataFrame) -> pd.Series:
    """표본별 핀볼 손실. 세 분위의 **평균** — 학습 손실과 <Table 4> 와 같은 척도다.

    ⚠️ `scripts/dm_test.py` 는 같은 식을 합으로 둔다. DM 통계량은 척도에 불변이라
    거기서는 문제가 없지만, 여기 숫자는 논문 본문에 그대로 들어가므로 3으로 나눈다.
    """
    loss = 0.0
    for tau, col in zip(QUANTILES, QCOLS):
        e = p["target"] - p[col]
        loss = loss + np.maximum(tau * e, (tau - 1) * e)
    return loss / len(QUANTILES)


def abstain_flags(p: pd.DataFrame, cfg_abstain: dict, *,
                  score_col: str | None = None) -> np.ndarray:
    """판단일마다 최근 90일 분포로 임계값을 잡아 기권 여부를 판정한다.

    `score_col` 을 주면 예측 폭 대신 그 열을 점수로 쓴다 — 규칙은 그대로 두고
    점수만 바꿔 끼우는 통제군(A4)용이다. 임계값 계산은 signal.py 를 그대로 쓴다.
    """
    scores = (p[score_col].to_numpy(dtype=float) if score_col
              else abstain_scores(p, cfg_abstain))
    dates = p["date"].to_numpy()
    flags = np.zeros(len(p), dtype=bool)
    for d in pd.unique(dates):
        thr = recent_abstain_threshold(scores, dates, d, cfg_abstain)
        m = dates == d
        flags[m] = scores[m] > thr
    return flags


MATCHED_RATES = [2, 5, 6, 10, 20, 30]


def _per_date(loss: np.ndarray, flag: np.ndarray, dates: np.ndarray) -> pd.DataFrame:
    """날짜별 (기권 손실합·건수, 유지 손실합·건수) — 부트스트랩을 날짜 단위로 돌리려고."""
    d = pd.DataFrame({"date": dates, "loss": loss, "ab": flag})
    ab = d[d["ab"]].groupby("date")["loss"].agg(["sum", "size"])
    ke = d[~d["ab"]].groupby("date")["loss"].agg(["sum", "size"])
    agg = pd.DataFrame(index=pd.Index(sorted(d["date"].unique()), name="date"))
    for src, tag in ((ab, "ab"), (ke, "ke")):
        agg[f"s_{tag}"] = src["sum"].reindex(agg.index).fillna(0.0)
        agg[f"n_{tag}"] = src["size"].reindex(agg.index).fillna(0)
    return agg[["s_ab", "n_ab", "s_ke", "n_ke"]]


def _ratio(a: np.ndarray) -> float:
    """a: (n,4) 열이 s_ab, n_ab, s_ke, n_ke."""
    s_ab, n_ab, s_ke, n_ke = a.sum(axis=0)
    if n_ab == 0 or n_ke == 0:
        return float("nan")
    return float((s_ab / n_ab) / (s_ke / n_ke))


def matched_rate_compare(p: pd.DataFrame, *, n_boot: int = 1000, seed: int = 42) -> dict:
    """날짜마다 같은 개수를 보류시켜 폭과 실현변동성을 같은 기권율에서 비교한다."""
    d = p[["date", "loss", "rvol_20"]].copy()
    d["width"] = p["q90"] - p["q10"]
    g = d.groupby("date")
    rank = {k: g[k].rank(pct=True, method="first") for k in ("width", "rvol_20")}
    loss, dates = d["loss"].to_numpy(), d["date"].to_numpy()
    rng = np.random.default_rng(seed)
    out = {}
    for r in MATCHED_RATES:
        cut = 1 - r / 100
        arr, rates, ratios = {}, {}, {}
        for k in ("width", "rvol_20"):
            flag = (rank[k] > cut).to_numpy()
            arr[k] = _per_date(loss, flag, dates).to_numpy(dtype=float)
            rates[k] = round(float(flag.mean()) * 100, 2)
            ratios[k] = round(_ratio(arr[k]), 3)
        n = len(arr["width"])
        diffs = np.empty(n_boot)
        for b in range(n_boot):
            idx = rng.integers(0, n, n)
            diffs[b] = _ratio(arr["width"][idx]) - _ratio(arr["rvol_20"][idx])
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        out[str(r)] = {
            "rate_width": rates["width"], "rate_rvol_20": rates["rvol_20"],
            "ratio_width": ratios["width"], "ratio_rvol_20": ratios["rvol_20"],
            "diff": round(ratios["width"] - ratios["rvol_20"], 3),
            "diff_ci95": [round(float(lo), 3), round(float(hi), 3)],
            "diff_positive_rate": round(float((diffs > 0).mean()), 3),
            "n_dates": int(n),
        }
    return out


def threshold_drift(p: pd.DataFrame, cfg_abstain: dict, *, n_days: int = 244) -> dict:
    """검정구간 마지막 n_days 의 첫날·마지막날 임계값과 그 구간 평균 폭 (A6).

    임계값은 직전 90일 폭 분포의 분위이므로, 그 값이 오르면 폭이 시장 변동성 상승에
    반응했다는 뜻이다. 모형 간 상승률을 비교하려고 잰다.
    """
    scores = abstain_scores(p, cfg_abstain)
    dates = p["date"].to_numpy()
    last = sorted(pd.unique(dates))[-n_days:]
    first, final = (recent_abstain_threshold(scores, dates, d, cfg_abstain)
                    for d in (last[0], last[-1]))
    width = (p["q90"] - p["q10"]).to_numpy()[np.isin(dates, last)]
    return {
        "n_days": len(last),
        "date_first": str(pd.Timestamp(last[0]).date()),
        "date_last": str(pd.Timestamp(last[-1]).date()),
        "threshold_first": round(float(first), 4),
        "threshold_last": round(float(final), 4),
        "change_pct": round((final / first - 1) * 100, 1),
        "mean_width": round(float(width.mean()), 4),
    }


SEEDS = [42, 43, 44, 45, 46]
# 시드가 여러 개 있는 두 구조는 평균·표준편차까지 낸다 — 〈Table 6〉·〈Table 7〉이
# 한 실행의 값이라는 한계를 지우려고. 체크포인트는 〈Table 4〉와 같은 것들이다.
SEED_MODELS = {
    "tft":  "outputs/checkpoints/phase1_f5e8f4e4_tft_seed{}.pt",
    "ours": "outputs/checkpoints/phase1_b113237e_seed{}.pt",
}


def fold(vals: list):
    """같은 구조의 기록 여러 개를 잎마다 평균·표준편차·범위로 접는다."""
    v0 = vals[0]
    if isinstance(v0, dict):
        return {k: fold([v[k] for v in vals]) for k in v0}
    if isinstance(v0, list):
        return [fold([v[i] for v in vals]) for i in range(len(v0))]
    if isinstance(v0, str):
        return v0 if len(set(vals)) == 1 else list(vals)
    return {"mean": round(statistics.fmean(vals), 6),
            "sd": round(statistics.stdev(vals), 6) if len(vals) > 1 else 0.0,
            "range": [min(vals), max(vals)]}


def measure(ck: str, cfg: dict, base: dict, *, point: bool) -> dict:
    """한 체크포인트의 기권·구간 지표 전부 (A1~A6)."""
    loaded = load_model(find_checkpoint(ck, cfg))
    bundle = load_features(cfg, loaded)
    p, _ = predict_split(loaded, bundle, cfg, "test")
    return measure_preds(p, ck, base, point=point)


def measure_preds(p: pd.DataFrame, ck: str, base: dict, *, point: bool) -> dict:
    """예측 표(code·date·q10·q50·q90·target·rvol_20)만으로 A1~A6 — 다른 모형도 같은 코드로 잰다."""
    p = p.dropna(subset=["target"]).copy()
    p["loss"] = pinball(p)
    p["abs_err"] = (p["target"] - p["q50"]).abs()
    p["hit"] = np.sign(p["q50"]) == np.sign(p["target"])
    rec: dict = {"checkpoint": ck, "n": int(len(p))}

    if not point:
        # ── A1  기권 대 유지 (운용 설정 percentile)
        f = abstain_flags(p, base)
        rec["a1"] = {}
        for lab, sub in (("abstained", p[f]), ("retained", p[~f])):
            rec["a1"][lab] = {
                "n": int(len(sub)),
                "pinball": round(float(sub["loss"].mean()), 6),
                "abs_err": round(float(sub["abs_err"].mean()), 6),
                "hit_rate": round(float(sub["hit"].mean()), 4),
            }
        # 날짜별 쌍대 차이로 검정 — 같은 날 종목들은 시장 요인을 공유한다
        daily = p.assign(ab=f).groupby(["date", "ab"])["loss"].mean().unstack()
        daily = daily.dropna()
        if len(daily) > 2:
            d = (daily[True] - daily[False]).to_numpy()
            t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
            rec["a1"]["paired"] = {"n_dates": int(len(d)),
                                   "mean_diff": round(float(d.mean()), 6),
                                   "t_stat": round(float(t), 3)}

        # ── A4  점수만 교체한 통제군 (규칙·분위는 동일)
        fv = abstain_flags(p, base, score_col="rvol_20")
        rec["a4"] = {}
        for lab, (fl, sub) in (("width", (f, None)), ("rvol_20", (fv, None))):
            ab, ke = p[fl], p[~fl]
            rec["a4"][lab] = {
                "rate": round(float(fl.mean()) * 100, 2),
                "pinball_abstained": round(float(ab["loss"].mean()), 6),
                "pinball_retained": round(float(ke["loss"].mean()), 6),
                "ratio": round(float(ab["loss"].mean() / ke["loss"].mean()), 3),
            }

        # ── A5  기권율을 날짜별로 맞춘 비교
        rec["a5"] = matched_rate_compare(p)

        # ── A6  마지막 12개월 임계값 변화 — 폭이 변동성에 반응하는가
        rec["a6"] = threshold_drift(p, base)

        # ── A3  임계값 분위 민감도
        rec["a3"] = {}
        for pc in PERCENTILES:
            fp = abstain_flags(p, {**base, "percentile": pc})
            q = pd.qcut(p["rvol_20"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
            by = pd.Series(fp, index=p.index).groupby(q, observed=True).mean()
            rec["a3"][str(pc)] = {
                "overall": round(float(fp.mean()) * 100, 2),
                **{k: round(float(v) * 100, 2) for k, v in by.items()},
            }

    # ── A2  폭 IC 대 단순 변동성 기준
    tgt = pd.DataFrame({"date": p["date"], "target": p["target"].abs()})
    rec["a2"] = {
        "width_ic": rank_ic(tgt.assign(q50=p["q90"] - p["q10"])),
        "rvol20_ic": rank_ic(tgt.assign(q50=p["rvol_20"])),
        "directional_ic": rank_ic(p[["date", "q50", "target"]]),
    }
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=SEEDS,
                    help="tft·ours 를 이 시드들로 반복해 평균·표준편차까지 낸다. "
                         "빈 값을 주면 시드 42 한 실행만 (예전 동작)")
    args = ap.parse_args()

    setup_logging(run_name="abstain_quality")
    cfg = load_config()
    base = dict(cfg["trading"]["abstain"])
    out = {"generated_at": datetime.now().isoformat(timespec="seconds"),
           "recent_days": base.get("recent_days"), "basis": base.get("basis"),
           "seeds": args.seeds, "models": {}}

    for name, ck in MODELS.items():
        point = name == "patchtst"      # 점 예측 — 폭이 0 이라 기권이 정의되지 않는다
        rec = measure(ck, cfg, base, point=point)
        if name in SEED_MODELS and len(args.seeds) > 1:
            runs = []
            for sd in args.seeds:
                path = SEED_MODELS[name].format(sd)
                r = rec if path == ck else measure(path, cfg, base, point=point)
                runs.append({"seed": sd, **r})
                log.info("%s seed %d 완료", name, sd)
            rec["seeds"] = {"n": len(runs), "runs": runs,
                            "agg": fold([{k: v for k, v in r.items() if k != "seed"}
                                         for r in runs])}
        out["models"][name] = rec
        log.info("%s 완료", name)

    path = REPORTS / f"abstain_quality_{datetime.now():%Y%m%d_%H%M%S}.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    for name, r in out["models"].items():
        print(f"\n=== {name}  (n={r['n']:,})")
        if "a1" in r:
            a, b = r["a1"]["abstained"], r["a1"]["retained"]
            print(f"  A1 기권 {a['n']:>6,}건  핀볼 {a['pinball']:.6f}  "
                  f"|오차| {a['abs_err']:.6f}  적중 {a['hit_rate']:.3f}")
            print(f"     유지 {b['n']:>6,}건  핀볼 {b['pinball']:.6f}  "
                  f"|오차| {b['abs_err']:.6f}  적중 {b['hit_rate']:.3f}")
            if "paired" in r["a1"]:
                q = r["a1"]["paired"]
                print(f"     날짜쌍 차 {q['mean_diff']:+.6f} (t = {q['t_stat']:+.2f}, "
                      f"{q['n_dates']}일)")
        a2 = r["a2"]
        print(f"  A2 폭 IC {a2['width_ic']['ic_mean']:+.4f} "
              f"(t = {a2['width_ic']['t_stat']:.2f})   "
              f"rvol_20 IC {a2['rvol20_ic']['ic_mean']:+.4f} "
              f"(t = {a2['rvol20_ic']['t_stat']:.2f})   "
              f"방향 IC {a2['directional_ic']['ic_mean']:+.4f}")
        if "a4" in r:
            for lab, v in r["a4"].items():
                print(f"  A4 {lab:<8} 기권 {v['rate']:>5.2f}%  "
                      f"기권손실 {v['pinball_abstained']:.6f}  "
                      f"유지손실 {v['pinball_retained']:.6f}  비 {v['ratio']:.3f}")
        if "a5" in r:
            print("  A5 기권율 맞춤(날짜별 동수) — 폭 비 / 변동성 비 / 차이 [95% CI]")
            for rt, v in r["a5"].items():
                print(f"       {rt:>2}%  {v['ratio_width']:.3f} / {v['ratio_rvol_20']:.3f} / "
                      f"{v['diff']:+.3f} [{v['diff_ci95'][0]:+.3f}, {v['diff_ci95'][1]:+.3f}]"
                      f"   실기권율 {v['rate_width']:.2f}% / {v['rate_rvol_20']:.2f}%")
        if "a3" in r:
            for pc, v in r["a3"].items():
                print(f"  A3 p{pc}  전체 {v['overall']:>5.2f}%  "
                      f"Q1 {v['Q1']:>5.2f}%  Q4 {v['Q4']:>5.2f}%")
    print(f"\n리포트: {path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
