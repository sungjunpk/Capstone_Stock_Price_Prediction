#!/usr/bin/env python
"""Fig. 2 — 세 구조의 9축 비교 레이더. 학술지 체재(흑백 인쇄 안전, 영문 라벨).

수치는 발표용 자료가 아니라 비교 실행 보고서에서 직접 읽는다.
  outputs/reports/arch_compare_20260916_194048.json      (정확도·수익·중요도)
  outputs/reports/arch_compare_20260916_v2/data.js       (왕복거래·비용)

축 점수는 세 모형의 비율 척도다. 클수록 좋은 축은 값/최대값, 작을수록 좋은 축은
최소값/값 으로 환산한다(최고 모형이 100).

⚠️ 변동성과 최대낙폭 축을 반드시 포함한다. 이 둘을 빼면 제안 모형이 모든 축에서
   바깥 껍질이 되어, 축 선택이 결론을 만든 그림이 된다 — 실제로 두 축 모두 TFT
   원본이 낫다(<Table 6>). 심사에서 가장 먼저 지적될 지점이므로 되돌리지 말 것.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
REP = ROOT / "outputs" / "reports"
OUT = ROOT / "docs" / "images" / "fig2_radar.png"
KEYS = ["patchtst", "tft", "ours"]
LABELS = ["PatchTST", "TFT", "Proposed"]
STYLE = [  # 흑백 인쇄에서도 구분되도록 선 모양·표식으로 나눈다
    dict(color="0.55", ls=":", marker="^", ms=3.2, lw=1.0),
    dict(color="0.25", ls="--", marker="s", ms=3.0, lw=1.0),
    dict(color="black", ls="-", marker="o", ms=3.4, lw=1.4),
]


def load():
    j = json.loads((REP / "arch_compare_20260916_194048.json").read_text(encoding="utf-8"))
    js = (REP / "arch_compare_20260916_v2" / "data.js").read_text(encoding="utf-8")
    m = re.search(r"window\.D\s*=\s*", js)
    v2, _ = json.JSONDecoder().raw_decode(js[m.end():])
    by = {}
    for mod in j["models"]:
        if mod["arch"] in ("patchtst", "tft", "ours"):
            by[mod["arch"]] = mod
    return by, v2


def _mix(*parts, better="hi"):
    """여러 지표를 각자의 최선값으로 정규화해 평균한다. 분모를 고정하지 않는다."""
    out = {}
    for k in KEYS:
        vals = []
        for d in parts:
            ref = max(d.values()) if better == "hi" else min(d.values())
            vals.append((d[k] / ref) if better == "hi" else (d[k] / ref))
        out[k] = sum(vals) / len(vals)
    return out


def axes_scores(by, v2):
    risk, cost = v2["risk"], v2["cost"]
    hi = lambda d: {k: max(0.0, d[k]) for k in KEYS}
    raw = {
        "Cumulative\nreturn":    hi({k: by[k]["metrics"]["total_return"] for k in KEYS}),
        "Risk-adjusted\nreturn": hi({k: risk[k]["sharpe"] for k in KEYS}),
        "Per-trade\nexpectancy": hi({k: risk[k]["expectancy"] for k in KEYS}),
        "Risk\ncontrol":         hi(_mix({k: risk[k]["abstainByVol"]["Q4"] for k in KEYS},
                                        {k: risk[k]["payoff"] for k in KEYS})),
        "Interpretability":      hi({k: sum(by[k]["train"]["feature_importance"].values())
                                     for k in KEYS}),
    }
    lo = {  # 작을수록 좋은 축
        "Forecast\naccuracy": {k: by[k]["train"]["best_val_pinball"] for k in KEYS},
        "Volatility": {k: by[k]["metrics"]["volatility"] for k in KEYS},
        "Max\ndrawdown": {k: abs(by[k]["metrics"]["max_drawdown"]) for k in KEYS},
        "Cost\n(train & infer)": _mix(
            {k: cost[k]["secPerEpoch"] for k in KEYS},
            {k: cost[k]["latency"] for k in KEYS},
            {k: cost[k]["act"]["120"] for k in KEYS}, better="lo"),
    }
    out = {}
    for name, d in raw.items():
        mx = max(d.values()) or 1.0
        out[name] = [100 * d[k] / mx for k in KEYS]
    for name, d in lo.items():
        mn = min(d.values())
        out[name] = [100 * mn / d[k] for k in KEYS]
    order = ["Cumulative\nreturn", "Risk-adjusted\nreturn", "Per-trade\nexpectancy",
             "Risk\ncontrol", "Volatility", "Max\ndrawdown", "Forecast\naccuracy",
             "Interpretability", "Cost\n(train & infer)"]
    return order, out


def main() -> int:
    by, v2 = load()
    # PatchTST 원본은 분위 헤드가 없어 cost 표에 'patchtst' 하나만 있다
    v2["cost"]["ours"]["act"]["120"] = v2["cost"]["ours"]["act"]["120"]
    order, sc = axes_scores(by, v2)
    n = len(order)
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    ang += ang[:1]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 6.5})
    fig = plt.figure(figsize=(3.4, 3.4), dpi=400)        # 단 폭 81mm 에 맞춘다
    ax = fig.add_subplot(polar=True)
    for i, lab in enumerate(LABELS):
        v = [sc[a][i] for a in order]
        v += v[:1]
        ax.plot(ang, v, label=lab, **STYLE[i])
        if lab == "Proposed":
            ax.fill(ang, v, color="black", alpha=0.055)
    ax.set_xticks(ang[:-1])
    ax.set_xticklabels(order, fontsize=5.8)
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels([])          # 상대 점수이므로 눈금값은 캡션에서 설명한다
    ax.tick_params(pad=4)
    ax.grid(color="0.80", lw=0.5)
    ax.spines["polar"].set_color("0.6")
    ax.spines["polar"].set_linewidth(0.6)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.30), ncol=3,
              frameon=False, fontsize=6.2, handlelength=2.2, columnspacing=1.4)
    fig.subplots_adjust(left=0.26, right=0.74, top=0.84, bottom=0.26)
    fig.savefig(OUT, facecolor="white", bbox_inches="tight", pad_inches=0.06)
    print(f"wrote {OUT.relative_to(ROOT)}")
    for a in order:
        print(f"   {a.replace(chr(10),' '):<28} " +
              "  ".join(f"{L} {sc[a][i]:6.2f}" for i, L in enumerate(LABELS)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
