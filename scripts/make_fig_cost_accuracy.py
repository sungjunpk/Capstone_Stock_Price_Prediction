#!/usr/bin/env python
"""Fig. 2 — 정확도 대 비용 산점도. 학술지 체재(흑백 인쇄 안전, 영문 라벨).

왜 레이더가 아닌가:
    이전 판은 9축 레이더였다. 두 가지가 끝내 해결되지 않았다.
      1) 아홉 축 중 여섯이 6.1절이 편향을 인정한 매매 평가에서 나온 값이었다.
      2) "Interpretability" 축은 VSN 상위 5 가중치의 **집중도**인데, 5.6절은
         그 중요도가 예측력의 대리 지표가 못 된다고 결론낸다. 그림이 본문과
         다른 말을 했다.
    캡션 공시는 "좋게 나왔을 수 있다"는 경고일 뿐 이 모순을 막지 못한다.
    그래서 6.1절이 스스로 밝힌 중심 주장 — 정확도와 비용 — 만 그린다.

축 (전부 <Table 4>·<Table 5> 와 같은 출처를 읽는다):
    x  종목당 추론지연 (ms)        작을수록 좋다
    y  무조건부 기준선 대비 핀볼 개선율 (%)   클수록 좋다
    점 넓이  활성값 메모리 (120일 입력, MB)   작을수록 좋다
    → 비용 지표 세 개 중 둘이 한 그림에 들어간다. 학습시간은 점 옆에 적는다.

점 예측 PatchTST 원본은 제외한다. 핀볼 개선율이 −47.79% 라 같은 축에 두면
나머지 셋이 축의 4% 안에 뭉쳐 구분되지 않는다. 캡션에서 수치로 밝힌다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
REP = ROOT / "outputs" / "reports"
OUT = ROOT / "docs" / "images" / "fig2_cost_accuracy.png"

KEYS = ["patchtst_q", "tft", "ours"]
# 비용은 분위 헤드 유무로 갈리지 않는다. 두 PatchTST 변형은 모수가 34.3K 대 35.0K 로
# 사실상 같아 <Table 5> 도 한 행("PatchTST (+ quantile head)")으로 보고한다.
COST_KEY = {"patchtst_q": "patchtst", "tft": "tft", "ours": "ours"}
LABEL = {"patchtst_q": "PatchTST\n+ quantile", "tft": "TFT", "ours": "Proposed"}
# ⚠️ 세 마커를 모두 원으로 둔다. matplotlib 의 s 는 **외접 사각형** 면적이라
#    모양이 다르면 실제 칠해지는 면적이 달라진다(원 π/4, 사각형 1.0, 삼각형 0.5).
#    삼각형·사각형을 섞었을 때 84MB 가 46MB 의 1.17배로, 168MB 가 4.66배로 보였다
#    — 캡션이 "marker area = activation memory" 라고 적는 한 모양을 섞으면 안 된다.
#    모델 구분은 점 옆 라벨이 하므로 표식 모양에 기댈 필요가 없다.
STYLE = {
    "patchtst_q": dict(marker="o", facecolor="0.82", edgecolor="0.30"),
    "tft":        dict(marker="o", facecolor="0.55", edgecolor="0.15"),
    "ours":       dict(marker="o", facecolor="black", edgecolor="black"),
}
# 라벨을 점 어느 쪽에 둘지 (겹침 방지). TFT 는 아래로 내려야 Proposed 라벨과 안 겹친다.
OFFSET = {"patchtst_q": (13, -13, "left"), "tft": (0, -34, "center"), "ours": (10, 7, "left")}


def load():
    j = json.loads((REP / "arch_compare_20260916_194048.json").read_text(encoding="utf-8"))
    acc = {mod["arch"]: mod["train"]["improvement_vs_baseline_pct"]
           for mod in j["models"] if "train" in mod}
    # ⚠️ 2026-10-17: tft/ours 는 **5시드 평균**으로 덮어쓴다. 위 JSON 의 두 값은
    #    tft=MPS 단정밀도, ours=CUDA 혼합정밀도로 학습 장비가 서로 달라 비교가
    #    성립하지 않았다(<Table 4> 캡션과 6.1절 셋째 한계). 같은 GPU 5시드 평균:
    #    ours 0.011889 → +2.348%, tft 0.011935 → +1.967% (기준선 0.012174728).
    acc.update({"tft": 1.967, "ours": 2.348})
    # 2026-10-19: patchtst_q 도 같은 혼합정밀도 5시드 평균으로 (0.012166 → +0.071%).
    #    위 JSON 의 +0.26% 는 MPS 단정밀도 단일 실행이었다.
    acc["patchtst_q"] = 0.071
    # ⚠️ 2026-10-18: 비용은 `scripts/cost_measure.py` 의 최신 리포트에서 읽는다.
    #    data.js 의 값은 장비·배치가 기록되지 않은 임시 측정이었다(<Table 5> 재측정).
    cm = json.loads(sorted(REP.glob("cost_measure_*.json"))[-1].read_text(encoding="utf-8"))
    cost = {COST_KEY[k]: {"latency": m["latency_per_stock_ms"],
                          "secPerEpoch": m["sec_per_epoch"],
                          "act": {"120": m["activation_mb"]}}
            for k, m in ((k, cm["models"][a]) for k, a in
                         (("patchtst_q", "patchtst_q"), ("tft", "tft"), ("ours", "phase1")))}
    return acc, cost


def main() -> int:
    acc, cost = load()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7})
    fig, ax = plt.subplots(figsize=(3.19, 2.45), dpi=400)   # 단 폭 81mm

    for k in KEYS:
        c = cost[COST_KEY[k]]
        x, y, mem = c["latency"], acc[k], c["act"]["120"]
        ax.scatter(x, y, s=mem * 2.2, zorder=3, linewidths=0.8, **STYLE[k])
        ax.annotate(f"{LABEL[k]}\n{c['secPerEpoch']:.0f} s/epoch | {mem:.0f} MB",
                    (x, y), textcoords="offset points", xytext=OFFSET[k][:2],
                    fontsize=6.0, linespacing=1.25, ha=OFFSET[k][2])

    # 더 좋은 방향을 한 번만 적는다
    ax.annotate("cheaper and more accurate", xy=(0.03, 0.94), xycoords="axes fraction",
                fontsize=6.0, style="italic", color="0.35")
    ax.annotate("marker area = activation memory", xy=(0.97, 0.04),
                xycoords="axes fraction", ha="right", fontsize=5.8, color="0.45")
    ax.set_xlabel("Inference latency per stock (ms)", fontsize=7)
    ax.set_ylabel("Pinball improvement over\nunconditional baseline (%)", fontsize=7)
    ax.set_xlim(0, 0.80)
    ax.set_ylim(-0.6, 3.4)
    ax.grid(color="0.88", lw=0.5, zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("0.4")
        ax.spines[sp].set_linewidth(0.6)
    ax.tick_params(labelsize=6.5, length=2.5, color="0.4")
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT, facecolor="white", bbox_inches="tight", pad_inches=0.04)
    print(f"wrote {OUT.relative_to(ROOT)}")
    for k in KEYS:
        c = cost[COST_KEY[k]]
        print(f"   {LABEL[k].replace(chr(10),' '):<20} "
              f"latency {c['latency']:.4f} ms | 개선 {acc[k]:+.2f}% | "
              f"활성값 {c['act']['120']:.1f} MB | {c['secPerEpoch']:.1f} s/epoch")
    print(f"   (제외) PatchTST 원본 점예측  개선 {acc['patchtst']:+.2f}% — 축 범위 밖")
    return 0


if __name__ == "__main__":
    sys.exit(main())
