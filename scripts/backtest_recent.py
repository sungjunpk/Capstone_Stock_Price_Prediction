#!/usr/bin/env python
"""최근 N개월 백테스트 — 지수 대비 성과를 한 화면에서 본다.

`scripts/backtest.py` 는 split(val/test) 전체를 돈다. 이 스크립트는 그중
**최근 구간만 잘라** 발표용 비교를 만든다. 백테스트 엔진·매매 규칙·비용은
전부 같은 코드다(`evaluation.backtest.run_backtest`) — 구간만 다르다.

벤치마크 3종:
    코스피200(201) / 코스피(001) — 지수 일봉
    유니버스 동일가중 매수후보유  — 종목 선택 능력만 분리해서 본다

사용:
    python scripts/backtest_recent.py                  # 최근 6개월
    python scripts/backtest_recent.py --months 3
    python scripts/backtest_recent.py --checkpoint outputs/checkpoints/phase1_0db568ae.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from html import escape
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from src.data.storage import RAW_DIR  # noqa: E402
from src.evaluation.backtest import buy_and_hold, run_backtest  # noqa: E402
from src.evaluation.metrics import equity_curve, summarize  # noqa: E402
from src.models.inference import load_features, load_model, predict_split  # noqa: E402
from src.utils.config import PROJECT_ROOT, load_config  # noqa: E402
from src.utils.logging import get_logger, setup_logging  # noqa: E402

log = get_logger("backtest_recent")
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
CKPT_DIR = PROJECT_ROOT / "outputs" / "checkpoints"

INDEXES = {"201": "코스피200", "001": "코스피"}


def index_returns(code: str, dates: pd.Index) -> pd.Series | None:
    """지수 일봉 → 전략과 같은 날짜축의 일간수익률."""
    path = RAW_DIR / "index_daily" / f"{code}.parquet"
    if not path.exists():
        log.warning("지수 %s 없음 — 건너뛴다", code)
        return None
    df = pd.read_parquet(path)
    s = (
        df.assign(date=pd.to_datetime(df["date"]).dt.date)
        .set_index("date")["close"]
        .sort_index()
    )
    s = s.reindex(dates).ffill()
    return s.pct_change(fill_method=None).fillna(0.0)


def monthly_table(series: dict[str, pd.Series]) -> list[dict]:
    """월별 수익률 — 전략과 벤치마크를 같은 행에 놓는다."""
    keys = sorted({str(d)[:7] for s in series.values() for d in s.index})
    out = []
    for ym in keys:
        row = {"month": ym}
        for name, s in series.items():
            sel = s[[str(d)[:7] == ym for d in s.index]]
            row[name] = round(float((1 + sel).prod() - 1), 5) if len(sel) else None
        out.append(row)
    return out


def attribution(trades: pd.DataFrame, prices: pd.DataFrame,
                dates: list) -> list[dict]:
    """종목별 수익 기여도.

    백테스트 루프는 리밸런싱 사이에 비중을 흘리지 않는다(`pos.weight` 고정) —
    따라서 거래기록의 `to` 를 다음 거래까지 유지하면 일별 비중이 **정확히** 복원된다.
    기여도 = Σ_t w(code,t-1) x r(code,t) 이고, 합은 비용 차감 전 전략수익과 같다.
    """
    if trades.empty:
        return []
    px = prices.pivot_table(index="date", columns="code", values="close").sort_index()
    rets = px.pct_change(fill_method=None)
    tr = trades.copy()
    tr["date"] = pd.to_datetime(tr["date"]).dt.date

    contrib: dict[str, float] = {}
    n_round: dict[str, int] = {}
    for code, g in tr.groupby("code"):
        if code not in rets.columns:
            continue
        g = g.sort_values("date")
        w = pd.Series(0.0, index=dates)
        for r in g.itertuples():
            w.loc[[d for d in dates if d >= r.date]] = float(r.to)
        r_code = rets[code].reindex(dates).fillna(0.0)
        contrib[code] = float((w.shift(1).fillna(0.0) * r_code).sum())
        n_round[code] = int((g["to"] <= 1e-6).sum())

    rows = [{"code": c, "contribution": round(v, 5), "exits": n_round.get(c, 0)}
            for c, v in contrib.items()]
    rows.sort(key=lambda r: r["contribution"], reverse=True)
    return rows


# ------------------------------------------------------------------ 리포트 HTML
# 외부 라이브러리 없이 인라인 SVG 로 그린다 — src/webapp/render.py 와 같은 이유다.
# 색 토큰도 그 파일과 맞춘다. 시리즈 3색만 따로 검증해서 골랐다.

SERIES = [
    ("strategy", "전략", "s1"),
    ("코스피200", "코스피200", "s2"),
    ("buy_and_hold", "유니버스 동일가중", "s3"),
]

_FONTS = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
    'family=IBM+Plex+Sans+KR:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">'
)

_CSS = """
:root {
  --bg:#f6f7f9; --panel:#ffffff; --ink:#16181d; --muted:#6b7280; --line:#e3e6eb;
  --accent:#2563eb; --pos:#067647; --neg:#b42318; --warn:#b54708; --chip:#eef2ff;
  --s1:#2563eb; --s2:#eb6834; --s3:#1baf7a; --grid:#eceef1;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg:#0e1116; --panel:#161a21; --ink:#e6e8ec; --muted:#9aa3b2; --line:#262c36;
    --accent:#6ea8fe; --pos:#46c08d; --neg:#f2777a; --warn:#e0a458; --chip:#1c2331;
    --s1:#3987e5; --s2:#d95926; --s3:#199e70; --grid:#1e242e;
  }
}
:root[data-theme="dark"] {
  --bg:#0e1116; --panel:#161a21; --ink:#e6e8ec; --muted:#9aa3b2; --line:#262c36;
  --accent:#6ea8fe; --pos:#46c08d; --neg:#f2777a; --warn:#e0a458; --chip:#1c2331;
  --s1:#3987e5; --s2:#d95926; --s3:#199e70; --grid:#1e242e;
}
* { box-sizing:border-box; }
body {
  margin:0; background:var(--bg); color:var(--ink);
  font:15px/1.6 "IBM Plex Sans KR", -apple-system, BlinkMacSystemFont,
       "Apple SD Gothic Neo", "Noto Sans KR", sans-serif;
}
.wrap { max-width:1080px; margin:0 auto; padding:32px 20px 72px; }
.mono { font-family:"IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, monospace;
        font-variant-numeric:tabular-nums; }
header { margin-bottom:18px; }
h1 { font-size:24px; margin:0 0 6px; letter-spacing:-0.02em; text-wrap:balance; font-weight:600; }
.sub { color:var(--muted); font-size:13.5px; }
.chips { display:flex; flex-wrap:wrap; gap:6px; margin:16px 0 24px; }
.chip { background:var(--chip); border:1px solid var(--line); border-radius:999px;
        padding:3px 11px; font-size:12px; color:var(--muted); }
.chip b { color:var(--ink); font-weight:600; }
.tiles { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
         gap:12px; margin-bottom:24px; }
.tile { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:14px 16px; }
.tile .k { color:var(--muted); font-size:12px; }
.tile .v { font-size:25px; font-weight:600; letter-spacing:-0.025em; margin-top:3px; }
.tile .n { color:var(--muted); font-size:11.5px; margin-top:2px; }
section { background:var(--panel); border:1px solid var(--line); border-radius:12px;
          padding:20px 22px; margin-bottom:16px; }
h2 { font-size:16px; margin:0 0 4px; font-weight:600; letter-spacing:-0.01em; }
.note { color:var(--muted); font-size:12.5px; margin:0 0 16px; }
.legend { display:flex; flex-wrap:wrap; gap:16px; margin:0 0 12px; font-size:12.5px; }
.legend span { display:flex; align-items:center; gap:6px; color:var(--muted); }
.legend i { width:11px; height:11px; border-radius:3px; display:block; }
.two { display:grid; grid-template-columns:1fr 1fr; gap:16px; }
@media (max-width:820px) { .two { grid-template-columns:1fr; } }
.chart { position:relative; }
svg { display:block; width:100%; height:auto; overflow:visible; }
.tip { position:absolute; pointer-events:none; opacity:0; transition:opacity .12s;
       background:var(--panel); border:1px solid var(--line); border-radius:8px;
       padding:8px 10px; font-size:12px; box-shadow:0 4px 14px rgb(0 0 0 / .12); z-index:5;
       white-space:nowrap; }
.tip b { display:block; margin-bottom:4px; font-size:11.5px; color:var(--muted); font-weight:500; }
.tip div { display:flex; align-items:center; gap:6px; }
.tip i { width:8px; height:8px; border-radius:2px; }
table { width:100%; border-collapse:collapse; font-size:13.5px; }
th, td { text-align:right; padding:8px 9px; border-bottom:1px solid var(--line); }
th:first-child, td:first-child { text-align:left; }
th { color:var(--muted); font-weight:500; font-size:12px; white-space:nowrap; }
tbody tr:last-child td { border-bottom:none; }
.pos { color:var(--pos); } .neg { color:var(--neg); } .muted { color:var(--muted); }
.tw { overflow-x:auto; }
.verdict { margin-top:14px; padding:12px 14px; border-radius:8px; font-size:13px;
           background:var(--chip); border:1px solid var(--line); }
.verdict b { color:var(--warn); }
details { margin-top:14px; } summary { cursor:pointer; color:var(--muted); font-size:12.5px; }
footer { color:var(--muted); font-size:12px; text-align:center; margin-top:28px; }
@media (prefers-reduced-motion:reduce) { .tip { transition:none; } }
"""

_JS = """
document.querySelectorAll('[data-chart]').forEach(function (root) {
  var pts = JSON.parse(root.dataset.points), labels = JSON.parse(root.dataset.labels);
  var names = JSON.parse(root.dataset.names), tip = root.querySelector('.tip');
  var line = root.querySelector('.cross'), dots = root.querySelectorAll('.hd');
  var svg = root.querySelector('svg'), x0 = +root.dataset.x0, x1 = +root.dataset.x1;
  function hide() { tip.style.opacity = 0; line.style.opacity = 0;
                    dots.forEach(function (d) { d.style.opacity = 0; }); }
  root.addEventListener('mousemove', function (e) {
    var r = svg.getBoundingClientRect();
    var vb = svg.viewBox.baseVal;
    var vx = (e.clientX - r.left) / r.width * vb.width;
    var i = Math.round((vx - x0) / (x1 - x0) * (labels.length - 1));
    if (i < 0) i = 0; if (i >= labels.length) i = labels.length - 1;
    var px = x0 + (x1 - x0) * i / (labels.length - 1);
    line.setAttribute('x1', px); line.setAttribute('x2', px); line.style.opacity = 1;
    var html = '<b>' + labels[i] + '</b>';
    pts.forEach(function (s, k) {
      dots[k].setAttribute('cx', px); dots[k].setAttribute('cy', s[i]); dots[k].style.opacity = 1;
      html += '<div><i style="background:var(--s' + (k + 1) + ')"></i>' + names[k] +
              ' <span class="mono">' + root.dataset['v' + k].split(',')[i] + '</span></div>';
    });
    tip.innerHTML = html; tip.style.opacity = 1;
    var left = e.clientX - r.left + 14;
    if (left > r.width - 150) left = e.clientX - r.left - tip.offsetWidth - 14;
    tip.style.left = left + 'px';
    tip.style.top = Math.min(e.clientY - r.top + 12, r.height - 90) + 'px';
  });
  root.addEventListener('mouseleave', hide);
  hide();
});
"""


def _pct(v, d=1, sign=True) -> str:
    if v is None:
        return "—"
    s = f"{100 * float(v):+.{d}f}%" if sign else f"{100 * float(v):.{d}f}%"
    return s


def _cls(v) -> str:
    if v is None:
        return "muted"
    return "pos" if float(v) > 0 else ("neg" if float(v) < 0 else "muted")


def _equity_chart(rep: dict) -> str:
    """누적수익 곡선 — 3계열 라인 + 크로스헤어."""
    W, H, L, R, T, B = 900, 300, 46, 58, 14, 26
    curves = {k: rep["curves"][k] for k, _, _ in SERIES if k in rep["curves"]}
    vals = [(v - 1) for c in curves.values() for v in c]
    lo, hi = min(vals), max(vals)
    pad = (hi - lo) * 0.08 or 0.01
    lo, hi = lo - pad, hi + pad
    n = len(rep["dates"])

    def sx(i): return L + (W - L - R) * i / (n - 1)
    def sy(v): return T + (H - T - B) * (1 - (v - lo) / (hi - lo))

    # 가로 격자 + 축 라벨
    grid, step = "", (hi - lo) / 4
    for k in range(5):
        v = lo + step * k
        y = sy(v)
        grid += (f'<line x1="{L}" y1="{y:.1f}" x2="{W - R}" y2="{y:.1f}" stroke="var(--grid)" '
                 f'stroke-width="1"/>'
                 f'<text x="{L - 8}" y="{y + 4:.1f}" text-anchor="end" font-size="11" '
                 f'fill="var(--muted)" class="mono">{100 * v:+.0f}%</text>')
    # 0% 기준선
    if lo < 0 < hi:
        grid += (f'<line x1="{L}" y1="{sy(0):.1f}" x2="{W - R}" y2="{sy(0):.1f}" '
                 f'stroke="var(--muted)" stroke-width="1" stroke-dasharray="3 3" opacity=".5"/>')

    paths, dots, ys_all, names, endlabels = "", "", [], [], ""
    for idx, (key, label, _) in enumerate(SERIES):
        if key not in curves:
            continue
        ys = [sy(v - 1) for v in curves[key]]
        ys_all.append([round(y, 1) for y in ys])
        names.append(label)
        d = " ".join(f"{'M' if i == 0 else 'L'}{sx(i):.1f} {y:.1f}" for i, y in enumerate(ys))
        paths += (f'<path d="{d}" fill="none" stroke="var(--s{idx + 1})" stroke-width="2" '
                  f'stroke-linejoin="round" stroke-linecap="round"/>')
        dots += (f'<circle class="hd" r="4.5" fill="var(--s{idx + 1})" stroke="var(--panel)" '
                 f'stroke-width="2" opacity="0"/>')
        # 끝점 직접 라벨 — 색 외의 2차 부호
        end = 100 * (curves[key][-1] - 1)
        endlabels += (f'<text x="{W - R + 4}" y="{ys[-1] + 4:.1f}" font-size="11.5" '
                      f'fill="var(--s{idx + 1})" class="mono">{end:+.1f}%</text>')

    # x축 — 월 시작점만
    xt, seen = "", set()
    for i, ds in enumerate(rep["dates"]):
        ym = ds[:7]
        if ym not in seen:
            seen.add(ym)
            xt += (f'<text x="{sx(i):.1f}" y="{H - 6}" text-anchor="middle" font-size="11" '
                   f'fill="var(--muted)" class="mono">{ds[5:7]}월</text>')

    vstr = {f"v{k}": ",".join(f"{100 * (curves[key][i] - 1):+.2f}%"
                             for i in range(n))
            for k, (key, _, _) in enumerate([s for s in SERIES if s[0] in curves])}
    attrs = " ".join(f'data-{k}="{v}"' for k, v in vstr.items())
    legend = "".join(
        f'<span><i style="background:var(--s{i + 1})"></i>{lb}</span>'
        for i, (k, lb, _) in enumerate(SERIES) if k in curves
    )
    return (
        f'<div class="legend">{legend}</div>'
        f'<div class="chart" data-chart data-x0="{L}" data-x1="{W - R}" '
        f'data-points=\'{json.dumps(ys_all)}\' data-labels=\'{json.dumps(rep["dates"])}\' '
        f'data-names=\'{json.dumps(names, ensure_ascii=False)}\' {attrs}>'
        f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="누적수익 곡선">'
        f'{grid}{paths}{endlabels}{xt}'
        f'<line class="cross" y1="{T}" y2="{H - B}" stroke="var(--muted)" stroke-width="1" '
        f'opacity="0"/>{dots}</svg><div class="tip"></div></div>'
    )


def _monthly_chart(rep: dict) -> str:
    """월별 수익률 — 전략 vs 코스피200 그룹 막대."""
    rows = rep["monthly"]
    W, H, L, R, T, B = 440, 230, 40, 10, 12, 28
    keys = [("strategy", 1), ("코스피200", 2)]
    vals = [r[k] for r in rows for k, _ in keys if r.get(k) is not None]
    lo, hi = min(vals + [0]), max(vals + [0])
    pad = (hi - lo) * 0.1 or 0.01
    lo, hi = lo - pad, hi + pad

    def sy(v): return T + (H - T - B) * (1 - (v - lo) / (hi - lo))
    zero = sy(0)
    gw = (W - L - R) / len(rows)
    bw = min(16, gw / 2.6)

    out = (f'<line x1="{L}" y1="{zero:.1f}" x2="{W - R}" y2="{zero:.1f}" '
           f'stroke="var(--muted)" stroke-width="1" opacity=".5"/>')
    for k in (lo, (lo + hi) / 2, hi):
        out += (f'<text x="{L - 6}" y="{sy(k) + 4:.1f}" text-anchor="end" font-size="10.5" '
                f'fill="var(--muted)" class="mono">{100 * k:+.0f}%</text>')
    for i, r in enumerate(rows):
        cx = L + gw * (i + 0.5)
        for j, (key, slot) in enumerate(keys):
            v = r.get(key)
            if v is None:
                continue
            x = cx + (j - 0.5) * (bw + 2) - bw / 2
            y, h = (sy(v), zero - sy(v)) if v >= 0 else (zero, sy(v) - zero)
            h = max(abs(h), 1.5)
            rad = min(4, bw / 2)
            out += (f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{h:.1f}" '
                    f'rx="{rad}" fill="var(--s{slot})"><title>{r["month"]} '
                    f'{"전략" if slot == 1 else "코스피200"} {100 * v:+.2f}%</title></rect>')
        out += (f'<text x="{cx:.1f}" y="{H - 8}" text-anchor="middle" font-size="10.5" '
                f'fill="var(--muted)" class="mono">{r["month"][5:]}월</text>')
    legend = ('<span><i style="background:var(--s1)"></i>전략</span>'
              '<span><i style="background:var(--s2)"></i>코스피200</span>')
    return (f'<div class="legend">{legend}</div>'
            f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="월별 수익률">{out}</svg>')


def _drawdown_chart(rep: dict) -> str:
    """낙폭 — 전략 vs 코스피200. 아래로 내려갈수록 나쁘다."""
    W, H, L, R, T, B = 440, 230, 44, 10, 12, 28
    series = []
    for key, slot in (("strategy", 1), ("코스피200", 2)):
        if key not in rep["curves"]:
            continue
        eq, peak, dd = rep["curves"][key], -1e9, []
        for v in eq:
            peak = max(peak, v)
            dd.append(v / peak - 1)
        series.append((key, slot, dd))
    lo = min(min(d) for _, _, d in series)
    lo = lo * 1.08 or -0.01
    n = len(rep["dates"])

    def sx(i): return L + (W - L - R) * i / (n - 1)
    def sy(v): return T + (H - T - B) * (v / lo)

    out = ""
    for k in (0, lo / 2, lo):
        out += (f'<line x1="{L}" y1="{sy(k):.1f}" x2="{W - R}" y2="{sy(k):.1f}" '
                f'stroke="var(--grid)" stroke-width="1"/>'
                f'<text x="{L - 6}" y="{sy(k) + 4:.1f}" text-anchor="end" font-size="10.5" '
                f'fill="var(--muted)" class="mono">{100 * k:.0f}%</text>')
    for key, slot, dd in series:
        pts = " ".join(f"{'M' if i == 0 else 'L'}{sx(i):.1f} {sy(v):.1f}"
                       for i, v in enumerate(dd))
        area = pts + f" L{sx(n - 1):.1f} {sy(0):.1f} L{L:.1f} {sy(0):.1f} Z"
        out += (f'<path d="{area}" fill="var(--s{slot})" opacity=".12"/>'
                f'<path d="{pts}" fill="none" stroke="var(--s{slot})" stroke-width="2" '
                f'stroke-linejoin="round"/>')
        worst = min(dd)
        out += (f'<text x="{W - R}" y="{sy(worst) + 13:.1f}" text-anchor="end" font-size="11" '
                f'fill="var(--s{slot})" class="mono">{100 * worst:.1f}%</text>')
    seen = set()
    for i, ds in enumerate(rep["dates"]):
        if ds[:7] not in seen:
            seen.add(ds[:7])
            out += (f'<text x="{sx(i):.1f}" y="{H - 8}" text-anchor="middle" font-size="10.5" '
                    f'fill="var(--muted)" class="mono">{ds[5:7]}월</text>')
    legend = ('<span><i style="background:var(--s1)"></i>전략</span>'
              '<span><i style="background:var(--s2)"></i>코스피200</span>')
    return (f'<div class="legend">{legend}</div>'
            f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="낙폭">{out}</svg>')


def _attribution_chart(rep: dict, names: dict) -> str:
    """종목별 기여도 — 상하위 8개 발산 막대."""
    attr = rep["attribution"]
    if not attr:
        return '<p class="note">거래 기록 없음</p>'
    rows = attr[:8] + attr[-8:] if len(attr) > 16 else attr
    mx = max(abs(r["contribution"]) for r in rows) or 0.01
    out = ""
    for r in rows:
        nm = names.get(r["code"], r["code"])
        v = r["contribution"]
        w = 100 * abs(v) / mx / 2
        bar = (f'<span style="display:inline-block;height:9px;border-radius:3px;'
               f'width:{w:.1f}%;background:var(--{"pos" if v > 0 else "neg"})"></span>')
        side = (f'<div style="display:flex;justify-content:flex-end;width:50%">{bar}</div>'
                f'<div style="width:50%"></div>') if v < 0 else \
               (f'<div style="width:50%"></div><div style="width:50%">{bar}</div>')
        out += (f'<tr><td>{escape(nm)} <span class="muted mono">{r["code"]}</span></td>'
                f'<td style="width:44%"><div style="display:flex;align-items:center">{side}</div></td>'
                f'<td class="mono {_cls(v)}">{_pct(v, 2)}</td>'
                f'<td class="mono muted">{r["exits"]}회</td></tr>')
    return (f'<div class="tw"><table><thead><tr><th>종목</th>'
            f'<th style="text-align:center">기여도</th><th>수익 기여</th><th>청산</th>'
            f'</tr></thead><tbody>{out}</tbody></table></div>')


LABELS = [("total_return", "누적수익", True), ("cagr", "CAGR", True),
          ("sharpe", "Sharpe", False), ("sortino", "Sortino", False),
          ("calmar", "Calmar", False), ("volatility", "변동성", True),
          ("max_drawdown", "최대낙폭", True), ("hit_rate", "적중률", True)]


def _metrics_table(rep: dict) -> str:
    order = ["strategy", "코스피200", "코스피", "buy_and_hold"]
    disp = {"strategy": "전략", "buy_and_hold": "유니버스 동일가중"}
    cols = [c for c in order if c in rep["metrics"]]
    head = "".join(f"<th>{escape(disp.get(c, c))}</th>" for c in cols)
    body = ""
    for key, label, as_pct in LABELS:
        cells = ""
        for i, c in enumerate(cols):
            v = rep["metrics"][c].get(key)
            txt = _pct(v, 1, sign=key != "hit_rate") if as_pct else f"{float(v):.2f}"
            cls = _cls(v) if (as_pct and key != "hit_rate") or key in ("sharpe", "sortino", "calmar") else ""
            strong = ' style="font-weight:600"' if i == 0 else ' class="muted"'
            cells += f'<td{strong}><span class="mono {cls}">{txt}</span></td>'
        body += f"<tr><td>{label}</td>{cells}</tr>"
    return f'<div class="tw"><table><thead><tr><th>지표</th>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _diag_section(rep: dict) -> str:
    ic = rep["diagnostics"].get("rank_ic", {})
    sp = rep["diagnostics"].get("decile_spread", {})
    cost = rep["signal_stats"].get("round_trip_cost", 0)
    t = abs(float(ic.get("t_stat", 0)))
    if t < 2:
        verdict = ("<b>이 구간에서 방향 예측력이 통계적으로 확인되지 않는다</b> (|t| &lt; 2). "
                   "위 성과는 매매 규칙·리스크 오버레이·국면의 산물일 수 있다 — "
                   "모델이 만든 것이라고 말하려면 더 긴 구간이 필요하다.")
    elif float(sp.get("spread_mean", 0)) <= float(cost):
        verdict = "순위는 맞히지만 십분위 스프레드가 왕복비용 이하 — 회전을 줄여야 한다."
    else:
        verdict = "방향 알파가 거래비용을 넘는다."
    rows = [
        ("랭크 IC", f'{ic.get("ic_mean", 0):+.4f}', f't = {ic.get("t_stat", 0):+.2f}',
         f'{ic.get("n_dates", 0)}일 · 양수비율 {100 * ic.get("ic_positive_rate", 0):.1f}%'),
        ("십분위 스프레드", f'{sp.get("spread_mean", 0):+.4f}', f't = {sp.get("t_stat", 0):+.2f}',
         f'왕복비용 {100 * cost:.2f}%'),
    ]
    body = "".join(
        f'<tr><td>{a}</td><td class="mono">{b}</td><td class="mono">{c}</td>'
        f'<td class="muted">{d}</td></tr>' for a, b, c, d in rows)
    return (f'<div class="tw"><table><thead><tr><th>진단</th><th>값</th><th>유의성</th><th></th>'
            f'</tr></thead><tbody>{body}</tbody></table></div>'
            f'<div class="verdict">판정 — {verdict}</div>')


def _activity(rep: dict) -> str:
    s = rep["signal_stats"]
    br = s.get("blocked_by_reason", {})
    rows = [
        ("기권률", f'{100 * s.get("abstain_rate", 0):.1f}%'),
        ("평균 노출도", f'{100 * s.get("avg_gross_exposure", 0):.1f}%'),
        ("평균 보유종목", f'{s.get("avg_n_positions", 0):.1f}개'),
        ("평균 보유일수", f'{s.get("avg_holding_days", 0):.1f}일'),
        ("리밸런싱", f'{s.get("n_rebalances", 0)}회'),
        ("체결", f'{s.get("n_trades", 0)}건 (신규 {s.get("n_entries", 0)})'),
        ("연 회전율", f'{s.get("annual_turnover", 0):.1f}회'),
        ("실지불 비용", f'{100 * s.get("total_cost_pct", 0):.2f}% (연 {100 * s.get("annual_cost_pct", 0):.2f}%)'),
        ("리스크 차단", f'손절 {br.get("손절", 0)} · 익절 {br.get("익절", 0)}'),
    ]
    body = "".join(f'<tr><td>{a}</td><td class="mono">{b}</td></tr>' for a, b in rows)
    return f'<div class="tw"><table><tbody>{body}</tbody></table></div>'


def render_html(rep: dict, names: dict, fragment: bool = False) -> str:
    w, m = rep["window"], rep["metrics"]["strategy"]
    k200 = rep["metrics"].get("코스피200", {})
    cfg = rep["config"]
    tiles = [
        ("누적수익", _pct(m["total_return"]), f'코스피200 {_pct(k200.get("total_return"))}', _cls(m["total_return"])),
        ("Sharpe", f'{m["sharpe"]:.2f}', f'코스피200 {k200.get("sharpe", 0):.2f}', _cls(m["sharpe"])),
        ("최대낙폭", _pct(m["max_drawdown"]), f'코스피200 {_pct(k200.get("max_drawdown"))}', "neg"),
        ("평균 노출", f'{100 * rep["signal_stats"].get("avg_gross_exposure", 0):.0f}%',
         f'{rep["signal_stats"].get("avg_n_positions", 0):.1f}종목', ""),
    ]
    tile_html = "".join(
        f'<div class="tile"><div class="k">{k}</div>'
        f'<div class="v mono {c}">{v}</div><div class="n">{n}</div></div>'
        for k, v, n, c in tiles)
    chips = "".join(f'<span class="chip">{a} <b>{b}</b></span>' for a, b in [
        ("구간", f'{w["start"]} ~ {w["end"]}'), ("거래일", f'{w["n_days"]}일'),
        ("보유종목", f'상위 {cfg["top_n"]}'), ("리밸런싱", f'{cfg["rebalance_days"]}일'),
        ("기권", f'상위 {cfg["abstain_percentile"]}%'), ("체크포인트", rep["checkpoint"]),
    ])
    monthly_rows = "".join(
        f'<tr><td class="mono">{r["month"]}</td>'
        + "".join(f'<td class="mono {_cls(r.get(k))}">{_pct(r.get(k), 2)}</td>'
                  for k in ("strategy", "코스피200", "코스피", "buy_and_hold"))
        + "</tr>" for r in rep["monthly"])

    body = f"""
<div class="wrap">
<header>
  <h1>최근 6개월 백테스트</h1>
  <div class="sub">거래비용을 반영한 실측이다. 모의투자와 <b>같은 매매 코드</b>로 돌았다 —
  구간만 최근 6개월로 잘랐다.</div>
</header>
<div class="chips">{chips}</div>
<div class="tiles">{tile_html}</div>

<section>
  <h2>누적수익</h2>
  <p class="note">1에서 시작한 자산곡선. 선 끝의 숫자가 구간 누적수익이다.</p>
  {_equity_chart(rep)}
</section>

<div class="two">
  <section>
    <h2>월별 수익률</h2>
    <p class="note">전략이 이긴 달과 진 달.</p>
    {_monthly_chart(rep)}
  </section>
  <section>
    <h2>낙폭</h2>
    <p class="note">고점 대비 하락폭. 0에 붙어 있을수록 좋다.</p>
    {_drawdown_chart(rep)}
  </section>
</div>

<section>
  <h2>지표 비교</h2>
  <p class="note">같은 구간·같은 날짜축에서 계산했다.</p>
  {_metrics_table(rep)}
  <details><summary>월별 수치 표로 보기</summary>
    <div class="tw"><table><thead><tr><th>월</th><th>전략</th><th>코스피200</th>
    <th>코스피</th><th>유니버스 동일가중</th></tr></thead>
    <tbody>{monthly_rows}</tbody></table></div>
  </details>
</section>

<div class="two">
  <section>
    <h2>종목별 기여</h2>
    <p class="note">기여도 = Σ 비중 × 일간수익. 합은 비용 차감 전 전략수익과 같다.</p>
    {_attribution_chart(rep, names)}
  </section>
  <section>
    <h2>거래 활동</h2>
    <p class="note">비용은 추정이 아니라 실제 차감한 값이다.</p>
    {_activity(rep)}
  </section>
</div>

<section>
  <h2>예측력 진단</h2>
  <p class="note">성과 지표는 모델의 예측력과 매매 규칙이 섞인 결과다.
  아래 두 지표는 거래 로직을 통째로 우회해 예측력만 직접 잰다.</p>
  {_diag_section(rep)}
</section>

<footer>생성 {rep["generated_at"]} · scripts/backtest_recent.py</footer>
</div>
<script>{_JS}</script>
"""
    if fragment:
        return (f"<title>Phase1 6개월 백테스트</title>\n{_FONTS}\n"
                f"<style>{_CSS}</style>\n{body}")
    return (
        '<!doctype html><html lang="ko"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>Phase1 6개월 백테스트</title>"
        + _FONTS +
        f"<style>{_CSS}</style></head><body>{body}</body></html>"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", type=int, default=6)
    ap.add_argument("--checkpoint")
    ap.add_argument("--end", help="YYYY-MM-DD. 기본은 데이터의 마지막 날")
    ap.add_argument("--fragment", action="store_true",
                    help="<html> 래퍼 없이 본문만 — 외부 퍼블리시용")
    args = ap.parse_args()

    setup_logging(run_name="backtest_recent")
    cfg = load_config().raw

    ckpt = Path(args.checkpoint) if args.checkpoint else CKPT_DIR / "phase1_0db568ae.pt"
    if not ckpt.is_absolute():
        ckpt = PROJECT_ROOT / ckpt
    log.info("체크포인트 %s", ckpt.name)

    loaded = load_model(ckpt)
    bundle = load_features(cfg, loaded)
    preds, prices = predict_split(loaded, bundle, cfg, "test")
    log.info("test 예측 %d행 (%s ~ %s)", len(preds), preds["date"].min(), preds["date"].max())

    end = pd.to_datetime(args.end).date() if args.end else max(prices["date"])
    start = (pd.Timestamp(end) - pd.DateOffset(months=args.months)).date()
    log.info("구간 %s ~ %s (%d개월)", start, end, args.months)

    w_preds = preds[(preds["date"] >= start) & (preds["date"] <= end)]
    # 가격은 구간보다 앞에서 시작해야 첫날의 전일 종가가 있다.
    pad = (pd.Timestamp(start) - pd.DateOffset(days=20)).date()
    w_prices = prices[(prices["date"] >= pad) & (prices["date"] <= end)]
    if w_preds.empty:
        log.error("구간에 예측이 없다"); return 1

    res = run_backtest(w_preds, w_prices, cfg)
    dates = list(res.returns.index)

    bh = buy_and_hold(w_prices).reindex(dates).fillna(0.0)
    series = {"strategy": res.returns, "buy_and_hold": bh}
    metrics = {"strategy": res.metrics,
               "buy_and_hold": summarize(bh)}
    for code, name in INDEXES.items():
        r = index_returns(code, pd.Index(dates))
        if r is not None:
            series[name] = r
            metrics[name] = summarize(r)

    curves = {k: [round(v, 6) for v in equity_curve(s).tolist()]
              for k, s in series.items()}

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "checkpoint": ckpt.name,
        "window": {"start": str(dates[0]), "end": str(dates[-1]),
                   "months": args.months, "n_days": len(dates)},
        "config": {k: cfg["backtest"].get(k) for k in ("rebalance_days", "execution_lag_days")}
        | {"top_n": cfg["trading"]["direction"]["top_n"],
           "exit_rank": cfg["trading"]["direction"]["exit_rank"],
           "abstain_percentile": cfg["trading"]["abstain"].get("percentile"),
           "exposure_scaling": cfg["trading"]["sizing"].get("exposure_scaling")},
        "metrics": metrics,
        "dates": [str(d) for d in dates],
        "curves": curves,
        "monthly": monthly_table(series),
        "signal_stats": res.signal_stats,
        "diagnostics": res.diagnostics,
        "attribution": attribution(res.trades, w_prices, dates),
        "n_trades": int(len(res.trades)),
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = REPORTS_DIR / f"backtest_recent_{args.months}m_{stamp}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    uni = yaml.safe_load((PROJECT_ROOT / "configs" / "universe.yaml").read_text())
    names = {str(u["code"]).zfill(6): u["name"] for u in uni["universe"]}
    html = out.with_suffix(".html")
    html.write_text(render_html(report, names, fragment=args.fragment), encoding="utf-8")

    log.info("=" * 60)
    for name, m in metrics.items():
        log.info("%-16s 누적 %+7.2f%%  Sharpe %6.2f  MDD %+7.2f%%",
                 name, 100 * m["total_return"], m["sharpe"], 100 * m["max_drawdown"])
    log.info("=" * 60)
    log.info("리포트 %s", out)
    log.info("대시보드 %s", html)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
