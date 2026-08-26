"""대시보드 HTML 생성 — 외부 라이브러리 없이 문자열로 만든다.

차트도 CSS 와 인라인 SVG 로 그린다. CDN 을 붙이면 오프라인에서 화면이 깨지고,
번들러를 붙이면 이 저장소에 프런트엔드 빌드가 하나 더 생긴다.
대시보드는 결과를 보는 도구지 이 프로젝트의 산출물이 아니다.

화면 순서는 **판단 순서와 같게** 둔다: 오늘의 판단 → 근거(백테스트) → 모델.
"""

from __future__ import annotations

from html import escape
from typing import Any

_CSS = """
:root {
  --bg: #f6f7f9; --panel: #ffffff; --ink: #16181d; --muted: #6b7280;
  --line: #e3e6eb; --accent: #2563eb; --pos: #067647; --neg: #b42318;
  --warn: #b54708; --chip: #eef2ff;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0e1116; --panel: #161a21; --ink: #e6e8ec; --muted: #9aa3b2;
    --line: #262c36; --accent: #6ea8fe; --pos: #46c08d; --neg: #f2777a;
    --warn: #e0a458; --chip: #1c2331;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--ink);
  font: 14px/1.55 -apple-system, BlinkMacSystemFont, "Apple SD Gothic Neo",
        "Pretendard", "Noto Sans KR", sans-serif;
}
.wrap { max-width: 1180px; margin: 0 auto; padding: 28px 20px 64px; }
header { display: flex; flex-wrap: wrap; align-items: baseline; gap: 12px; margin-bottom: 4px; }
h1 { font-size: 20px; margin: 0; letter-spacing: -0.01em; }
.sub { color: var(--muted); font-size: 13px; }
.chips { display: flex; flex-wrap: wrap; gap: 6px; margin: 14px 0 22px; }
.chip {
  background: var(--chip); border: 1px solid var(--line); border-radius: 999px;
  padding: 3px 10px; font-size: 12px; color: var(--muted);
}
.chip b { color: var(--ink); font-weight: 600; }
.chip.warn { color: var(--warn); border-color: var(--warn); }
.grid { display: grid; gap: 14px; }
.tiles { grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); margin-bottom: 22px; }
.tile {
  background: var(--panel); border: 1px solid var(--line); border-radius: 10px;
  padding: 14px 16px;
}
.tile .k { color: var(--muted); font-size: 12px; }
.tile .v { font-size: 22px; font-weight: 650; letter-spacing: -0.02em; margin-top: 2px; }
.tile .n { color: var(--muted); font-size: 12px; }
section {
  background: var(--panel); border: 1px solid var(--line); border-radius: 12px;
  padding: 18px 20px; margin-bottom: 16px;
}
section > h2 {
  font-size: 15px; margin: 0 0 4px; display: flex; align-items: baseline; gap: 10px;
}
section > h2 small { color: var(--muted); font-weight: 400; font-size: 12px; }
.note { color: var(--muted); font-size: 12.5px; margin: 0 0 14px; }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th, td { text-align: right; padding: 7px 8px; border-bottom: 1px solid var(--line); }
th:first-child, td:first-child { text-align: left; }
th { color: var(--muted); font-weight: 500; font-size: 12px; white-space: nowrap; }
tbody tr:last-child td { border-bottom: none; }
.mono { font-variant-numeric: tabular-nums; font-feature-settings: "tnum"; }
.pos { color: var(--pos); } .neg { color: var(--neg); } .muted { color: var(--muted); }
.tag {
  font-size: 11px; padding: 1px 7px; border-radius: 5px; border: 1px solid var(--line);
}
.tag.buy { color: var(--pos); border-color: var(--pos); }
.tag.sell { color: var(--neg); border-color: var(--neg); }
.tag.dry { color: var(--warn); border-color: var(--warn); }
.bar { background: var(--line); border-radius: 3px; height: 6px; overflow: hidden; }
.bar > i { display: block; height: 100%; background: var(--accent); }
.two { display: grid; grid-template-columns: 1fr 1fr; gap: 22px; }
@media (max-width: 780px) { .two { grid-template-columns: 1fr; } }
.empty { color: var(--muted); padding: 10px 0; }
.reason { color: var(--muted); font-size: 12px; }
footer { color: var(--muted); font-size: 12px; text-align: center; margin-top: 28px; }
svg { display: block; width: 100%; height: auto; }
"""


# ------------------------------------------------------------------ 유틸
def _n(v, digits: int = 0) -> str:
    try:
        return f"{float(v):,.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def _pct(v, digits: int = 1) -> str:
    try:
        return f"{float(v) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return "—"


def _signed(v, digits: int = 4) -> str:
    try:
        return f"{float(v):+.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def _cls(v) -> str:
    try:
        return "pos" if float(v) > 0 else ("neg" if float(v) < 0 else "")
    except (TypeError, ValueError):
        return ""


def _tile(k: str, v: str, note: str = "") -> str:
    n = f'<div class="n">{escape(note)}</div>' if note else ""
    return (f'<div class="tile"><div class="k">{escape(k)}</div>'
            f'<div class="v mono">{v}</div>{n}</div>')


def _section(title: str, note: str, body: str, sub: str = "") -> str:
    s = f" <small>{escape(sub)}</small>" if sub else ""
    p = f'<p class="note">{escape(note)}</p>' if note else ""
    return f"<section><h2>{escape(title)}{s}</h2>{p}{body}</section>"


def _table(headers: list[str], rows: list[list[str]], empty: str = "기록 없음") -> str:
    if not rows:
        return f'<div class="empty">{escape(empty)}</div>'
    head = "".join(f"<th>{escape(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


# ------------------------------------------------------------------ 섹션
def _header(d: dict) -> str:
    data = d.get("data") or {}
    ckpt = d.get("checkpoint") or {}
    run = d.get("run") or {}
    plan = run.get("plan", {})

    chips = [
        '<span class="chip">환경 <b>모의투자(mock)</b></span>',
        f'<span class="chip">체크포인트 <b>{escape(str(ckpt.get("name", "없음")))}</b></span>',
    ]
    if data.get("available"):
        chips.append(
            f'<span class="chip">데이터 <b>{escape(str(data["last_date"]))}</b>까지 '
            f'· {_n(data["rows"])}행 · {data["codes"]}종목</span>'
        )
    else:
        chips.append('<span class="chip warn">패널 없음 — build_features.py 필요</span>')
    if run:
        mode = "계획만(dry-run)" if run.get("dry_run", True) else "주문 전송됨"
        chips.append(f'<span class="chip">마지막 실행 <b>{escape(mode)}</b> '
                     f'{escape(str(plan.get("generated_at", "")))}</span>')
    return (
        "<header><h1>주가예측 자동매매 대시보드</h1>"
        f'<span class="sub">생성 {escape(str(d.get("generated_at", "")))}</span></header>'
        f'<div class="chips">{"".join(chips)}</div>'
    )


def _tiles(d: dict) -> str:
    run = d.get("run") or {}
    plan = run.get("plan", {})
    stats = plan.get("stats", {})
    bt = (d.get("backtest") or {}).get("main", {})
    strat = bt.get("strategy", {})

    tiles = [
        _tile("총자산", _n(plan.get("equity")) + "원" if plan else "—",
              f"주문가능 {_n(plan.get('cash'))}원" if plan else "실행 기록 없음"),
        _tile("오늘 주문", f"{len(plan.get('orders', []))}건" if plan else "—",
              "리밸런싱" if plan.get("rebalancing") else "손절/익절 점검만"),
        _tile("기권률", _pct(stats.get("abstain_rate")) if stats else "—",
              f"후보 {stats.get('n_candidates', 0)}종목 중" if stats else ""),
        _tile("목표 노출도", _pct(stats.get("target_gross")) if stats else "—",
              "나머지는 현금"),
        _tile("백테스트 Sharpe", _n(strat.get("sharpe"), 2) if strat else "—",
              f"CAGR {_pct(strat.get('cagr'))}" if strat else ""),
    ]
    return f'<div class="grid tiles">{"".join(tiles)}</div>'


def _orders_section(d: dict) -> str:
    run = d.get("run")
    if not run:
        return _section(
            "오늘의 판단", "아직 실행 기록이 없다. `python scripts/paper_trade.py` 로 시작한다.",
            '<div class="empty">모의투자 실행 기록 없음</div>',
        )

    plan = run.get("plan", {})
    sent = {(r["code"], r["side"]): r for r in run.get("orders_sent", [])}
    rows = []
    for o in plan.get("orders", []):
        key = (o["code"], o["side"])
        r = sent.get(key, {})
        side = ("<span class='tag buy'>매수</span>" if o["side"] == "buy"
                else "<span class='tag sell'>매도</span>")
        if r.get("dry_run", True):
            status = "<span class='tag dry'>미전송</span>"
        elif r.get("error"):
            status = f"<span class='tag sell'>실패</span> <span class='reason'>{escape(str(r['error']))}</span>"
        else:
            status = f"<span class='tag buy'>전송</span> <span class='reason'>#{escape(str(r.get('order_no', '')))}</span>"
        rows.append([
            f"<b>{escape(o['code'])}</b>", side,
            f'<span class="mono">{_n(o["quantity"])}</span>',
            f'<span class="mono">{_n(o["price"])}</span>',
            f'<span class="mono">{_n(o.get("amount"))}</span>',
            f'<span class="mono muted">{_pct(o["weight_from"])} → {_pct(o["weight_to"])}</span>',
            status,
        ])

    table = _table(
        ["종목", "구분", "수량", "단가", "금액", "비중 변화", "상태"], rows,
        "주문 없음 — 오늘은 거래하지 않는다",
    )

    extra = ""
    if plan.get("forced_exits"):
        extra += (f'<p class="note">강제 청산: '
                  f'{escape(", ".join(plan["forced_exits"]))} '
                  f'(손절/익절은 신호보다 먼저 적용된다)</p>')
    if plan.get("blocked_by_reason"):
        items = " · ".join(f"{escape(k)} {v}건"
                           for k, v in sorted(plan["blocked_by_reason"].items()))
        extra += f'<p class="note">리스크 차단: {items}</p>'
    for n in plan.get("notes", []):
        extra += f'<p class="note">⚠️ {escape(str(n))}</p>'

    return _section(
        "오늘의 판단", "", table + extra,
        sub=f"기준일 {plan.get('decision_date', '')}",
    )


def _signals_section(d: dict) -> str:
    run = d.get("run")
    if not run:
        return ""
    plan = run.get("plan", {})
    signals = plan.get("signals", [])
    if not signals:
        return ""

    buys = [s for s in signals if s["action"] == "buy"]
    buys.sort(key=lambda s: -s["target_weight"])
    holds = [s for s in signals if s["action"] == "hold"]
    holds.sort(key=lambda s: -s["confidence"])

    def rows_of(items):
        out = []
        for s in items[:12]:
            out.append([
                f"<b>{escape(s['code'])}</b>" + (" <span class='tag'>보유</span>"
                                                 if s.get("held") else ""),
                f'<span class="mono">{_pct(s["target_weight"])}</span>',
                f'<span class="mono">{_n(s["confidence"], 2)}</span>',
                f'<span class="reason">{escape(str(s["reason"]))}</span>',
            ])
        return out

    left = _table(["매수 선정", "목표비중", "확신도", "근거"], rows_of(buys), "선정 없음")
    right = _table(["미선정", "목표비중", "확신도", "근거"], rows_of(holds), "없음")
    stats = plan.get("stats", {})
    note = (
        f"기권 {stats.get('abstain', 0)}종목 — 신뢰구간 폭이 임계 "
        f"{plan.get('abstain_threshold', 0):.4f} 를 넘어 판단을 보류했다. "
        "기권은 방향 판단보다 먼저 온다."
    )
    return _section("신호 상세", note, f'<div class="two"><div>{left}</div><div>{right}</div></div>')


def _holdings_section(d: dict) -> str:
    """보유 현황은 실행 기록의 신호에 붙은 current_weight 로 재구성한다."""
    run = d.get("run")
    state = d.get("state") or {}
    if not run:
        return ""
    plan = run.get("plan", {})
    held = [s for s in plan.get("signals", []) if s.get("held")]
    if not held:
        return ""
    rows = []
    for s in sorted(held, key=lambda x: -x["current_weight"]):
        entry = (state.get("entry_dates") or {}).get(s["code"], "")
        rows.append([
            f"<b>{escape(s['code'])}</b>",
            f'<span class="mono">{_pct(s["current_weight"])}</span>',
            f'<span class="mono">{_pct(s["target_weight"])}</span>',
            f'<span class="mono">{_n(s.get("price"))}</span>',
            f'<span class="muted">{escape(str(entry))}</span>',
        ])
    return _section(
        "보유 현황", "진입일은 로컬 기록이고, 수량·매입가는 브로커가 정본이다.",
        _table(["종목", "현재 비중", "목표 비중", "가격", "진입일"], rows),
    )


def _backtest_section(d: dict) -> str:
    bt = d.get("backtest") or {}
    main = bt.get("main")
    if not main:
        return _section("백테스트", "", '<div class="empty">리포트 없음 — scripts/backtest.py</div>')

    strat, bh = main.get("strategy", {}), main.get("buy_and_hold", {})
    labels = [("cagr", "CAGR", True), ("sharpe", "Sharpe", False),
              ("sortino", "Sortino", False), ("calmar", "Calmar", False),
              ("volatility", "변동성", True), ("max_drawdown", "최대낙폭", True),
              ("hit_rate", "적중률", True), ("total_return", "누적수익", True)]
    rows = []
    for key, label, as_pct in labels:
        fmt = _pct if as_pct else (lambda v: _n(v, 2))
        rows.append([
            label,
            f'<span class="mono {_cls(strat.get(key))}">{fmt(strat.get(key))}</span>',
            f'<span class="mono muted">{fmt(bh.get(key))}</span>',
        ])
    perf = _table(["지표", "전략", "매수후보유"], rows)

    diag = main.get("diagnostics", {})
    ic = diag.get("rank_ic", {})
    spread = diag.get("decile_spread", {})
    stats = main.get("signal_stats", {})
    cost = stats.get("round_trip_cost", 0)

    if abs(float(ic.get("t_stat", 0))) < 2:
        verdict = "방향 예측력 확인 안 됨 (|t| &lt; 2) — 매매 규칙을 손봐도 성과는 안 나온다"
    elif float(spread.get("spread_mean", 0)) <= float(cost):
        verdict = "순위는 맞히지만 십분위 스프레드가 왕복비용 이하 — 회전을 줄여야 한다"
    else:
        verdict = "방향 알파가 거래비용을 넘는다"

    diag_rows = [
        ["랭크 IC",
         f'<span class="mono {_cls(ic.get("ic_mean"))}">{_signed(ic.get("ic_mean"))}</span>',
         f'<span class="mono">t = {_signed(ic.get("t_stat"), 2)}</span>',
         f'<span class="muted">{ic.get("n_dates", 0)}일 · 양수비율 {_pct(ic.get("ic_positive_rate"))}</span>'],
        ["십분위 스프레드",
         f'<span class="mono {_cls(spread.get("spread_mean"))}">{_signed(spread.get("spread_mean"))}</span>',
         f'<span class="mono">t = {_signed(spread.get("t_stat"), 2)}</span>',
         f'<span class="muted">왕복비용 {_pct(cost, 2)}</span>'],
    ]
    diag_tbl = _table(["진단", "값", "유의성", ""], diag_rows)

    activity = _table(
        ["거래 활동", "값"],
        [
            ["기권률", f'<span class="mono">{_pct(stats.get("abstain_rate"))}</span>'],
            ["평균 노출도", f'<span class="mono">{_pct(stats.get("avg_gross_exposure"))}</span>'],
            ["평균 보유종목", f'<span class="mono">{_n(stats.get("avg_n_positions"), 1)}개</span>'],
            ["평균 보유일수", f'<span class="mono">{_n(stats.get("avg_holding_days"), 1)}일</span>'],
            ["연 회전율", f'<span class="mono">{_n(stats.get("annual_turnover"), 1)}회</span>'],
            ["실지불 비용(연)", f'<span class="mono neg">{_pct(stats.get("annual_cost_pct"), 2)}</span>'],
            ["체결 건수", f'<span class="mono">{_n(stats.get("n_trades"))}건</span>'],
        ],
    )

    body = (
        f'<div class="two"><div>{perf}</div><div>{activity}</div></div>'
        f'<p class="note" style="margin-top:16px">성과보다 먼저 읽는 것 — 예측력 진단</p>'
        f"{diag_tbl}"
        f'<p class="note">판정: {verdict}</p>'
    )
    return _section(
        "백테스트", "거래비용을 반영한 실측이다. 이 화면의 매매 규칙과 같은 코드로 돌았다.",
        body, sub=f"{main.get('split', '')} 구간 · {escape(str(main.get('checkpoint', '')))}",
    )


def _variants_section(d: dict) -> str:
    variants = [v for v in (d.get("backtest") or {}).get("variants", []) if v.get("variant")]
    if len(variants) < 2:
        return ""
    headers = ["지표"] + [v["variant"] for v in variants]
    keys = [("sharpe", "Sharpe", False), ("cagr", "CAGR", True),
            ("max_drawdown", "최대낙폭", True)]
    rows = []
    for key, label, as_pct in keys:
        fmt = _pct if as_pct else (lambda v: _n(v, 2))
        rows.append([label] + [
            f'<span class="mono">{fmt(v.get("strategy", {}).get(key))}</span>'
            for v in variants
        ])
    for key, label in [("annual_turnover", "연 회전율"), ("avg_holding_days", "평균 보유일수")]:
        rows.append([label] + [
            f'<span class="mono">{_n(v.get("signal_stats", {}).get(key), 1)}</span>'
            for v in variants
        ])
    rows.append(["실지불 비용(연)"] + [
        f'<span class="mono neg">{_pct(v.get("signal_stats", {}).get("annual_cost_pct"), 2)}</span>'
        for v in variants
    ])
    return _section(
        "매매 규칙 비교",
        "예측은 한 번만 계산하고 규칙만 갈아끼운 결과다. 격차 중 비용 몫과 종목선택 몫을 가른다.",
        _table(headers, rows),
    )


def _sparkline(history: list[dict]) -> str:
    """학습 곡선. train/val 두 줄만 그린다."""
    if len(history) < 2:
        return ""
    tr = [float(h["train"]) for h in history]
    va = [float(h["val"]) for h in history]
    lo, hi = min(tr + va), max(tr + va)
    span = (hi - lo) or 1.0
    w, h = 320, 90

    def path(values, color):
        pts = " ".join(
            f"{i / (len(values) - 1) * w:.1f},{h - (v - lo) / span * h:.1f}"
            for i, v in enumerate(values)
        )
        return f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{pts}"/>'

    return (
        f'<svg viewBox="0 0 {w} {h}" preserveAspectRatio="none" style="max-height:90px">'
        f'{path(tr, "var(--muted)")}{path(va, "var(--accent)")}</svg>'
        '<p class="note">회색 train · 파랑 val — val 이 먼저 꺾이는 지점이 best epoch 다</p>'
    )


def _equity_chart(curve: list[dict]) -> str:
    """계좌 곡선 + KOSPI 를 **누적수익률(%)로 정규화해** 겹쳐 그린다.

    금액과 지수는 스케일이 달라 그대로 겹치면 비교가 안 된다.
    각자 첫 값을 100 으로 놓으면 같은 축에서 읽힌다.
    """
    if len(curve) < 2:
        return ""

    def norm(key: str) -> list[float] | None:
        vals = [(r["date"], r.get(key)) for r in curve]
        have = [(dt, float(v)) for dt, v in vals if v not in (None, 0)]
        if len(have) < 2 or have[0][1] == 0:
            return None
        base = have[0][1]
        return [v / base - 1.0 for _, v in have]

    eq = norm("equity")
    bm = norm("kospi")
    if not eq:
        return ""

    series = [s for s in (eq, bm) if s]
    lo = min(min(s) for s in series)
    hi = max(max(s) for s in series)
    span = (hi - lo) or 0.01
    pad = span * 0.12
    lo, hi = lo - pad, hi + pad
    span = hi - lo
    w, h = 720, 180

    def path(values, color, dash=""):
        if len(values) < 2:
            return ""
        pts = " ".join(
            f"{i / (len(values) - 1) * w:.1f},{h - (v - lo) / span * h:.1f}"
            for i, v in enumerate(values)
        )
        da = f' stroke-dasharray="4 3"' if dash else ""
        return (f'<polyline fill="none" stroke="{color}" stroke-width="2" '
                f'points="{pts}"{da}/>')

    zero = h - (0 - lo) / span * h
    return (
        f'<svg viewBox="0 0 {w} {h}" preserveAspectRatio="none" '
        f'style="width:100%;height:180px">'
        f'<line x1="0" y1="{zero:.1f}" x2="{w}" y2="{zero:.1f}" '
        f'stroke="var(--line)" stroke-width="1"/>'
        f'{path(bm, "var(--muted)", dash=True) if bm else ""}'
        f'{path(eq, "var(--accent)")}</svg>'
        f'<p class="note">파랑 계좌 · 회색점선 KOSPI — 각자 시작일을 0% 로 놓았다. '
        f'가로 눈금은 거래일 순서다(달력 간격 아님)</p>'
    )


def _performance_section(d: dict) -> str:
    perf = d.get("performance") or {}
    curve = d.get("equity") or []
    if not perf or not perf.get("n_days"):
        return _section(
            "누적 수익률", "아직 기록이 없다 — scripts/snapshot_account.py 가 하루 한 줄 남긴다",
            '<div class="empty">기록 없음</div>',
        )

    tiles = [
        _tile("누적 수익률",
              f'<span class="{_cls(perf.get("total_return"))}">'
              f'{_pct(perf.get("total_return"), 2)}</span>',
              f'{_n(perf.get("start_equity"))} → {_n(perf.get("equity"))}원'),
        _tile("관측", f'{perf.get("n_days", 0)}일',
              f'{perf.get("start_date", "")} ~ {perf.get("end_date", "")}'),
        _tile("실현손익",
              f'<span class="{_cls(perf.get("realized_pnl"))}">'
              f'{_n(perf.get("realized_pnl"))}</span>', "체결 기준 (수수료·세금 반영)"),
        _tile("지불한 비용", _n(perf.get("fee_tax")), "수수료 + 거래세 누적"),
    ]
    if "excess_return" in perf:
        tiles.append(_tile(
            "KOSPI 대비",
            f'<span class="{_cls(perf.get("excess_return"))}">'
            f'{_pct(perf.get("excess_return"), 2)}</span>',
            f'벤치마크 {_pct((perf.get("benchmark") or {}).get("total_return"), 2)}'))

    st = perf.get("strategy") or {}
    if perf.get("reliable") and st:
        tiles += [
            _tile("Sharpe", _n(st.get("sharpe"), 2), "실거래 — 백테스트 1.10"),
            _tile("최대낙폭", _pct(st.get("max_drawdown"), 1), "실거래 — 백테스트 -13.5%"),
        ]

    warn = ""
    if not perf.get("reliable"):
        warn = (f'<p class="note" style="color:var(--warn)">⚠️ 관측 '
                f'{perf.get("n_days", 0)}일 — {perf.get("min_days_for_metrics", 20)}일 '
                f'미만이라 Sharpe·최대낙폭을 숫자로 인용하지 않는다. '
                f'짧은 표본을 연율화한 지표는 해석이 아니라 착시다.</p>')
    if not perf.get("baseline_seeded"):
        warn += ('<p class="note" style="color:var(--warn)">⚠️ 기준선이 없어 첫날 '
                 '진입 수수료가 수익률에서 빠져 있다</p>')

    return _section(
        "누적 수익률", "입출금이 없는 계좌라 총자산 변화가 곧 수익률이다. "
                    "지표는 백테스트와 같은 함수(evaluation/metrics.py)로 계산한다.",
        f'<div class="grid tiles">{"".join(tiles)}</div>{warn}'
        f'{_equity_chart(curve)}',
        sub=f"{perf.get('start_date','')} ~ {perf.get('end_date','')}",
    )


def _attribution_section(d: dict) -> str:
    rows_in = d.get("attribution") or []
    rows = []
    for a in rows_in:
        realized = a["sell_qty"] > 0
        rows.append([
            f'<span class="mono">{escape(a["code"])}</span> {escape(a["name"])}',
            f'<span class="mono">{_n(a["buy_qty"])}</span>',
            f'<span class="mono">{_n(a["buy_amount"])}</span>',
            f'<span class="mono">{_n(a["sell_qty"])}</span>',
            f'<span class="mono">{_n(a["sell_amount"])}</span>',
            f'<span class="mono">{_n(a["fee_tax"])}</span>',
            (f'<span class="mono {_cls(a["pnl_amount"])}">{_n(a["pnl_amount"])}</span>'
             if realized else '<span class="muted">보유중</span>'),
        ])
    return _section(
        "종목별 손익 귀속", "체결 기준 누적이라 부분체결이 있어도 실제 사고판 것만 잡힌다. "
                       "매도가 없는 종목은 실현손익이 아직 없다(평가손익은 '현재 보유' 참고).",
        _table(["종목", "매수수량", "매수금액", "매도수량", "매도금액", "비용", "실현손익"],
               rows, empty="체결 기록 없음"),
    )


def _model_section(d: dict) -> str:
    tr = d.get("training")
    if not tr:
        return ""
    imp = tr.get("feature_importance", {})
    top = sorted(imp.items(), key=lambda kv: -kv[1])[:12]
    peak = top[0][1] if top else 1.0
    bars = "".join(
        f'<tr><td>{escape(k)}</td>'
        f'<td style="width:60%"><div class="bar"><i style="width:{v / peak * 100:.1f}%"></i></div></td>'
        f'<td class="mono">{v:.3f}</td></tr>'
        for k, v in top
    )
    imp_tbl = (f"<table><thead><tr><th>피처</th><th>중요도 (VSN 학습값)</th><th></th></tr>"
               f"</thead><tbody>{bars}</tbody></table>") if top else ""

    facts = _table(
        ["학습", "값"],
        [
            ["파라미터", f'<span class="mono">{_n(tr.get("n_params"))}</span>'],
            ["best epoch", f'<span class="mono">{tr.get("best_epoch", "—")}</span>'],
            ["val pinball", f'<span class="mono">{_n(tr.get("best_val_loss"), 6)}</span>'],
            ["기준선 대비", f'<span class="mono pos">+{_n(tr.get("improvement_vs_baseline_pct"), 2)}%</span>'],
            ["학습 시각", f'<span class="muted">{escape(str(tr.get("timestamp", "")))}</span>'],
        ],
    )
    curve = _sparkline(tr.get("history", []))

    # 화면의 체크포인트와 이 리포트가 같은 학습인지 확인한다.
    # 캐글에서 학습하면 리포트가 로컬에 없어 smoke 리포트로 물러나는데,
    # 그걸 표시하지 않으면 스모크 숫자를 본 학습 성과로 읽게 된다.
    warn = ""
    ckpt = (d.get("checkpoint") or {}).get("name", "")
    report_ckpt = str(tr.get("checkpoint", "")).split("/")[-1]
    if tr.get("_smoke"):
        warn = ("⚠️ 이건 <b>스모크 실행</b> 리포트다 — 실제 학습 리포트가 로컬에 없다"
                "(캐글에서 학습하면 정상). 아래 숫자는 배관 점검용이다.")
    elif ckpt and report_ckpt and ckpt != report_ckpt:
        warn = (f"⚠️ 리포트({escape(report_ckpt)})와 사용 중인 체크포인트"
                f"({escape(ckpt)})가 다르다.")
    warn_html = f'<p class="note" style="color:var(--warn)">{warn}</p>' if warn else ""

    return _section(
        "모델 — 해석가능성",
        "변수선택망(VSN)이 학습으로 정한 피처 중요도다. 사람이 고른 순서가 아니다.",
        warn_html + f'<div class="two"><div>{imp_tbl}</div><div>{facts}{curve}</div></div>',
    )


def _history_section(d: dict) -> str:
    hist = d.get("history") or []
    if not hist:
        return ""
    rows = []
    for h in hist:
        tag = ("<span class='tag dry'>dry-run</span>" if h["dry_run"]
               else f"<span class='tag buy'>전송 {h['n_sent']}건</span>")
        rows.append([
            f'<span class="muted">{escape(str(h["at"]))}</span>',
            escape(str(h["decision_date"])),
            "리밸런싱" if h["rebalancing"] else "점검",
            f'<span class="mono">{h["n_orders"]}건</span>',
            f'<span class="mono">{_n(h["equity"])}</span>',
            tag,
        ])
    return _section(
        "실행 이력", "실행마다 outputs/paper_trading/runs/ 에 남는다. 덮어쓰지 않는다.",
        _table(["실행 시각", "기준일", "종류", "계획 주문", "총자산", "결과"], rows),
    )


def render_html(d: dict[str, Any]) -> str:
    body = "".join([
        _header(d),
        _tiles(d),
        _performance_section(d),
        _orders_section(d),
        _signals_section(d),
        _holdings_section(d),
        _attribution_section(d),
        _backtest_section(d),
        _variants_section(d),
        _model_section(d),
        _history_section(d),
        '<footer>모의투자 전용 — 실전투자 경로는 존재하지 않는다. '
        'KIWOOM_ENV=live 는 코드가 예외로 막는다.</footer>',
    ])
    return (
        "<!doctype html><html lang='ko'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<title>주가예측 자동매매 대시보드</title>"
        f"<style>{_CSS}</style></head><body><div class='wrap'>{body}</div></body></html>"
    )
