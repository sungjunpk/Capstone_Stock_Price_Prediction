"""모의투자 성과 기록 — 발표에서 보여줄 3개월치가 여기 쌓인다.

`runs/*.json` 은 **실행할 때만** 남는다. 누적 수익률 곡선을 그리려면
거래가 없는 날에도 총자산이 찍혀 있어야 하므로 별도 시계열이 필요하다.

    outputs/paper_trading/equity.jsonl      하루 한 줄 — 총자산·현금·주식·벤치마크
    outputs/paper_trading/fills.jsonl       하루 × 종목 — 체결 기준 실현손익
    outputs/paper_trading/performance.json  위에서 계산한 지표 (최신 1개)

두 jsonl 은 **날짜 키 upsert** 다. 같은 날 두 번 돌려도 줄이 하나다.
runs/ 와 달리 시계열이라 덮어쓰기가 맞다(절대 규칙 8은 실험 리포트 얘기다).

⚠️ 지표는 **여기서** 계산한다. 대시보드는 읽기만 한다(`src/webapp/CLAUDE.md` 규칙 2).
   그리고 백테스트와 **같은 함수**(`evaluation.metrics`)를 쓴다 — 그래야
   "백테스트 Sharpe 1.10 vs 실거래 Sharpe X" 비교가 성립한다.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd

from src.data import storage
from src.evaluation import metrics
from src.trading.broker import AccountSnapshot
from src.utils.config import PROJECT_ROOT
from src.utils.logging import get_logger

log = get_logger(__name__)

RECORD_DIR = PROJECT_ROOT / "outputs" / "paper_trading"
EQUITY_PATH = RECORD_DIR / "equity.jsonl"
FILLS_PATH = RECORD_DIR / "fills.jsonl"
HOLDINGS_PATH = RECORD_DIR / "holdings.jsonl"
PERFORMANCE_PATH = RECORD_DIR / "performance.json"
BASELINE_PATH = RECORD_DIR / "baseline.json"

# 벤치마크. 이미 매일 수집되는 파일이라 추가 API 호출이 없다.
KOSPI_CODE = "001"

# 이보다 짧으면 Sharpe·MDD 를 숫자로 인용하지 않는다.
# 며칠짜리 표본으로 연율화한 지표는 해석이 아니라 착시다.
MIN_DAYS_FOR_METRICS = 20


# ---------------------------------------------------------------- jsonl 입출력
def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            log.warning("%s: 깨진 줄을 건너뛴다", path.name)
    return rows


def upsert_jsonl(path: Path, new_rows: list[dict], *, key: tuple[str, ...]) -> int:
    """key 가 같은 줄은 새 것으로 교체. 날짜순 정렬해 다시 쓴다."""
    if not new_rows:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    merged = {tuple(r[k] for k in key): r for r in read_jsonl(path)}
    merged.update({tuple(r[k] for k in key): r for r in new_rows})
    ordered = sorted(merged.values(), key=lambda r: tuple(str(r[k]) for k in key))
    return _write_jsonl(path, ordered)


def _write_jsonl(path: Path, rows: list[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
    )
    return len(rows)


# ---------------------------------------------------------------- 기록
def kospi_close(on: date) -> float | None:
    """벤치마크 종가. 없으면 None — 벤치마크 없다고 기록을 멈추지 않는다."""
    df = storage.read_parquet(storage.raw_path("index_daily", KOSPI_CODE))
    if df is None or df.empty or "close" not in df:
        return None
    d = pd.to_datetime(df["date"]).dt.date
    hit = df[d == on]
    if hit.empty:
        return None
    return float(hit["close"].iloc[-1])


def snapshot_row(account: AccountSnapshot, on: date) -> dict:
    """하루치 계좌 상태 한 줄."""
    return {
        "date": on.isoformat(),
        "equity": round(account.equity, 0),
        "deposit": round(account.deposit, 0),
        "orderable": round(account.cash, 0),
        "stock_eval": round(account.total_eval, 0),
        "n_holdings": len(account.holdings),
        "kospi": kospi_close(on),
    }


def record_equity(account: AccountSnapshot, on: date) -> int:
    return upsert_jsonl(EQUITY_PATH, [snapshot_row(account, on)], key=("date",))


# 현금도 한 행으로 남긴다. 종목 비중과 합이 1 이 되는지 파일 안에서 바로 검증되고,
# 스택 영역 차트가 pivot 한 번으로 끝난다.
CASH_CODE = "CASH"


def holdings_rows(account: AccountSnapshot, on: date) -> list[dict]:
    """종목별 보유 상태 — 하루 × 종목. **발표용 비중 시계열이 여기 쌓인다.**

    비중은 **총자산(NAV) 기준**이다:  w_i = 평가금액 / 총자산.
    현금 비중 c = 1 - Σw_i 를 CASH 행으로 같이 남기므로 Σw + c = 1 이 성립한다.
    (Boyd et al., *Markowitz Portfolio Construction at Seventy*, 2024, §2.1 —
     w 는 총자산 대비 비율, c 는 현금 비중, 제약이 1ᵀw + c = 1)

    ⚠️ 현금은 **잔차(총자산 - 주식평가합)** 다. 주문가능금액이 아니다.
       D+2 결제 때문에 둘이 다르다 — 실측 2026-08-27: 주문가능 9,672,457 vs
       잔차 9,174,777. 주문가능금액을 쓰면 비중 합이 1 을 넘는다.
       주문가능금액은 equity.jsonl 의 `orderable` 에 따로 있다.
    """
    eq = account.equity
    if eq <= 0:
        log.warning("총자산이 %s 다 — 비중을 계산할 수 없어 %s 기록을 건너뛴다", eq, on)
        return []

    rows = [
        {
            "date": on.isoformat(),
            "code": h.code,
            "name": h.name,
            "quantity": h.quantity,
            "avg_price": round(h.avg_price, 0),
            "current_price": round(h.current_price, 0),
            "eval_amount": round(h.eval_amount, 0),
            "weight": round(h.eval_amount / eq, 6),
            "pnl_rate": round(h.pnl_rate, 4),
            "pnl_amount": round(h.pnl_amount, 0),
        }
        for h in sorted(account.holdings.values(), key=lambda h: -h.eval_amount)
    ]
    cash = eq - sum(h.eval_amount for h in account.holdings.values())
    rows.append({
        "date": on.isoformat(),
        "code": CASH_CODE,
        "name": "현금",
        "quantity": None,
        "avg_price": None,
        "current_price": None,
        "eval_amount": round(cash, 0),
        "weight": round(cash / eq, 6),
        "pnl_rate": 0.0,
        "pnl_amount": 0.0,
    })
    return rows


def record_holdings(account: AccountSnapshot, on: date) -> int:
    """holdings.jsonl 에 그날 보유 상태를 남긴다.

    같은 날 다시 돌리면 **그 날짜 행을 통째로 교체**한다.
    (date, code) upsert 로는 안 된다 — 그 사이 매도된 종목의 행이 그대로 남아
    비중 합이 1 을 넘어버린다.
    """
    rows = holdings_rows(account, on)
    if not rows:
        return 0
    day = on.isoformat()
    kept = [r for r in read_jsonl(HOLDINGS_PATH) if r.get("date") != day]
    merged = sorted(kept + rows,
                    key=lambda r: (str(r["date"]), -float(r.get("weight") or 0.0)))
    return _write_jsonl(HOLDINGS_PATH, merged)


def record_fills(rows: list[dict], on: date) -> int:
    """당일매매일지 → fills.jsonl. 거래가 없으면 아무것도 쓰지 않는다."""
    traded = [
        {"date": on.isoformat(), **r}
        for r in rows
        if r.get("buy_qty") or r.get("sell_qty")
    ]
    return upsert_jsonl(FILLS_PATH, traded, key=("date", "code"))


def save_baseline(amount: float, on: date) -> Path:
    """계좌 개시 잔고. **첫 진입 수수료를 수익률에 포함시키기 위해 필요하다.**

    첫 스냅샷은 이미 매매를 끝낸 뒤라, 그걸 시작점으로 삼으면 최초 진입에 낸
    비용이 수익률에서 통째로 사라진다(실측 317,358원 = 0.35%). 3개월 기록에서
    리밸런싱이 열두 번이면 비용을 8%쯤 과소평가하게 된다.

    한 번만 심는다. 이미 있으면 덮어쓰지 않는다 — 기준선이 바뀌면 과거 수익률이
    소급해서 달라진다.
    """
    if BASELINE_PATH.exists():
        log.info("기준선이 이미 있다 — 건드리지 않는다: %s", BASELINE_PATH.name)
        return BASELINE_PATH
    BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "date": on.isoformat(),
        "equity": float(amount),
        # 벤치마크도 같은 날부터 시작해야 초과수익 비교의 구간이 맞는다
        "kospi": kospi_close(on),
    }
    BASELINE_PATH.write_text(json.dumps(row, ensure_ascii=False, indent=2),
                             encoding="utf-8")
    log.info("기준선 기록: %s %s원", on, f"{amount:,.0f}")
    return BASELINE_PATH


def read_baseline() -> dict | None:
    if not BASELINE_PATH.exists():
        return None
    try:
        return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        log.warning("기준선 파일이 깨졌다 — 첫 스냅샷을 시작점으로 쓴다")
        return None


# ---------------------------------------------------------------- 지표
def _returns(equity: list[dict]) -> pd.Series:
    s = pd.Series(
        [float(r["equity"]) for r in equity],
        index=pd.to_datetime([r["date"] for r in equity]),
    ).sort_index()
    return s.pct_change().dropna()


def _benchmark_returns(equity: list[dict]) -> pd.Series:
    have = [r for r in equity if r.get("kospi")]
    if len(have) < 2:
        return pd.Series(dtype=float)
    s = pd.Series(
        [float(r["kospi"]) for r in have],
        index=pd.to_datetime([r["date"] for r in have]),
    ).sort_index()
    return s.pct_change().dropna()


def compute_performance() -> dict:
    """equity.jsonl + fills.jsonl → 지표 묶음.

    **관측일수가 짧으면 지표를 내보내되 `reliable=False` 로 표시한다.**
    지우지 않는 이유는 화면에서 '아직 못 믿는다'를 보여주기 위해서다.
    """
    equity = sorted(read_jsonl(EQUITY_PATH), key=lambda r: r["date"])
    fills = read_jsonl(FILLS_PATH)

    if not equity:
        return {"n_days": 0, "reliable": False, "note": "기록이 없다"}

    # 기준선이 있으면 맨 앞에 붙인다 — 첫 진입 비용을 수익률에 포함시키기 위해서다.
    #
    # ⚠️ 기준선 날짜를 옮기지 않는다. 한 번 옮겨봤다가 벤치마크(KOSPI)는 옛 날짜의
    #    값을 그대로 들고 가서 곡선이 어긋났다. 날짜는 데이터의 일부다.
    #    기준선은 **첫 스냅샷보다 이른 거래일**에 심어야 한다.
    base = read_baseline()
    start_from_baseline = bool(base) and base["date"] < equity[0]["date"]
    if start_from_baseline:
        equity = [base] + equity
    elif base:
        log.warning("기준선 날짜(%s)가 첫 스냅샷(%s)보다 이르지 않다 — "
                    "누적수익률에만 쓰고 곡선에는 넣지 않는다",
                    base["date"], equity[0]["date"])

    # 누적수익률의 분모는 기준선이 있으면 언제나 기준선이다(첫날 비용 포함).
    start = float(base["equity"]) if base else float(equity[0]["equity"])
    last = float(equity[-1]["equity"])
    r = _returns(equity)
    br = _benchmark_returns(equity)

    out: dict = {
        "start_date": equity[0]["date"],
        "end_date": equity[-1]["date"],
        "n_days": len(equity),
        "start_equity": start,
        "baseline_seeded": bool(base),   # False 면 첫날 진입 비용이 수익률에서 빠져 있다
        "baseline_in_curve": start_from_baseline,
        "equity": last,
        # 입출금이 없는 계좌라 이게 곧 누적 수익률이다.
        "total_return": round(last / start - 1.0, 5) if start > 0 else 0.0,
        "reliable": len(r) >= MIN_DAYS_FOR_METRICS,
        "min_days_for_metrics": MIN_DAYS_FOR_METRICS,
    }
    if len(r) >= 2:
        # 백테스트 리포트와 같은 키가 나오도록 같은 함수를 쓴다
        out["strategy"] = metrics.summarize(r)
    if len(br) >= 2:
        out["benchmark"] = metrics.summarize(br)
        out["excess_return"] = round(
            out["total_return"] - float(out["benchmark"]["total_return"]), 5
        )

    # 병합된 곡선을 여기 담아 둔다. 대시보드가 기준선 병합을 다시 구현하면
    # 화면과 지표가 다른 구간을 보게 된다 — 계산은 한 곳에만 둔다.
    out["curve"] = [
        {"date": e["date"], "equity": float(e["equity"]), "kospi": e.get("kospi")}
        for e in equity
    ]

    realized = sum(float(f.get("pnl_amount", 0.0)) for f in fills)
    out["realized_pnl"] = round(realized, 0)
    out["fee_tax"] = round(sum(float(f.get("fee_tax", 0.0)) for f in fills), 0)
    out["n_fill_rows"] = len(fills)
    return out


def save_performance(perf: dict) -> Path:
    PERFORMANCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    PERFORMANCE_PATH.write_text(
        json.dumps(perf, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return PERFORMANCE_PATH
