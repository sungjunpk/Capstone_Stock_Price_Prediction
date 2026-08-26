"""대시보드가 그릴 데이터를 outputs/ 에서 모은다.

**여기서 계산하지 않는다.** 지표는 백테스트가, 예측은 추론이, 주문은 실행기가 이미
남긴 값이다. 대시보드가 자기 식으로 다시 계산하면 화면과 리포트가 서로 다른 말을 한다.
읽고, 고르고, 정리만 한다.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from src.utils.config import PROJECT_ROOT

REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
RUNS_DIR = PROJECT_ROOT / "outputs" / "paper_trading" / "runs"
STATE_PATH = PROJECT_ROOT / "outputs" / "paper_trading" / "state.json"
CKPT_DIR = PROJECT_ROOT / "outputs" / "checkpoints"
PANEL_PATH = PROJECT_ROOT / "data" / "processed" / "panel.parquet"
_PT = PROJECT_ROOT / "outputs" / "paper_trading"
EQUITY_PATH = _PT / "equity.jsonl"
FILLS_PATH = _PT / "fills.jsonl"
PERFORMANCE_PATH = _PT / "performance.json"


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _read_jsonl(path: Path) -> list[dict]:
    """날짜순 정렬된 줄들. 없거나 깨져도 빈 목록 — 화면은 죽지 않아야 한다."""
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
            continue
    return sorted(rows, key=lambda r: str(r.get("date", "")))


def _newest(paths: list[Path]) -> Path | None:
    return max(paths, key=lambda p: p.stat().st_mtime) if paths else None


def latest_run() -> dict | None:
    """가장 최근 모의투자 실행 기록."""
    if not RUNS_DIR.exists():
        return None
    newest = _newest(list(RUNS_DIR.glob("run_*.json")))
    if newest is None:
        return None
    data = _read_json(newest)
    if data:
        data["_file"] = newest.name
        data["_mtime"] = datetime.fromtimestamp(newest.stat().st_mtime).isoformat(
            timespec="seconds"
        )
    return data


def run_history(limit: int = 20) -> list[dict]:
    """최근 실행들의 요약. 무엇을 언제 했는지 훑는 용도다."""
    if not RUNS_DIR.exists():
        return []
    files = sorted(RUNS_DIR.glob("run_*.json"), key=lambda p: p.stat().st_mtime,
                   reverse=True)[:limit]
    out = []
    for f in files:
        d = _read_json(f) or {}
        plan = d.get("plan", {})
        sent = d.get("orders_sent", [])
        out.append({
            "file": f.name,
            "at": plan.get("generated_at", ""),
            "decision_date": plan.get("decision_date", ""),
            "dry_run": bool(d.get("dry_run", True)),
            "rebalancing": bool(plan.get("rebalancing", False)),
            "n_orders": len(plan.get("orders", [])),
            "n_sent": sum(1 for r in sent if not r.get("error") and not r.get("dry_run")),
            "equity": plan.get("equity", 0),
        })
    return out


def latest_backtests() -> dict[str, Any]:
    """가장 최근 백테스트 세션. `--compare` 로 만든 변형들을 함께 묶는다.

    변형 비교는 같은 시각에 여러 파일로 떨어지므로, **가장 최근 파일과 같은
    체크포인트·같은 타임스탬프 근처**의 것들을 한 세션으로 본다.
    """
    if not REPORTS_DIR.exists():
        return {}
    files = list(REPORTS_DIR.glob("backtest_*.json"))
    newest = _newest(files)
    if newest is None:
        return {}

    base = _read_json(newest) or {}
    ckpt = base.get("checkpoint")
    newest_at = newest.stat().st_mtime

    variants = []
    for f in files:
        if abs(f.stat().st_mtime - newest_at) > 600:      # 10분 안쪽이면 같은 세션
            continue
        d = _read_json(f)
        if not d or d.get("checkpoint") != ckpt:
            continue
        variants.append({"file": f.name, **d})

    variants.sort(key=lambda d: d.get("timestamp", ""))
    # ⚠️ '변형 없음' 중에서 **가장 최근 것**을 고른다.
    #    첫 번째를 고르면 같은 세션에 남은 옛 실행이 대표가 되어, 화면에 0 이 뜬다(실측).
    plain = [v for v in variants if not v.get("variant")]
    main = plain[-1] if plain else variants[-1]
    return {"main": main, "variants": variants}


def latest_training() -> dict | None:
    """가장 최근 학습 리포트 — 피처 중요도와 학습 곡선이 여기 있다."""
    if not REPORTS_DIR.exists():
        return None
    files = [
        p for p in REPORTS_DIR.glob("*.json")
        if not p.name.startswith(("backtest_", "sweep_"))
    ]
    real = [p for p in files if "smoke" not in p.name]
    # 실제 학습 리포트가 없으면 smoke 리포트로 물러나되, **그 사실을 표시한다.**
    # 캐글에서 학습하면 리포트가 로컬에 없는 게 정상이라 이 상황이 실제로 자주 나온다.
    newest = _newest(real or files)
    if newest is None:
        return None
    data = _read_json(newest)
    if data is not None:
        data["_file"] = newest.name
        data["_smoke"] = bool(data.get("smoke")) or "smoke" in newest.name
    return data


def checkpoint_info() -> dict | None:
    if not CKPT_DIR.exists():
        return None
    cands = [p for p in CKPT_DIR.glob("phase1_*.pt") if "smoke" not in p.name]
    newest = _newest(cands)
    if newest is None:
        return None
    return {
        "name": newest.name,
        "size_mb": round(newest.stat().st_size / 1e6, 2),
        "modified": datetime.fromtimestamp(newest.stat().st_mtime).isoformat(
            timespec="seconds"
        ),
    }


def data_status() -> dict:
    """패널이 언제까지 채워져 있는가. 낡은 데이터로 낸 주문을 화면에서 바로 알아채야 한다."""
    if not PANEL_PATH.exists():
        return {"available": False}
    try:
        import pandas as pd

        # date 컬럼만 읽는다 — 패널 전체는 수십 MB 다
        dates = pd.read_parquet(PANEL_PATH, columns=["date"])["date"]
        codes = pd.read_parquet(PANEL_PATH, columns=["code"])["code"]
        return {
            "available": True,
            "last_date": str(pd.to_datetime(dates).max().date()),
            "first_date": str(pd.to_datetime(dates).min().date()),
            "rows": int(len(dates)),
            "codes": int(codes.nunique()),
        }
    except Exception as exc:  # noqa: BLE001 — 대시보드가 데이터 문제로 죽으면 안 된다
        return {"available": False, "error": str(exc)}


def performance() -> dict | None:
    """모의투자 누적 성과. `src/trading/record.py` 가 계산해 남긴 값이다."""
    return _read_json(PERFORMANCE_PATH)


def equity_history() -> list[dict]:
    """일별 총자산 + 벤치마크 곡선.

    `record.compute_performance()` 가 기준선까지 병합해 `performance.json` 에
    넣어둔 것을 그대로 읽는다. 여기서 다시 병합하지 않는다 —
    두 번 구현하면 화면의 곡선과 지표가 다른 구간을 보게 된다(실제로 그랬다).
    """
    perf = _read_json(PERFORMANCE_PATH) or {}
    return list(perf.get("curve") or [])


def attribution(limit: int = 30) -> list[dict]:
    """종목별 누적 손익 — 모델이 무엇을 사서 얼마 벌었나.

    합산만 한다(계산이 아니라 집계다). 체결 기준이라 부분체결이 있어도
    실제 사고판 것만 잡힌다.
    """
    agg: dict[str, dict] = {}
    for f in _read_jsonl(FILLS_PATH):
        a = agg.setdefault(f["code"], {
            "code": f["code"], "name": f.get("name", ""), "days": 0,
            "buy_qty": 0, "sell_qty": 0, "buy_amount": 0.0, "sell_amount": 0.0,
            "pnl_amount": 0.0, "fee_tax": 0.0,
        })
        a["days"] += 1
        a["name"] = f.get("name") or a["name"]
        for k in ("buy_qty", "sell_qty"):
            a[k] += int(f.get(k, 0) or 0)
        for k in ("buy_amount", "sell_amount", "pnl_amount", "fee_tax"):
            a[k] += float(f.get(k, 0.0) or 0.0)
    return sorted(agg.values(), key=lambda a: -a["pnl_amount"])[:limit]


def collect_all() -> dict:
    """대시보드 한 장에 필요한 전부."""
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run": latest_run(),
        "history": run_history(),
        "state": _read_json(STATE_PATH),
        "performance": performance(),
        "equity": equity_history(),
        "attribution": attribution(),
        "backtest": latest_backtests(),
        "training": latest_training(),
        "checkpoint": checkpoint_info(),
        "data": data_status(),
    }
