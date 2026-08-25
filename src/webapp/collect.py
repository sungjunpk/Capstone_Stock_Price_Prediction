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


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


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


def collect_all() -> dict:
    """대시보드 한 장에 필요한 전부."""
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run": latest_run(),
        "history": run_history(),
        "state": _read_json(STATE_PATH),
        "backtest": latest_backtests(),
        "training": latest_training(),
        "checkpoint": checkpoint_info(),
        "data": data_status(),
    }
