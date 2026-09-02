#!/usr/bin/env bash
# 세션이 열릴 때 "지금 어디까지 왔나"를 한 번 찍는다.
# 매번 손으로 물어보게 하면 결국 안 물어보고 낡은 데이터로 판단하게 된다.
set -uo pipefail
root="${CLAUDE_PROJECT_DIR:-$PWD}"
[ -x "$root/.venv/bin/python" ] || exit 0

txt=$("$root/.venv/bin/python" - "$root" <<'PY' 2>/dev/null
import json, sys, subprocess
from pathlib import Path
root = Path(sys.argv[1]); out = []

# 패널 신선도 — 낡으면 리밸런싱이 막힌다
try:
    import pyarrow.parquet as pq
    t = pq.read_table(root/"data/processed/panel.parquet", columns=["date"])
    out.append(f"패널 최종일 {max(t.column('date').to_pylist())}")
except Exception as e:
    out.append(f"패널 확인 실패 ({type(e).__name__})")

# 실거래
try:
    p = json.loads((root/"outputs/paper_trading/performance.json").read_text())
    flag = "" if p.get("reliable") else "  ⚠️ 관측 20일 미만 — Sharpe·MDD 인용 금지"
    out.append(f"실거래 {p['n_days']}일차 · 누적 {p['total_return']:+.2%} · "
               f"평가 {p['end_date']}{flag}")
except Exception:
    pass

# 다음 리밸런싱
try:
    import yaml
    n = int(yaml.safe_load((root/"configs/config.yaml").read_text())["backtest"]["rebalance_days"])
    s = json.loads((root/"outputs/paper_trading/state.json").read_text())
    out.append(f"마지막 리밸런싱 {s['last_rebalance']} · 주기 {n}거래일")
except Exception:
    pass

# 어제 자동화가 끝까지 돌았나
try:
    logs = sorted((root/"outputs/logs/daily").glob("20*.log"))
    if logs:
        tail = logs[-1].read_text(errors="ignore").strip().splitlines()[-1]
        ok = "완료" in tail
        out.append(f"최근 수집 {logs[-1].stem} {'정상' if ok else '⚠️ 끝까지 안 돌았다'}")
except Exception:
    pass

# git
try:
    d = subprocess.run(["git","-C",str(root),"status","--porcelain"],
                       capture_output=True, text=True).stdout.strip()
    a = subprocess.run(["git","-C",str(root),"rev-list","--count","@{u}..HEAD"],
                       capture_output=True, text=True).stdout.strip() or "?"
    out.append(f"git 미커밋 {len(d.splitlines())}개 · 미푸시 {a}커밋")
except Exception:
    pass

print(" | ".join(out))
PY
)
[ -z "$txt" ] && exit 0
jq -Rn --arg s "$txt" \
  '{hookSpecificOutput:{hookEventName:"SessionStart",additionalContext:("[프로젝트 현황] "+$s)}}'
