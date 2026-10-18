#!/usr/bin/env python
"""메뉴바 앱 — 그리드 사다리와 계좌 상태를 상단바에서 본다.

    python scripts/menubar.py

**읽기 전용이다** (절대 규칙 9). 이 앱에서 주문이 나가는 경로는 없다 —
'재배치 미리보기'는 `paper_trade.py --grid` 를 **`--execute` 없이** 부르고,
결과를 로그로 보여줄 뿐이다. 실주문은 지금도 터미널 한 곳에서만 나간다.

데이터는 API 가 아니라 `outputs/` 에서 읽는다. 장 마감 뒤에도 돌고,
호출 한도를 먹지 않고, 키움이 죽어도 화면은 산다.
"""

from __future__ import annotations

import contextlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import rumps
from Foundation import NSUserDefaults

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import PROJECT_ROOT  # noqa: E402
from src.webapp.collect import data_status, holdings, latest_run, performance  # noqa: E402

STATE_PATH = PROJECT_ROOT / "outputs" / "paper_trading" / "state.json"
LOG_DIR = PROJECT_ROOT / "outputs" / "logs" / "daily"
REFRESH_SEC = 60


def _won(n: float) -> str:
    return f"{'+' if n >= 0 else '−'}{abs(n):,.0f}원"


def _age(iso: str | None) -> str:
    if not iso:
        return "기록 없음"
    try:
        d = datetime.fromisoformat(str(iso)[:10]).date()
    except ValueError:
        return str(iso)
    n = (datetime.now().date() - d).days
    return f"{d} ({'오늘' if n == 0 else f'{n}일 전'})"


class GridBar(rumps.App):
    def __init__(self) -> None:
        # 노치 맥북은 새 아이콘을 맨 왼쪽(노치 밑)에 두고 숨긴다 — 시계 쪽에서 시작하게 한다.
        # registerDefaults 라 사용자가 ⌘-드래그로 옮긴 자리가 있으면 그게 이긴다.
        NSUserDefaults.standardUserDefaults().registerDefaults_(
            {"NSStatusItem Preferred Position Item-0": 300})
        super().__init__("🪜", quit_button=None)
        self.menu = [
            "상태", "수익", "데이터", None,
            rumps.MenuItem("사다리"), rumps.MenuItem("보유"), None,
            rumps.MenuItem("대시보드 열기", callback=self.open_dashboard),
            rumps.MenuItem("재배치 미리보기 (dry-run)", callback=self.preview),
            rumps.MenuItem("로그 폴더 열기", callback=self.open_logs), None,
            rumps.MenuItem("새로고침", callback=lambda _: self.refresh()),
            rumps.MenuItem("종료", callback=rumps.quit_application),
        ]
        self.refresh()

    @rumps.timer(REFRESH_SEC)
    def _tick(self, _) -> None:
        self.refresh()

    # ── 화면 ────────────────────────────────────────────────
    def refresh(self) -> None:
        try:
            self._render()
        except Exception as exc:  # noqa: BLE001 — 상단바가 데이터 문제로 죽으면 안 된다
            self.title = "🪜 ⚠️"
            self.menu["상태"].title = f"읽기 실패: {exc}"

    def _render(self) -> None:
        perf = performance() or {}
        ret = perf.get("total_return")
        eq = perf.get("equity")
        self.title = f"🪜 {ret * 100:+.2f}%" if ret is not None else "🪜"

        run = latest_run() or {}
        mode = "dry-run" if run.get("dry_run") else "실주문"
        self.menu["상태"].title = (
            f"마지막 실행 {_age(run.get('_mtime'))} · {mode}"
            if run else "마지막 실행: 기록 없음"
        )
        self.menu["수익"].title = (
            f"평가 {eq:,.0f}원 · 누적 {ret * 100:+.2f}% · {perf.get('n_days', 0)}일차"
            if eq is not None else "성과 기록 없음"
        )

        ds = data_status()
        last = ds.get("last_date")
        stale = "  ⚠️ 낡음" if last and (datetime.now().date() - datetime.fromisoformat(last).date()).days > 3 else ""
        self.menu["데이터"].title = f"패널 {last} · {ds.get('codes', 0)}종목{stale}" if ds.get("available") else "패널 없음"

        # 사다리 — state.json 의 그리드 사이클 (아직 없으면 그렇게 말한다)
        grids = {}
        if STATE_PATH.exists():
            with contextlib.suppress(json.JSONDecodeError):
                grids = json.loads(STATE_PATH.read_text(encoding="utf-8")).get("grids") or {}
        sub = self.menu["사다리"]
        if len(sub):  # rumps: 한 번도 안 채운 서브메뉴를 clear() 하면 터진다
            sub.clear()
        sub.title = f"사다리 {len(grids)}종목" if grids else "사다리 — 아직 없음"
        for code, g in sorted(grids.items()):
            sub.add(rumps.MenuItem(
                f"{code}   중심 {float(g['center']):,.0f}원 · 간격 {float(g['spacing']) * 100:.2f}% · {_age(g.get('date'))}"
            ))

        held = [h for h in (holdings() or []) if h["code"] != "CASH"]
        sub = self.menu["보유"]
        if len(sub):  # rumps: 한 번도 안 채운 서브메뉴를 clear() 하면 터진다
            sub.clear()
        sub.title = f"보유 {len(held)}종목" if held else "보유 없음"
        for h in sorted(held, key=lambda x: -(x.get("pnl_amount") or 0)):
            sub.add(rumps.MenuItem(
                f"{h.get('name') or h['code']}   {_won(h.get('pnl_amount') or 0)}"
                f"  ({(h.get('pnl_rate') or 0) * 100:+.1f}%)"
            ))

    # ── 동작 (전부 읽기 전용) ───────────────────────────────
    def open_dashboard(self, _) -> None:
        """브라우저 대신 창 앱(scripts/desktop.py) — 열 때마다 최신으로 그린다."""
        subprocess.Popen([sys.executable, str(PROJECT_ROOT / "scripts" / "desktop.py")], cwd=PROJECT_ROOT)

    def open_logs(self, _) -> None:
        subprocess.run(["open", str(LOG_DIR)], check=False)

    def preview(self, _) -> None:
        """`--execute` 를 **붙이지 않는다.** 이 앱에서 주문이 나가는 경로는 없다."""
        out = LOG_DIR / "menubar_preview.log"
        out.parent.mkdir(parents=True, exist_ok=True)
        rumps.notification("그리드", "재배치 미리보기 실행 중", "끝나면 알림이 뜬다")
        try:
            r = subprocess.run(
                [sys.executable, str(PROJECT_ROOT / "scripts" / "paper_trade.py"), "--grid"],
                cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=600,
            )
            out.write_text(r.stdout + r.stderr, encoding="utf-8")
            rumps.notification("그리드", "미리보기 완료" if r.returncode == 0 else "미리보기 실패",
                               f"결과: {out.name}")
        except subprocess.TimeoutExpired:
            rumps.notification("그리드", "시간 초과", "10분 안에 안 끝났다")
        subprocess.run(["open", "-a", "TextEdit", str(out)], check=False)
        self.refresh()


if __name__ == "__main__":
    GridBar().run()
