#!/usr/bin/env python
"""결과 대시보드 — 모의투자·백테스트·모델을 한 화면에서 본다.

    python scripts/dashboard.py              # 서버 실행 후 브라우저 열기
    python scripts/dashboard.py --port 9000
    python scripts/dashboard.py --build      # 정적 HTML 한 장으로 저장(서버 없이 열림)

읽기 전용이다. 화면에서 주문을 낼 수 없다 —
주문은 `scripts/paper_trade.py --execute` 한 곳에서만 나간다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import PROJECT_ROOT  # noqa: E402
from src.utils.logging import get_logger, setup_logging  # noqa: E402
from src.webapp.collect import collect_all  # noqa: E402
from src.webapp.render import render_html  # noqa: E402
from src.webapp.server import serve  # noqa: E402

log = get_logger("dashboard")
DEFAULT_OUT = PROJECT_ROOT / "outputs" / "dashboard" / "index.html"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1",
                    help="기본 127.0.0.1 — 계좌 상태가 보이는 화면이라 외부 노출하지 않는다")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--build", nargs="?", const=str(DEFAULT_OUT), metavar="PATH",
                    help="서버 대신 정적 HTML 파일로 저장한다")
    args = ap.parse_args()

    setup_logging(run_name="dashboard")

    if args.build:
        out = Path(args.build)
        if not out.is_absolute():
            out = PROJECT_ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(render_html(collect_all()), encoding="utf-8")
        print(f"저장: {out}")
        return 0

    serve(args.host, args.port, open_browser=not args.no_browser)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
