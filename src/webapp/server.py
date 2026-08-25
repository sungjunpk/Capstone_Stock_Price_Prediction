"""대시보드 서버 — 표준 라이브러리만 쓴다.

**읽기 전용이다.** 화면에서 주문을 낼 수 있게 만들지 않는다.
버튼 하나로 실주문이 나가는 경로를 브라우저에 열어두면, 이 프로젝트에서 제일
되돌릴 수 없는 동작이 제일 누르기 쉬운 곳에 놓인다.
주문은 `scripts/paper_trade.py --execute` 한 곳에서만 나간다.

요청마다 outputs/ 를 다시 읽어 렌더링한다 — 백테스트나 모의투자를 돌린 뒤
새로고침만 하면 최신 상태가 보인다.
"""

from __future__ import annotations

import json
import threading
import webbrowser
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from src.utils.logging import get_logger
from src.webapp.collect import collect_all
from src.webapp.render import render_html

log = get_logger(__name__)


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "CapstoneDashboard/1.0"

    def do_GET(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler 규약
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        if path == "/":
            self._send(render_html(collect_all()).encode("utf-8"), "text/html; charset=utf-8")
        elif path == "/api/state.json":
            payload = json.dumps(collect_all(), ensure_ascii=False, default=str, indent=2)
            self._send(payload.encode("utf-8"), "application/json; charset=utf-8")
        elif path == "/healthz":
            self._send(b"ok", "text/plain; charset=utf-8")
        else:
            self.send_error(404, "not found")

    def _send(self, body: bytes, content_type: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        # 매 요청 새로 만드니 캐시를 남기면 오래된 화면을 본다
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args) -> None:
        log.debug("%s %s", self.address_string(), fmt % args)


def serve(host: str = "127.0.0.1", port: int = 8765, *, open_browser: bool = True) -> None:
    """로컬 전용. 기본 바인딩이 127.0.0.1 인 건 의도다 — 계좌 상태가 보이는 화면이다."""
    httpd = ThreadingHTTPServer((host, port), partial(DashboardHandler))
    url = f"http://{host}:{port}/"
    log.info("대시보드 시작 — %s (Ctrl+C 로 종료)", url)
    print(f"\n  대시보드: {url}\n  종료: Ctrl+C\n")

    if open_browser:
        threading.Timer(0.6, webbrowser.open, args=(url,)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n종료.")
    finally:
        httpd.server_close()
