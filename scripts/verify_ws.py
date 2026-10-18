#!/usr/bin/env python
"""WebSocket 실시간 검증 — 접속·등록·수신이 되는지, 필드명이 무엇인지 확인한다.

이 저장소는 WebSocket 을 한 번도 호출해본 적이 없다. 엔드포인트도 메시지 모양도
**전부 문서 기준**이라, 수신 계층(`ws_feed.py`)을 쓰기 전에 실호출로 확정해야 한다.
확인 결과는 `docs/KIWOOM_VERIFY.md` 에 적는다.

    python scripts/verify_ws.py                      # 그리드 종목, 60초 수신
    python scripts/verify_ws.py --seconds 300        # 더 오래
    python scripts/verify_ws.py --codes 005930 000660
    python scripts/verify_ws.py --types 00           # 주문체결 통보만

**읽기 전용이다 — 주문을 내지 않는다.** 체결 통보(타입 00)를 받으려면 장중에
다른 창에서 1주 주문을 내야 한다 (`scripts/verify_trading_trs.py --order ...`).
장 마감 후에는 틱(0B)이 흐르지 않는다 — 접속·로그인·등록 응답까지만 확인된다.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import certifi  # noqa: E402
import websocket  # noqa: E402

from src.data.kiwoom.client import KiwoomClient  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

log = get_logger(__name__)

# ⚠️ UNVERIFIED — 문서 기준. 이 스크립트가 확인하려는 대상 자체다.
#    REST base_url 과 호스트는 같고 포트·경로만 다르다고 되어 있다.
WS_PORT = 10000
WS_PATH = "/api/dostk/websocket"

# 등록 타입. 0B=주식체결(틱), 00=주문체결 통보.
DEFAULT_TYPES = ("0B", "00")


def ws_url(base_url: str) -> str:
    host = base_url.replace("https://", "").replace("http://", "").rstrip("/")
    return f"wss://{host}:{WS_PORT}{WS_PATH}"


def default_codes(limit: int) -> list[str]:
    """유니버스 앞쪽 몇 개. 검증 목적이라 어느 종목이든 상관없다 —
    종목 선별은 여기서 하지 않는다(규칙 7: 선별은 signal 하나다)."""
    uni = load_config()["data"]["universe"]
    return [str(row["code"]) for row in uni[:limit]]


def main() -> int:
    ap = argparse.ArgumentParser(description="키움 WebSocket 실시간 검증 (읽기 전용)")
    ap.add_argument("--codes", nargs="*", help="등록할 종목코드 (기본: 설정 유니버스 5개)")
    ap.add_argument("--types", nargs="*", default=list(DEFAULT_TYPES), help="등록 타입")
    ap.add_argument("--seconds", type=float, default=60.0, help="수신 시간")
    ap.add_argument("--url", help="WS 엔드포인트 직접 지정")
    args = ap.parse_args()

    codes = args.codes or default_codes(5)

    client = KiwoomClient()
    token = client._auth_header().removeprefix("Bearer ")
    url = args.url or ws_url(client.settings.base_url)

    log.info("접속 %s", url)
    log.info("등록 예정 종목 %s · 타입 %s", codes, args.types)

    frames: list[dict] = []          # 받은 것 전부. 이게 이 스크립트의 산출물이다.
    seen_types: dict[str, int] = {}  # 타입별 수신 건수

    # requests 는 certifi 번들을 쓰는데 websocket-client 는 시스템 저장소를 쓴다.
    # 같은 인증서(Sectigo EV)인데도 체인 검증이 깨져서 번들을 명시한다(2026-10-09 실측).
    ws = websocket.create_connection(
        url, timeout=10, sslopt={"ca_certs": certifi.where()}
    )
    deadline = time.time() + args.seconds
    try:
        # ⚠️ LOGIN 응답을 받기 **전에** REG 를 보내면 그 전문은 버려진다
        #    (return_code 100013, 2026-10-09 실측). 순서가 아니라 왕복이 필요하다.
        ws.send(json.dumps({"trnm": "LOGIN", "token": token}))
        ack = json.loads(ws.recv())
        frames.append({"at": datetime.now().isoformat(timespec="seconds"), "msg": ack})
        log.info("LOGIN: %s", json.dumps(ack, ensure_ascii=False))
        if ack.get("return_code") != 0:
            log.error("로그인 실패 — 등록하지 않는다")
            return 1

        ws.send(json.dumps({
            "trnm": "REG", "grp_no": "1", "refresh": "1",
            "data": [{"item": codes, "type": list(args.types)}],
        }))

        while time.time() < deadline:
            ws.settimeout(max(1.0, deadline - time.time()))
            try:
                raw = ws.recv()
            except websocket.WebSocketTimeoutException:
                break
            if not raw:
                continue

            msg = json.loads(raw)
            trnm = msg.get("trnm")

            # PING 은 받은 내용을 그대로 돌려줘야 연결이 유지된다.
            if trnm == "PING":
                ws.send(raw)
                log.info("PING ↔ PONG")
                continue

            frames.append({"at": datetime.now().isoformat(timespec="seconds"), "msg": msg})
            if trnm == "REAL":
                for item in msg.get("data") or []:
                    t = str(item.get("type"))
                    seen_types[t] = seen_types.get(t, 0) + 1
                    if seen_types[t] == 1:  # 타입별 첫 건만 원문을 보여준다
                        log.info("첫 %s 프레임:\n%s", t, json.dumps(item, ensure_ascii=False, indent=2))
            else:
                log.info("%s: %s", trnm, json.dumps(msg, ensure_ascii=False))
    finally:
        ws.close()

    out = Path("outputs/reports") / f"ws_probe_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "url": url, "codes": codes, "types": args.types,
        "seconds": args.seconds, "counts": seen_types, "frames": frames,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    log.info("수신 %d건 %s → %s", len(frames), seen_types or "(없음)", out)
    if not frames:
        log.warning("한 건도 못 받았다 — 로그인 응답조차 없으면 엔드포인트부터 의심한다")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
