#!/usr/bin/env python
"""매매 TR 검증 — 주문을 내기 전에 응답 스키마가 정의와 맞는지 확인한다.

수집 TR 은 필드명이 틀리면 컬럼이 None 으로 비어서 바로 눈에 띈다.
**매매 TR 은 그렇지 않다.** 잔고 필드명이 하나 틀리면 보유수량이 0 으로 읽히고,
그 상태로 "미보유 → 신규 매수" 판단이 나간다. 조용히 틀리는 쪽이라 먼저 검증한다.

    python scripts/verify_trading_trs.py            # 조회계 TR 만 (읽기 전용, 안전)
    python scripts/verify_trading_trs.py --order 005930   # 1주 매수 주문까지 검증

⚠️ `--order` 는 **모의투자 계좌에 실제 주문을 낸다.** 1주짜리지만 체결된다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.kiwoom import endpoints as ep  # noqa: E402
from src.data.kiwoom.client import KiwoomAPIError  # noqa: E402
from src.trading.broker import BUY, PaperBroker  # noqa: E402
from src.utils.logging import get_logger, setup_logging  # noqa: E402

log = get_logger("verify_trading")


def _check_schema(name: str, body: dict, spec: ep.TRSpec, *, summary_fields=None) -> bool:
    """정의한 응답키가 실제 응답에 있는지 대조한다."""
    records = body if not spec.list_key else (body.get(spec.list_key) or [])
    if not spec.list_key:
        records = [body]

    print(f"\n--- {name} ({spec.api_id}) ---")
    print(f"  return_code={body.get('return_code')} msg={body.get('return_msg')!r}")

    ok = True
    if spec.list_key and not records:
        print(f"  ⚠️ list_key '{spec.list_key}' 가 비었다 "
              f"(응답 키: {sorted(body)[:12]})")
        ok = False

    sample = records[0] if records else {}
    for out_col, (src_key, kind) in spec.schema.items():
        mark = "✅" if src_key in sample else "❌"
        if src_key not in sample:
            ok = False
        print(f"  {mark} {out_col:<16} ← {src_key:<16} {kind:<10} "
              f"{sample.get(src_key, '(없음)')!r}")

    for out_col, (src_key, kind) in (summary_fields or {}).items():
        mark = "✅" if src_key in body else "❌"
        if src_key not in body:
            ok = False
        print(f"  {mark} [요약] {out_col:<10} ← {src_key:<20} {kind:<10} "
              f"{body.get(src_key, '(없음)')!r}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", metavar="CODE",
                    help="이 종목으로 1주 매수 주문까지 검증한다 (실제 주문이 나간다)")
    ap.add_argument("--quote", default="005930", help="현재가 검증에 쓸 종목")
    ap.add_argument("--dump", action="store_true", help="원본 응답 전체를 출력")
    args = ap.parse_args()

    setup_logging(run_name="verify_trading")
    results: dict[str, bool] = {}

    with PaperBroker() as broker:
        client = broker.client

        # --- 1) 예수금
        body, _ = client.request(ep.DEPOSIT, {"qry_tp": "2"})
        results["deposit"] = _check_schema("예수금", body, ep.DEPOSIT)
        if args.dump:
            print(json.dumps(body, ensure_ascii=False, indent=2)[:2000])

        # --- 2) 계좌평가잔고
        body, _ = client.request(
            ep.ACCOUNT_BALANCE, {"qry_tp": "1", "dmst_stex_tp": "KRX"}
        )
        results["account_balance"] = _check_schema(
            "계좌평가잔고", body, ep.ACCOUNT_BALANCE,
            summary_fields=ep.ACCOUNT_SUMMARY_FIELDS,
        )
        if args.dump:
            print(json.dumps(body, ensure_ascii=False, indent=2)[:3000])

        # --- 3) 현재가
        body, _ = client.request(ep.QUOTE, {"stk_cd": args.quote})
        results["quote"] = _check_schema(f"현재가({args.quote})", body, ep.QUOTE)

        # 파싱까지 통과하는지 — 스키마가 맞아도 값 변환에서 깨질 수 있다
        print("\n--- 파싱 결과 ---")
        snap = broker.snapshot()
        print(f"  총자산 {snap.equity:,.0f}원 (예수금 {snap.deposit:,.0f} / "
              f"주식 {snap.total_eval:,.0f}) | 주문가능 {snap.cash:,.0f}")
        for h in snap.holdings.values():
            print(f"  보유 {h.code} {h.quantity}주 @ {h.avg_price:,.0f} "
                  f"→ 평가 {h.eval_amount:,.0f} ({h.pnl_rate:+.2f}%)")
        if not snap.holdings:
            print("  보유 종목 없음 — 잔고 스키마는 --order 로 1주 사본 뒤 다시 확인할 것")

        # --- 4) 주문 (선택)
        if args.order:
            print(f"\n--- 주문 검증: {args.order} 1주 시장가 매수 ---")
            try:
                res = broker.place_order(args.order, BUY, 1, order_type="market",
                                         dry_run=False)
                results["buy_order"] = res.ok
                print(f"  {'✅' if res.ok else '❌'} 주문번호={res.order_no} "
                      f"error={res.error}")
            except KiwoomAPIError as exc:
                results["buy_order"] = False
                print(f"  ❌ {exc}")

    print("\n" + "=" * 50)
    for name, ok in results.items():
        print(f"  {'✅ 통과' if ok else '❌ 불일치'}  {name}")
    if not all(results.values()):
        print("\n불일치가 있으면 src/data/kiwoom/endpoints.py 의 schema 를 실제 키로 고치고,")
        print("맞으면 해당 TRSpec 의 verified=True 로 바꾼 뒤 docs/KIWOOM_VERIFY.md 에 기록할 것.")
        return 1
    print("\n전부 일치 — endpoints.py 의 verified 플래그를 갱신할 것.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
