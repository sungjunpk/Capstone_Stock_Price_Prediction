"""키움 모의투자 계좌·주문 어댑터 — 매매 판단 5단계의 '전송' 부분.

이 파일의 책임은 딱 세 가지다:
  1) 계좌 상태를 읽는다 (예수금 / 보유종목 / 매입단가)
  2) 현재가를 읽는다 (목표비중 → 주문수량 환산의 분모)
  3) 주문을 낸다 (매수/매도)

**매매 판단은 여기 없다.** 무엇을 얼마나 살지는 signal.py + risk.py 가 정하고,
이 파일은 그 결정을 주문으로 옮기기만 한다. 판단 로직이 여기 들어오면
백테스트와 모의투자가 갈라진다(CLAUDE.md 절대 규칙 7).

⚠️ 안전장치 3중:
  - `KIWOOM_ENV=live` 는 config.py 가 예외로 막는다
  - 그걸 통과해도 `PaperBroker.__init__` 이 `is_mock` 을 한 번 더 확인한다
  - 주문 함수는 `dry_run=True` 가 기본값이다. 실제 전송은 명시적으로 꺼야 나간다
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from src.data.kiwoom import endpoints as ep
from src.data.kiwoom.client import KiwoomAPIError, KiwoomClient
from src.utils.config import KiwoomSettings
from src.utils.logging import get_logger
from src.utils.parsing import parse_records, to_float

log = get_logger(__name__)

BUY = "buy"
SELL = "sell"

# 주문 유형 → 키움 trde_tp. 시장가는 ord_uv 를 보내지 않는다.
TRADE_TYPE = {"market": "3", "limit": "0"}
EXCHANGE = "KRX"


@dataclass(frozen=True)
class Holding:
    """브로커가 알려주는 보유 상태. 로컬 추정이 아니다."""

    code: str
    name: str
    quantity: int
    sellable: int
    avg_price: float          # 매입단가 — 손절 기준가가 여기서 나온다
    current_price: float
    eval_amount: float
    pnl_amount: float = 0.0
    pnl_rate: float = 0.0


@dataclass(frozen=True)
class AccountSnapshot:
    """한 시점의 계좌 전체. 이 값으로 목표비중을 금액으로 환산한다."""

    cash: float                       # 주문가능금액
    deposit: float                    # 예수금
    holdings: dict[str, Holding] = field(default_factory=dict)
    total_eval: float = 0.0           # 주식 평가금액 합
    total_pnl: float = 0.0
    estimated_assets: float = 0.0     # 추정예탁자산 — 키움이 계산해준 총자산
    fetched_at: str = ""

    @property
    def equity(self) -> float:
        """총자산. 비중(target_weight)의 분모다.

        **예수금 + 주식평가로 계산하면 안 된다.** 예수금은 D+2 결제라 매수 대금이
        아직 안 빠져 있어서, 매수 직후 같은 돈을 현금과 주식으로 두 번 센다.
        2026-08-26 실측: 실제 9,929만원인 계좌가 1억 + 8,203만 = 1억 8,203만으로 나왔다.
        결제 전에 리밸런싱이 돌면 목표비중의 분모가 1.8배라 주문가능금액을
        훨씬 넘는 주문이 나간다.

        **주문가능금액 + 주식평가합** 이다. 주문가능금액은 매수 즉시 차감되므로
        예수금과 달리 이중계상이 없다.

        ⚠️ 키움의 `prsm_dpst_aset_amt`(추정예탁자산)를 쓰지 않는다. 정답처럼 보이지만
           증권사 화면과 어긋난다 — 2026-08-28 실측: 키움 웹 모의투자 화면이
           100,683,305원(+0.68%)인데 이 필드는 그보다 작게 온다. 주문가능금액 +
           주식평가합은 **원 단위까지 화면과 일치**했다(차이 0원).
           2026-08-27 도 같다: 재구성 99,770,415(-0.23%) = 화면, 필드 99,310,070(-0.69%).
           방향도 일정하지 않다 — 미체결 주문이 있으면 반대로 크게 온다.
        """
        return self.cash + self.total_eval

    def weight_of(self, code: str) -> float:
        eq = self.equity
        if eq <= 0 or code not in self.holdings:
            return 0.0
        return self.holdings[code].eval_amount / eq


@dataclass(frozen=True)
class OrderResult:
    code: str
    side: str
    quantity: int
    price: float              # 주문 시점 참고가(시장가면 체결가와 다르다)
    order_type: str
    dry_run: bool
    order_no: str | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def to_dict(self) -> dict:
        return {
            "code": self.code, "side": self.side, "quantity": self.quantity,
            "price": self.price, "order_type": self.order_type,
            "dry_run": self.dry_run, "order_no": self.order_no, "error": self.error,
        }


class PaperBroker:
    """모의투자 전용 브로커. 실전 경로는 존재하지 않는다."""

    def __init__(self, client: KiwoomClient | None = None):
        self.client = client or KiwoomClient()
        settings: KiwoomSettings = self.client.settings
        if not settings.is_mock:
            # config.py 가 이미 막지만, 여기서도 막는다. 주문은 되돌릴 수 없다.
            raise RuntimeError(
                f"PaperBroker 는 모의투자 전용이다 (KIWOOM_ENV={settings.env}). "
                "이 프로젝트에 실전투자 경로는 없다."
            )
        log.info("모의투자 브로커 준비 (%s)", settings.base_url)

    # ------------------------------------------------------------ 조회
    def fetch_deposit(self) -> dict:
        """예수금·주문가능금액. qry_tp='2' 일반조회."""
        data, _ = self.client.request(ep.DEPOSIT, {"qry_tp": "2"})
        row = parse_records([data], ep.DEPOSIT.schema).iloc[0].to_dict()
        return {k: (0.0 if v is None else float(v)) for k, v in row.items()}

    def fetch_holdings(self) -> tuple[dict[str, Holding], dict]:
        """보유종목 + 계좌요약. 연속조회로 전부 받는다."""
        body = {"qry_tp": "1", "dmst_stex_tp": EXCHANGE}
        holdings: dict[str, Holding] = {}
        summary: dict = {}

        for page in self.client.paginate(ep.ACCOUNT_BALANCE, body):
            if not summary:
                summary = {
                    key: to_float(page.get(src), abs_value=kind.startswith("abs"))
                    for key, (src, kind) in ep.ACCOUNT_SUMMARY_FIELDS.items()
                }
            records = page.get(ep.ACCOUNT_BALANCE.list_key) or []
            df = parse_records(records, ep.ACCOUNT_BALANCE.schema)
            for r in df.itertuples():
                code = _normalize_code(r.code)
                qty = int(r.quantity or 0)
                if not code or qty <= 0:
                    continue
                holdings[code] = Holding(
                    code=code,
                    name=str(r.name or ""),
                    quantity=qty,
                    # 매도가능수량이 안 오면 보유수량으로 물러난다(D+2 미결제분은 어차피 거부된다)
                    sellable=int(r.sellable if r.sellable is not None else qty),
                    avg_price=float(r.avg_price or 0.0),
                    current_price=float(r.current_price or 0.0),
                    eval_amount=float(r.eval_amount or 0.0),
                    pnl_amount=float(r.pnl_amount or 0.0),
                    pnl_rate=float(r.pnl_rate or 0.0),
                )
        return holdings, {k: (v or 0.0) for k, v in summary.items()}

    def snapshot(self) -> AccountSnapshot:
        """계좌 한 장. 매 실행의 출발점이다."""
        dep = self.fetch_deposit()
        holdings, summary = self.fetch_holdings()

        total_eval = summary.get("total_eval") or sum(
            h.eval_amount for h in holdings.values()
        )
        snap = AccountSnapshot(
            cash=dep.get("orderable", 0.0),
            deposit=dep.get("deposit", 0.0),
            holdings=holdings,
            total_eval=float(total_eval),
            total_pnl=float(summary.get("total_pnl") or 0.0),
            estimated_assets=float(summary.get("estimated_assets") or 0.0),
            fetched_at=datetime.now().isoformat(timespec="seconds"),
        )
        log.info(
            "계좌: 총자산 %s원 (주문가능 %s / 예수금 %s / 주식 %s) | 보유 %d종목",
            f"{snap.equity:,.0f}", f"{snap.cash:,.0f}", f"{snap.deposit:,.0f}",
            f"{snap.total_eval:,.0f}", len(holdings),
        )
        return snap

    def fetch_unfilled(self) -> dict[str, int]:
        """종목코드 → 미체결 수량. **중복 매수를 막는 유일한 근거다.**

        시장가라도 호가 잔량이 모자라면 남는다. 남은 걸 모르고 다시 주문하면
        부족분을 또 사서 목표비중을 넘긴다.

        ⚠️ `ord_stt` 가 '체결' 이어도 `oso_qty` 가 남아 있다(부분체결). 상태 문자열이
           아니라 수량으로 판단한다.
        """
        out: dict[str, int] = {}
        body = {"all_stk_tp": "0", "trde_tp": "0", "stk_cd": "", "stex_tp": "0"}
        try:
            data, _ = self.client.request(ep.UNFILLED_ORDERS, body)
        except KiwoomAPIError as exc:
            # 조회 실패를 '미체결 없음'으로 읽으면 중복 주문이 나간다.
            log.error("[미체결] 조회 실패: %s — 주문 전에 확인할 것", exc)
            raise
        df = parse_records(data.get(ep.UNFILLED_ORDERS.list_key) or [],
                           ep.UNFILLED_ORDERS.schema)
        for r in df.itertuples():
            code = _normalize_code(r.code)
            qty = int(r.unfilled_qty or 0)
            if code and qty > 0:
                out[code] = out.get(code, 0) + qty
        if out:
            log.warning("[미체결] %d종목 남아 있다 — 이번 회차에서 제외한다: %s",
                        len(out), ", ".join(f"{c}({q}주)" for c, q in sorted(out.items())))
        return out

    def fetch_trade_diary(self, base_dt: str) -> list[dict]:
        """당일매매일지 — 종목별 실현손익. 주문이 아니라 **체결** 기준이다.

        ⚠️ `ottks_tp` 는 반드시 **"2"(당일매도전체)** 다. "1"(당일매수에대한매도)로
        두면 **같은 날 사서 같은 날 판 것만** 손익이 붙고, 하루라도 들고 있다 판
        종목은 `pl_amt` 가 조용히 0 으로 온다 — 오류가 아니라 0 이라 눈치채기 어렵다.

        실측(2026-09-02, 같은 날짜에 값만 바꿔 비교):

            8/26 삼성전자(당일 왕복)  tp=1 → -3,118      tp=2 → -3,118
            8/27 기아(오버나잇)       tp=1 →      0 ❌   tp=2 → -423,864
            9/2  009540(오버나잇)     tp=1 →      0 ❌   tp=2 → -596,348

        tp=2 는 당일 왕복도 그대로 주므로 상위집합이다. 행 개수도 같다.
        이 프로젝트는 리밸런싱 주기가 10거래일이라 **매도는 거의 전부 오버나잇**이다
        — tp=1 이면 실현손익이 사실상 항상 0 이 된다.
        """
        body = {"base_dt": base_dt, "ottks_tp": "2", "ch_crd_tp": "0"}
        data, _ = self.client.request(ep.TRADE_DIARY, body)
        df = parse_records(data.get(ep.TRADE_DIARY.list_key) or [],
                           ep.TRADE_DIARY.schema)
        rows = []
        for r in df.itertuples():
            code = _normalize_code(r.code)
            if not code:
                continue
            rows.append({
                "code": code,
                "name": str(r.name or ""),
                "buy_qty": int(r.buy_qty or 0),
                "buy_avg": float(r.buy_avg or 0.0),
                "buy_amount": float(r.buy_amount or 0.0),
                "sell_qty": int(r.sell_qty or 0),
                "sell_avg": float(r.sell_avg or 0.0),
                "sell_amount": float(r.sell_amount or 0.0),
                "pnl_amount": float(r.pnl_amount or 0.0),
                "fee_tax": float(r.fee_tax or 0.0),
                "pnl_rate": float(r.pnl_rate or 0.0),
            })
        return rows

    def fetch_prices(self, codes: list[str]) -> dict[str, float]:
        """현재가. 종목당 1회 호출이라 유니버스 전체가 아니라 **거래 대상만** 넘길 것."""
        out: dict[str, float] = {}
        for code in codes:
            try:
                data, _ = self.client.request(ep.QUOTE, {"stk_cd": code})
                price = to_float(data.get("cur_prc"), abs_value=True)
                if price and price > 0:
                    out[code] = float(price)
                else:
                    log.warning("[quote] %s: 현재가를 못 읽었다", code)
            except KiwoomAPIError as exc:
                log.warning("[quote] %s 실패: %s", code, exc)
        return out

    # ------------------------------------------------------------ 주문
    def place_order(
        self,
        code: str,
        side: str,
        quantity: int,
        *,
        price: float = 0.0,
        order_type: str = "market",
        dry_run: bool = True,
    ) -> OrderResult:
        """단일 주문. **기본이 dry_run 이다** — 실제 전송은 호출자가 명시해야 한다."""
        if quantity <= 0:
            return OrderResult(code, side, quantity, price, order_type, dry_run,
                               error="수량 0 — 주문하지 않는다")
        if order_type not in TRADE_TYPE:
            raise ValueError(f"알 수 없는 주문 유형: {order_type}")
        if order_type == "limit" and price <= 0:
            return OrderResult(code, side, quantity, price, order_type, dry_run,
                               error="지정가인데 가격이 없다")

        spec = ep.BUY_ORDER if side == BUY else ep.SELL_ORDER
        body = {
            "dmst_stex_tp": EXCHANGE,
            "stk_cd": code,
            "ord_qty": str(int(quantity)),
            "trde_tp": TRADE_TYPE[order_type],
        }
        if order_type == "limit":
            body["ord_uv"] = str(int(price))

        if dry_run:
            log.info("[모의주문] %s %s %d주 (%s) — 전송하지 않음",
                     code, "매수" if side == BUY else "매도", quantity, order_type)
            return OrderResult(code, side, quantity, price, order_type, True)

        try:
            data, _ = self.client.request(spec, body)
        except KiwoomAPIError as exc:
            log.error("[주문실패] %s %s %d주: %s", code, side, quantity, exc)
            return OrderResult(code, side, quantity, price, order_type, False,
                               error=str(exc))

        order_no = str(data.get("ord_no") or "")
        log.info("[주문] %s %s %d주 → 주문번호 %s",
                 code, "매수" if side == BUY else "매도", quantity, order_no or "?")
        return OrderResult(code, side, quantity, price, order_type, False,
                           order_no=order_no or None)

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> PaperBroker:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _normalize_code(raw) -> str:
    """키움 잔고는 종목코드에 접두어(A005930)나 공백이 붙어 오는 경우가 있다.

    유니버스/패널의 코드는 순수 6자리라, 여기서 맞춰주지 않으면
    **보유 중인 종목을 미보유로 착각해 이력 버퍼가 통째로 무력화된다.**
    """
    s = str(raw or "").strip().upper()
    if s.startswith("A") and len(s) == 7 and s[1:].isdigit():
        s = s[1:]
    return s.zfill(6) if s.isdigit() else s
