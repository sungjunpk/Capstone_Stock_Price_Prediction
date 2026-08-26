"""키움 REST API TR 정의.

⚠️ 여기 값들은 전부 `# UNVERIFIED` 상태다.
MCP 서버(`kiwoom-rest-api`)로 실제 호출해서 응답을 확인한 뒤,
검증된 항목만 UNVERIFIED 주석을 지우고 사용할 것.
검증 절차는 docs/KIWOOM_VERIFY.md 참고.

각 TR 은 다음을 정의한다:
  path        : 호출 경로
  api_id      : 요청 헤더에 넣는 TR ID
  list_key    : 응답 JSON에서 레코드 배열이 들어있는 키
  schema      : {출력컬럼: (응답키, 타입)} — src.utils.parsing.parse_records 용
                타입: date/int/float/abs_int/abs_float/str
                가격류는 부호가 등락 방향으로 붙어오므로 abs_* 사용
"""

from __future__ import annotations

from dataclasses import dataclass, field

TOKEN_PATH = "/oauth2/token"  # VERIFIED
TOKEN_REVOKE_PATH = "/oauth2/revoke"  # UNVERIFIED


@dataclass(frozen=True)
class TRSpec:
    name: str
    path: str
    api_id: str
    list_key: str
    schema: dict[str, tuple[str, str]]
    rate_limit_per_sec: float = 3.0
    verified: bool = False
    note: str = ""
    cont_keys: tuple[str, str] = field(default=("cont-yn", "next-key"))


# --- 일봉 차트: 가격/거래량 피처의 근간 -------------------------------------
DAILY_CHART = TRSpec(  # VERIFIED 2026-08-24 (005930 mock, 10,914행 1985~현재)
    name="daily_chart",
    path="/api/dostk/chart",
    api_id="ka10081",
    list_key="stk_dt_pole_chart_qry",
    schema={
        "date": ("dt", "date"),
        "open": ("open_pric", "abs_float"),
        "high": ("high_pric", "abs_float"),
        "low": ("low_pric", "abs_float"),
        "close": ("cur_prc", "abs_float"),
        "volume": ("trde_qty", "abs_int"),
        "value": ("trde_prica", "abs_int"),  # 거래대금 — 단위 백만원 (검증됨)
    },
    rate_limit_per_sec=3.0,
    verified=True,
    note="upd_stkpc_tp='1' 필수. 검증: 2018-05 삼성전자 50:1 분할 구간에서 "
         "ON=53,000원 연속 / OFF=2,650,000원→53,000원 점프 확인. "
         "응답은 최신→과거 내림차순, 페이지당 600건. "
         "⚠️ 거래정지일은 volume=0 + OHLC 동일값으로 채워져 온다(build 단계에서 제거).",
)

# --- 종목 기본정보: static covariate + PER/PBR ------------------------------
STOCK_INFO = TRSpec(  # VERIFIED 2026-08-24 (005930 mock)
    name="stock_info",
    path="/api/dostk/stkinfo",
    api_id="ka10001",
    list_key="",  # 단일 객체 응답 — list_key 비어있으면 body 자체를 레코드로 취급
    schema={
        "code": ("stk_cd", "str"),
        "name": ("stk_nm", "str"),
        "market_cap": ("mac", "abs_float"),        # 억원
        "listed_shares": ("flo_stk", "abs_int"),   # 천주
        "per": ("per", "float"),
        "pbr": ("pbr", "float"),
        "eps": ("eps", "float"),
        "bps": ("bps", "float"),
        "roe": ("roe", "float"),
        "foreign_ratio": ("for_exh_rt", "float"),  # 외국인 소진율 %
    },
    verified=True,
    note="⚠️ 업종(sector) 필드가 없다 — 초기 가정 'upName' 은 존재하지 않는 필드였다. "
         "업종은 configs/config.yaml 의 universe 에 직접 적어 static covariate 로 쓴다. "
         "가격류에 +/- 부호가 붙으므로(cur_prc='-257000') abs_ 필수. "
         "PER/PBR 등은 조회 시점 스냅샷이라 과거 시계열로는 쓸 수 없다(look-ahead 주의).",
)

# --- 투자자별 매매동향: 수급 피처 -------------------------------------------
INVESTOR_FLOW = TRSpec(  # VERIFIED 2026-08-24 (005930 mock)
    name="investor_flow",
    path="/api/dostk/stkinfo",
    api_id="ka10059",
    list_key="stk_invsr_orgn",
    schema={
        "date": ("dt", "date"),
        "close": ("cur_prc", "abs_float"),
        "individual": ("ind_invsr", "int"),    # 개인 순매수 (부호 유지)
        "foreign": ("frgnr_invsr", "int"),     # 외국인 순매수
        "institution": ("orgn", "int"),        # 기관계 순매수
        "fin_invest": ("fnnc_invt", "int"),    # 금융투자 (기관 내 세부)
        "pension": ("penfnd_etc", "int"),      # 연기금등
        "etc_corp": ("etc_corp", "int"),       # 기타법인
    },
    verified=True,
    note="순매수는 부호가 의미를 갖는다 — abs_ 쓰지 말 것. "
         "페이지당 100건(일봉 TR의 600건보다 작다). next-key 는 날짜 문자열. "
         "⚠️ flu_rt 가 이 TR 에서는 '-870'(=-8.70%) 형식이라 ka10001('-8.70')과 다르다 — 안 쓴다.",
)

# --- 지수(KOSPI/KOSDAQ) 일봉: 매크로 시퀀스 --------------------------------
INDEX_DAILY = TRSpec(  # VERIFIED 2026-08-24 (KOSPI=001 mock)
    name="index_daily",
    path="/api/dostk/chart",   # sect 아님 — 초기 가정 오류였다
    api_id="ka20006",
    list_key="inds_dt_pole_qry",
    schema={
        "date": ("dt", "date"),
        "open": ("open_pric", "abs_float"),
        "high": ("high_pric", "abs_float"),
        "low": ("low_pric", "abs_float"),
        "close": ("cur_prc", "abs_float"),
        "volume": ("trde_qty", "abs_int"),
        "value": ("trde_prica", "abs_int"),
    },
    verified=True,
    note="지수값은 100배 스케일로 온다(KOSPI 669696 = 6696.96). "
         "수익률/비율로만 쓰므로 상수배는 상쇄되어 무해하지만, 그래프에 그릴 땐 /100 할 것. "
         "페이지당 600건.",
)

# --- 해외 지수/ETF (SOXX 등): 글로벌 크로스어텐션 입력 ----------------------
# ⚠️ 확인 결과: 키움 REST API 에 해외 일봉 차트 TR 이 없다.
#    해외주식 세그먼트(shsa)에 도구가 ka10014 하나뿐이고 일봉 차트가 아니다.
#    (ka20001 은 해외가 아니라 업종(sect) TR 이었다 — 초기 가정 오류)
#    → 글로벌 지표는 **국내상장 해외지수 ETF** 를 DAILY_CHART 로 받는 fallback 을 쓴다.
#      config.yaml 의 data.macro.overseas_etf_fallback 참고.
OVERSEAS_DAILY = TRSpec(  # DEPRECATED — 사용하지 말 것. 위 주석 참고
    name="overseas_daily",
    path="/api/dostk/sect",
    api_id="ka20001",
    list_key="ovs_stk_dt_pole",
    schema={
        "date": ("dt", "date"),
        "open": ("open_pric", "abs_float"),
        "high": ("high_pric", "abs_float"),
        "low": ("low_pric", "abs_float"),
        "close": ("cur_prc", "abs_float"),
        "volume": ("trde_qty", "abs_int"),
    },
    note="사용 금지 — 해외 일봉 TR 없음. 국내상장 해외지수 ETF 를 DAILY_CHART 로 수집할 것",
)

# --- 종목 리스트: 유니버스 자동 선정용 --------------------------------------
STOCK_LIST = TRSpec(  # VERIFIED 2026-08-24 (KOSPI mrkt_tp=0, 2477종목)
    name="stock_list",
    path="/api/dostk/stkinfo",
    api_id="ka10099",
    list_key="list",
    schema={
        "code": ("code", "str"),
        "name": ("name", "str"),
        "sector": ("upName", "str"),          # 업종 — ka10001 에는 없고 여기에 있다
        "size_class": ("upSizeName", "str"),  # 대형주/중형주/소형주
        "listed_shares": ("listCount", "abs_int"),
        "last_price": ("lastPrice", "abs_float"),  # 시가총액 = listed_shares × last_price
        "listing_date": ("regDay", "date"),
        "audit": ("auditInfo", "str"),        # '정상' 이 아니면 관리종목 등
        "market": ("marketName", "str"),
        "state": ("state", "str"),
        "kind": ("kind", "str"),
    },
    rate_limit_per_sec=0.5,  # 응답이 커서 제한이 빡빡하다 — 429 관측됨
    verified=True,
    note="mrkt_tp: '0'=KOSPI, '10'=KOSDAQ. 한 번에 전 종목이 오고 연속조회 없음(cont-yn=N). "
         "⚠️ upName(업종)이 여기 있다 — ka10001 에는 없으므로 업종은 이 TR 에서 받는다.",
)

ALL_SPECS: dict[str, TRSpec] = {
    spec.name: spec
    # OVERSEAS_DAILY 는 의도적으로 제외 — 해외 일봉 TR 이 존재하지 않는다
    for spec in (DAILY_CHART, STOCK_INFO, INVESTOR_FLOW, INDEX_DAILY, STOCK_LIST)
}


def unverified_specs() -> list[str]:
    return [name for name, spec in ALL_SPECS.items() if not spec.verified]


# ============================================================================
# 매매(모의투자) TR — 수집용이 아니라 자동매매 실행용이다.
#
# ⚠️ 위 수집 TR 과 달리 **주문 TR 은 상태를 바꾼다.** 잘못된 필드명 하나가
#    "주문이 안 나간다"가 아니라 "엉뚱한 수량이 나간다"로 이어질 수 있다.
#    scripts/verify_trading_trs.py 로 조회계 TR 을 먼저 검증하고,
#    주문은 반드시 --dry-run 으로 수량을 눈으로 확인한 뒤 --execute 한다.
# ============================================================================

# --- 예수금: 주문가능금액의 근거 --------------------------------------------
DEPOSIT = TRSpec(  # VERIFIED 2026-08-25 (mock, return_msg="모의투자 조회완료")
    name="deposit",
    path="/api/dostk/acnt",
    api_id="kt00001",
    list_key="",  # 단일 객체 응답
    schema={
        "deposit": ("entr", "abs_float"),             # 예수금
        "d2_deposit": ("d2_entra", "abs_float"),      # D+2 추정예수금
        "orderable": ("ord_alow_amt", "abs_float"),   # 주문가능금액
        "withdrawable": ("pymn_alow_amt", "abs_float"),
    },
    verified=True,
    note="qry_tp: '2'=일반조회, '3'=추정조회. 값은 15자리 zero-pad 문자열"
         "('000000010000000' = 10,000,000원) — parsing.to_float 가 처리한다. "
         "검증 시점 모의계좌 예수금 1,000만원.",
)

# --- 계좌평가잔고: 보유 종목과 매입가 ---------------------------------------
# 매입가가 여기서 나온다 = **손절 기준가를 로컬 상태가 아니라 브로커가 들고 있다.**
# 로컬 상태 파일이 날아가도 손절이 계속 동작한다.
ACCOUNT_BALANCE = TRSpec(  # VERIFIED 2026-08-26 (mock, 005930 1주 보유 상태에서 확인)
    name="account_balance",
    path="/api/dostk/acnt",
    api_id="kt00018",
    list_key="acnt_evlt_remn_indv_tot",
    schema={
        "code": ("stk_cd", "str"),
        "name": ("stk_nm", "str"),
        "quantity": ("rmnd_qty", "abs_int"),          # 보유수량
        "sellable": ("trde_able_qty", "abs_int"),     # 매도가능수량(미결제 제외)
        "avg_price": ("pur_pric", "abs_float"),       # 매입단가 — 손절 기준
        "current_price": ("cur_prc", "abs_float"),
        "eval_amount": ("evlt_amt", "abs_float"),
        "pnl_amount": ("evltv_prft", "float"),        # 부호가 의미를 갖는다
        "pnl_rate": ("prft_rt", "float"),
    },
    verified=True,
    note="qry_tp: '1'=합산, '2'=개별 / dmst_stex_tp: 'KRX'. "
         "응답 최상위에 요약(tot_evlt_amt 등)이 함께 오고 보유목록은 list_key 배열이다. "
         "요약은 ACCOUNT_SUMMARY_FIELDS 로 별도 파싱한다. "
         "⚠️ stk_cd 에 접두어가 붙어서 온다('A005930') — 실측 확인. "
         "broker._normalize_code 가 벗기지 않으면 보유분을 미보유로 읽어 이력 버퍼가 죽는다. "
         "보유가 0이면 배열이 비고 return_msg='모의투자 해당조회내역이 없습니다.' 다.",
)

# 계좌 요약(총평가/총손익)은 배열이 아니라 응답 최상위에 붙는다.
# TRSpec.schema 는 배열 레코드용이라 여기에 따로 둔다.
ACCOUNT_SUMMARY_FIELDS: dict[str, tuple[str, str]] = {
    "total_purchase": ("tot_pur_amt", "abs_float"),   # 총매입금액
    "total_eval": ("tot_evlt_amt", "abs_float"),      # 총평가금액
    "total_pnl": ("tot_evlt_pl", "float"),            # 총평가손익
    "total_pnl_rate": ("tot_prft_rt", "float"),
    "estimated_assets": ("prsm_dpst_aset_amt", "abs_float"),  # 추정예탁자산
}

# --- 현재가: 주문 수량 계산의 분모 ------------------------------------------
QUOTE = TRSpec(  # VERIFIED 2026-08-24 (ka10001 검증분과 동일 TR — 필드만 다르게 뽑는다)
    name="quote",
    path="/api/dostk/stkinfo",
    api_id="ka10001",
    list_key="",
    schema={
        "code": ("stk_cd", "str"),
        "name": ("stk_nm", "str"),
        "price": ("cur_prc", "abs_float"),
    },
    verified=True,
    note="cur_prc 에 등락 부호가 붙는다('-257000') → abs_float 필수. "
         "STOCK_INFO 와 같은 TR 이지만 매매 경로에서는 현재가만 필요해 스키마를 좁혔다.",
)

# --- 미체결: 같은 종목을 두 번 사지 않기 위한 것 -----------------------------
# 시장가라도 호가 잔량이 모자라면 남는다(실측: 셀트리온제약 207주 중 16주만 즉시 체결).
# 미체결을 모르고 다시 주문하면 부족분을 또 사서 **중복 매수**가 된다.
UNFILLED_ORDERS = TRSpec(  # VERIFIED 2026-08-26 (mock, 부분체결 3건 상태에서 확인)
    name="unfilled_orders",
    path="/api/dostk/acnt",
    api_id="ka10075",
    list_key="oso",
    schema={
        "code": ("stk_cd", "str"),
        "name": ("stk_nm", "str"),
        "order_no": ("ord_no", "str"),
        "order_qty": ("ord_qty", "abs_int"),
        "filled_qty": ("cntr_qty", "abs_int"),
        "unfilled_qty": ("oso_qty", "abs_int"),   # 남은 수량 — 이게 본체다
        "side": ("io_tp_nm", "str"),              # '+매수' / '-매도'
    },
    rate_limit_per_sec=1.0,
    verified=True,
    note="all_stk_tp '0'=전체 / trde_tp '0'=전체 / stex_tp '0'=통합. "
         "ord_stt 는 '체결'로 와도 oso_qty 가 남아 있을 수 있다 — 부분체결이다. "
         "상태 문자열이 아니라 oso_qty 로 판단할 것.",
)

# --- 당일매매일지: 종목별 실현손익 귀속의 원천 -------------------------------
# 발표용 성과 기록이 여기서 나온다. 주문이 아니라 **체결** 기준이라
# 부분체결이 있어도 실제 사고판 것만 남는다.
TRADE_DIARY = TRSpec(  # VERIFIED 2026-08-26 (mock, 매수 10 + 매도 1 상태에서 확인)
    name="trade_diary",
    path="/api/dostk/acnt",
    api_id="ka10170",
    list_key="tdy_trde_diary",
    schema={
        "code": ("stk_cd", "str"),
        "name": ("stk_nm", "str"),
        "buy_qty": ("buy_qty", "abs_int"),
        "buy_avg": ("buy_avg_pric", "abs_float"),
        "buy_amount": ("buy_amt", "abs_float"),
        "sell_qty": ("sell_qty", "abs_int"),
        "sell_avg": ("sel_avg_pric", "abs_float"),
        "sell_amount": ("sell_amt", "abs_float"),
        "pnl_amount": ("pl_amt", "float"),        # 부호가 의미를 갖는다
        "fee_tax": ("cmsn_alm_tax", "abs_float"),
        "pnl_rate": ("prft_rt", "float"),
    },
    rate_limit_per_sec=1.0,
    verified=True,
    note="base_dt(YYYYMMDD) / ottks_tp '1'=당일매수에대한매도 / ch_crd_tp '0'=전체. "
         "pl_amt 는 수수료·세금을 반영한 실현손익이다(매도가 없으면 0). "
         "⚠️ 필드명이 sell_qty 인데 매도평균가는 sel_avg_pric 이다 — 철자가 다르다.",
)

# --- 주문: 여기부터는 계좌 상태를 바꾼다 ------------------------------------
BUY_ORDER = TRSpec(  # VERIFIED 2026-08-26 (mock, 005930 1주 시장가 → ord_no=0123420)
    name="buy_order",
    path="/api/dostk/ordr",
    api_id="kt10000",
    list_key="",
    schema={
        "order_no": ("ord_no", "str"),
        "exchange": ("dmst_stex_tp", "str"),
    },
    rate_limit_per_sec=1.0,   # 주문은 천천히. 429 로 중복주문 재시도를 만들지 않는다
    verified=True,
    note="필수: dmst_stex_tp('KRX') stk_cd ord_qty trde_tp. "
         "trde_tp '3'=시장가(ord_uv 불필요), '0'=보통(지정가, ord_uv 필요). "
         "응답 주문번호 키는 ord_no 가 맞다. 계좌번호는 요청에 없다 — 계좌는 APP_KEY 에 묶인다.",
)

SELL_ORDER = TRSpec(  # VERIFIED 2026-08-26 (mock, 005930 1주 → ord_no=0125994)
    name="sell_order",
    path="/api/dostk/ordr",
    api_id="kt10001",
    list_key="",
    schema={
        "order_no": ("ord_no", "str"),
        "exchange": ("dmst_stex_tp", "str"),
    },
    rate_limit_per_sec=1.0,
    verified=True,
    note="BUY_ORDER 와 요청 형식 동일. 매도만 거래세가 붙는다(costs.tax_bps). "
         "실측 확인: 264,750 매수 → 264,000 매도에 수수료·세금 2,368원, 실현손익 -3,118원.",
)

TRADING_SPECS: dict[str, TRSpec] = {
    spec.name: spec
    for spec in (DEPOSIT, ACCOUNT_BALANCE, QUOTE, UNFILLED_ORDERS, TRADE_DIARY,
                 BUY_ORDER, SELL_ORDER)
}

# 레이트 리미터에 등록할 전체 목록. ALL_SPECS 는 '수집' TR 만 담는다 —
# collect.py 의 미검증 경고가 매매 TR 까지 끌어오지 않게 분리해 둔다.
RATE_LIMITED_SPECS: dict[str, TRSpec] = {**ALL_SPECS, **TRADING_SPECS}
