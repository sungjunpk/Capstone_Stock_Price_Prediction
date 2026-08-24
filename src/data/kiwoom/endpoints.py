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
