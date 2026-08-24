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
    verified=True,
    note="upd_stkpc_tp='1' 필수. 검증: 2018-05 삼성전자 50:1 분할 구간에서 "
         "ON=53,000원 연속 / OFF=2,650,000원→53,000원 점프 확인. "
         "응답은 최신→과거 내림차순, 페이지당 600건. "
         "⚠️ 거래정지일은 volume=0 + OHLC 동일값으로 채워져 온다(build 단계에서 제거).",
)

# --- 종목 기본정보: static covariate + PER/PBR ------------------------------
STOCK_INFO = TRSpec(  # UNVERIFIED
    name="stock_info",
    path="/api/dostk/stkinfo",
    api_id="ka10001",
    list_key="",  # 단일 객체 응답 — list_key 비어있으면 body 자체를 레코드로 취급
    schema={
        "code": ("stk_cd", "str"),
        "name": ("stk_nm", "str"),
        "sector": ("upName", "str"),
        "market_cap": ("mac", "abs_float"),
        "per": ("per", "float"),
        "pbr": ("pbr", "float"),
        "listed_shares": ("flo_stk", "abs_int"),
    },
)

# --- 투자자별 매매동향: 수급 피처 -------------------------------------------
INVESTOR_FLOW = TRSpec(  # UNVERIFIED
    name="investor_flow",
    path="/api/dostk/stkinfo",
    api_id="ka10059",
    list_key="stk_invsr_orgn",
    schema={
        "date": ("dt", "date"),
        "close": ("cur_prc", "abs_float"),
        "individual": ("ind_invsr", "int"),   # 개인 순매수 (부호 유지)
        "foreign": ("frgnr_invsr", "int"),    # 외국인 순매수
        "institution": ("orgn", "int"),       # 기관계 순매수
    },
    note="순매수는 부호가 의미를 갖는다 — abs_ 쓰지 말 것",
)

# --- 지수(KOSPI/KOSDAQ) 일봉: 매크로 시퀀스 --------------------------------
INDEX_DAILY = TRSpec(  # UNVERIFIED
    name="index_daily",
    path="/api/dostk/chart",   # VERIFIED(path) — MCP 패키지 api_paths 기준. sect 아님
    api_id="ka20006",
    list_key="inds_dt_pole_qry",
    schema={
        "date": ("dt", "date"),
        "open": ("open_pric", "abs_float"),
        "high": ("high_pric", "abs_float"),
        "low": ("low_pric", "abs_float"),
        "close": ("cur_prc", "abs_float"),
        "volume": ("trde_qty", "abs_int"),
    },
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

ALL_SPECS: dict[str, TRSpec] = {
    spec.name: spec
    # OVERSEAS_DAILY 는 의도적으로 제외 — 해외 일봉 TR 이 존재하지 않는다
    for spec in (DAILY_CHART, STOCK_INFO, INVESTOR_FLOW, INDEX_DAILY)
}


def unverified_specs() -> list[str]:
    return [name for name, spec in ALL_SPECS.items() if not spec.verified]
