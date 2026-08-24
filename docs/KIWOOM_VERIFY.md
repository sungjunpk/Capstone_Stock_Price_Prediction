# 키움 REST API 검증 절차

`src/data/kiwoom/endpoints.py` 의 모든 TR 은 현재 **`# UNVERIFIED`** 상태다.
서드파티 MCP 패키지와 비공식 문서를 근거로 적어둔 값이라, 실제 응답과 다를 수 있다.
아래 절차로 하나씩 확인하고 검증된 것만 표시를 지운다.

## 0. 준비

MCP 서버는 등록 완료 상태다(local scope, 이 프로젝트 폴더 전용).

```bash
# 등록 확인
claude mcp list          # kiwoom: ✔ Connected 나와야 정상

# 키 재발급 후 다시 등록할 때
claude mcp remove kiwoom
set -a; source .env; set +a
claude mcp add kiwoom \
  -e KIWOOM_ENV=mock \
  -e "KIWOOM_APP_KEY=$KIWOOM_APP_KEY" \
  -e "KIWOOM_APP_SECRET=$KIWOOM_APP_SECRET" \
  -- npx -y kiwoom-rest-api
```

⚠️ `--scope project` 는 쓰지 말 것 — `.mcp.json` 은 커밋되는 파일이라 시크릿이 노출된다.
⚠️ 키움 포털에서 **내 IP 등록**(My Page → IP 관리)을 해야 호출이 통과한다.

MCP 도구 이름 규칙: `kiwoom_{세그먼트}_{TR ID}` (예: `kiwoom_chart_ka10081`). 총 181개.

## 1. TR 하나씩 검증

각 TR 마다 확인할 것:

| 확인 항목 | 무엇을 보나 |
|---|---|
| `path` / `api_id` | 200 응답이 오는가 |
| `list_key` | 레코드 배열이 실제로 그 키에 들어있는가 |
| `schema` 의 응답키 | 필드명이 정확한가 (오타 하나로 전 컬럼이 None) |
| 부호 처리 | 가격 필드에 `+/-` 가 붙는가 → 붙으면 `abs_*` 유지, 아니면 일반 타입 |
| 연속조회 | 응답 헤더에 `cont-yn` / `next-key` 가 오는가 |
| 초당 제한 | 몇 회에서 429 가 뜨는가 → `rate_limit_per_sec` 조정 |

## 2. 검증 완료 처리

응답이 정의와 일치하면:

1. `endpoints.py` 의 해당 TRSpec 에서 `# UNVERIFIED` 주석 삭제
2. `verified=True` 로 변경
3. 실제 응답 샘플을 `docs/samples/{tr_name}.json` 으로 저장 (민감정보 있으면 마스킹)

`scripts/collect.py` 는 실행 시 미검증 TR 목록을 경고로 출력한다.

## 3. 검증 현황

| TR | api_id | path | 상태 | 비고 |
|---|---|---|---|---|
| daily_chart | ka10081 | `/api/dostk/chart` | ✅ **검증완료 2026-08-24** | 아래 상세 참고 |
| stock_info | ka10001 | `/api/dostk/stkinfo` | path ✅ / 응답 UNVERIFIED | 단일 객체 응답 — list_key 없음 |
| investor_flow | ka10059 | `/api/dostk/stkinfo` | path ✅ / 응답 UNVERIFIED | 순매수 부호 유지 필요 |
| index_daily | ka20006 | `/api/dostk/chart` | path ✅ / 응답 UNVERIFIED | 최초 가정(`/sect`)은 틀렸음. KOSPI=001, KOSDAQ=101 코드 확인 필요 |
| ~~overseas_daily~~ | ~~ka20001~~ | — | ❌ **존재하지 않음** | 키움에 해외 일봉 TR 없음(shsa 세그먼트에 ka10014 하나뿐). ka20001 은 업종 TR 이었다. → 국내상장 ETF fallback 사용 |

*path 는 MCP 패키지의 `api_paths` 정의로 대조 완료.*

### daily_chart (ka10081) 검증 상세 — 005930, mock

| 항목 | 결과 |
|---|---|
| 토큰 발급 | `POST /oauth2/token` → `{"token": ..., "expires_dt": "20260825163737", "return_code": 0}` (≈24h) |
| list_key | `stk_dt_pole_chart_qry` ✅ 정의와 일치 |
| 응답 필드 | `dt` `open_pric` `high_pric` `low_pric` `cur_prc` `trde_qty` `trde_prica` ✅ 전부 일치 |
| 부호 | 이 TR 은 가격에 `+/-` 안 붙음 (`"257000"`). `abs_float` 유지해도 무해 |
| 정렬 | **최신 → 과거 내림차순**, 페이지당 600건 |
| 연속조회 | 응답 헤더 `cont-yn: Y` / `next-key` ✅ 동작 확인 |
| **제공 범위** | **1985-01-04 ~ 현재, 10,914 거래일** — mock 도 전체 이력 제공 ✅ |
| 수정주가 | `upd_stkpc_tp="1"` 정상. 2018-05 삼성전자 50:1 분할 구간에서 ON=53,000원 연속 / OFF=2,650,000→53,000 점프 확인 |
| `trde_prica` 단위 | **백만원** (32.45M주 × 257,000원 ≈ 8.34조 vs 응답 8,455,948) |

**⚠️ 발견된 함정: 거래정지일**

분할 전후 거래정지일(2018-04-30, 05-02, 05-03)이 `volume=0` + `OHLC 전부 직전 종가` 로 채워져 온다.
그대로 두면 가짜 0 수익률 → 변동성 과소추정 → ATR/RSI 왜곡으로 번진다.
→ `src/features/technical.py: drop_halted_days()` 가 지표 계산 **전에** 제거한다.

## 4. 함정 목록

- **수정주가**: 끄고 받으면 액면분할 구간에서 수익률이 폭발한다. 반드시 켤 것.
- **거래일 정렬**: 키움 차트 TR 은 보통 **최신 → 과거** 순으로 온다. 저장 전 정렬은
  `storage.upsert(sort_by=["date"])` 가 처리하지만, 지표 계산 전에도 오름차순인지 확인.
- **모의투자 데이터 범위**: mock 서버는 과거 데이터 제공 기간이 짧을 수 있다.
  이 경우 학습 데이터 확보 방안을 별도로 정해야 한다 — **초기에 반드시 확인할 것**.
- **live 금지**: `KIWOOM_ENV=live` 는 `src/utils/config.py` 에서 예외를 던지도록 막아뒀다.
