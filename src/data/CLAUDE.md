# src/data — 데이터 수집·저장

> 루트 [`CLAUDE.md`](../../CLAUDE.md)의 절대 규칙이 우선한다. 이 문서는 이 폴더 상세.
> 실제 API 응답 검증 결과는 [`docs/KIWOOM_VERIFY.md`](../../docs/KIWOOM_VERIFY.md).

## 데이터 소스 결정

**키움증권 신규 REST API 하나로 통일** — 과거 데이터 수집과 모의투자 실행 모두.

KRX Data Marketplace, 한국투자증권 API, pykrx도 검토했으나 키움 단일 소스로 결정했다.
이미 키움 계좌를 보유하고 있어 별도 계좌 개설이 불필요하고, 모의투자까지 한 파이프라인으로
연결할 수 있기 때문이다.

MCP 서버(`kiwoom-rest-api` npm 패키지)도 등록되어 있지만 **보조 도구**다.
응답 구조를 빠르게 훑어볼 때만 쓰고, 실제 수집은 `src/data/kiwoom/client.py`가 담당한다.

## 키움 REST API 제약 (코드 작성 시 반드시 고려)

| 제약 | 대응 |
|---|---|
| TR별 초당 호출 제한 | 토큰 버킷 레이트 리미터 필수 (`src/utils/ratelimit.py`) |
| 벌크 엔드포인트 없음 | 종목별 반복 호출 구조. 순회 중 한 종목 실패가 전체를 죽이면 안 된다 |
| 응답값이 전부 문자열 | `"+70000"`, `"1,234,567"` → 공통 파싱 유틸(`src/utils/parsing.py`) 경유 필수 |
| 가격에 등락 부호가 붙음 | 가격류는 `abs_*` 타입으로 파싱. **순매수는 부호가 의미를 가지므로 절대 abs 금지** |
| mock/live base URL 분리 | 환경변수로만 분기. 코드에 URL 하드코딩 금지 |
| 응답이 최신→과거 내림차순 | 저장 전 정렬. 조기 종료 로직이 이 순서에 의존한다 |

## 수집 전략

- 한 번 받은 데이터는 `data/raw/`에 parquet으로 저장. **매번 재수집 금지**
- 이후에는 장마감 후 당일치만 추가하는 **증분 수집**
- 증분은 `storage.upsert()`가 담당: 기존 읽기 → concat → 키 기준 dedup → 정렬 → 덮어쓰기
  - 같은 key가 겹치면 **새 데이터가 이긴다** (수정주가 소급 반영 때문)
- 재실행 안전(idempotent)이 이 폴더의 핵심 계약이다. 깨지면 `tests/test_storage.py`가 잡는다

## 파일 레이아웃

```
data/raw/{TR이름}/{종목코드}.parquet   ← API 응답 원본 (가공 전)
data/processed/{이름}.parquet          ← 지표·라벨까지 계산된 학습용 테이블
```

원본을 남기는 이유: 지표 계산 로직을 바꿔도 **재수집 없이** 다시 만들 수 있다.

## TR 정의 규칙 (`kiwoom/endpoints.py`)

- 실제 응답으로 검증하기 전까지는 `# UNVERIFIED` 주석과 `verified=False`를 유지한다
- 검증이 끝나면 주석을 지우고 `verified=True`, `note`에 함정을 기록한다
- `scripts/collect.py`는 실행 시 미검증 TR을 경고로 출력한다

**현재 상태 (2026-08-24)**: `daily_chart` `stock_info` `investor_flow` `index_daily` 4종 검증 완료.
해외 일봉 TR은 **존재하지 않아** 국내상장 ETF로 대체한다.

## 모의투자

모델 학습이 안정화된 후반부에 별도 파이프라인으로 구축한다. 지금은 수집만 다룬다.
