# 주가예측 딥러닝 캡스톤 프로젝트

기존에 단순 Transformer로 만든 주가예측 프로젝트를 처음부터 갈아엎는 졸업작품.
"AI 원클릭"처럼 보이지 않도록 **구조적 참신함과 깊이를 확보**하는 것이 최우선 목표.
섹터는 특정 업종에 얽매이지 않음 — 데이터 수급 가능 여부가 우선.

핵심 차별점 두 가지:
1. **기권 로직** — "오를까/내릴까"뿐 아니라 "지금 판단해도 되는가"까지 모델이 결정
2. **해석가능성** — TFT 변수선택망으로 피처 중요도를 자동 학습해 리포트 근거로 사용

## 절대 규칙 (위반 금지)

1. **`live` 금지.** `KIWOOM_ENV`는 항상 `mock`. 실전 키는 저장조차 하지 않는다.
   `src/utils/config.py`가 `live`를 예외로 막고 있다 — 이 방어를 풀지 말 것.
2. **비밀값은 코드/커밋에 넣지 않는다.** `.env`만 사용하고 `.env.example`에는 키 이름만.
3. **Phase 1이 안정적으로 완성되기 전에 Phase 2로 넘어가지 않는다.**
   Phase 2는 실패해도 제출에 지장 없어야 한다.
4. **수집은 항상 증분(idempotent).** 같은 명령을 두 번 실행해도 중복 행이 생기면 안 된다.
5. **look-ahead 금지.** 새 피처가 t 시점 이후 정보를 참조하지 않는지 반드시 확인한다.
6. **정규화 통계는 train 구간에서만 계산한다.** 날짜 기준 global split 필수.
7. **매매 신호는 `src/trading/signal.py` 하나만 사용한다.**
   백테스트와 모의투자가 같은 코드를 공유 — 실행 경로별 분기 금지.
   추론도 마찬가지다: `src/models/inference.py` 하나만 쓴다.
8. **실험 결과는 `outputs/reports/`에 날짜+설정 해시로 남긴다.** 덮어쓰지 않는다.
9. **주문은 기본이 dry-run 이다.** `scripts/paper_trade.py --execute` 로만 나간다.
   대시보드에서 주문을 낼 수 있게 만들지 않는다 — 화면은 읽기 전용이다.
   **자동 매매(2026-08-26~)도 이 규칙 안에 있다.** `scripts/daily_trade.sh` 가
   평일 15:15 에 도는데, 하는 일은 위 명령을 부르는 것뿐이다 — 새 주문 경로를
   만들지 않는다. 설치는 수집 자동화와 **분리**돼 있다
   (`install_daily_trade.sh`) — 주문을 내는 자동화는 스스로 켜는 동작이어야 한다.

## 작업 원칙

> **속도보다 신중함.** 사소한 작업에는 판단해서 적용한다.

**코딩 전에 생각한다.** 가정하지 않는다. 확신이 없으면 묻는다. 해석이 여러 갈래면
전부 제시한다 — 혼자 하나 고르고 넘어가지 않는다. 더 단순한 방법이 있으면 말하고,
근거가 있으면 반대 의견을 낸다. 불분명하면 멈추고, 무엇이 헷갈리는지 이름 붙여 묻는다.

**단순함이 먼저다.** 문제를 푸는 최소한의 코드. 요청받지 않은 기능·추상화·"유연성"·
일어날 수 없는 상황의 예외 처리를 넣지 않는다. 200줄을 썼는데 50줄로 되겠다면 다시 쓴다.
기준: **"시니어 엔지니어가 이걸 보고 과하다고 할까?"**

**최소 침습.** 건드려야 하는 것만 건드린다. 옆의 코드·주석·서식을 "개선"하지 않고,
안 깨진 것을 리팩터링하지 않고, 내 취향과 달라도 기존 스타일에 맞춘다. 무관한 죽은
코드는 **말만 하고 지우지 않는다** — 단, 내 변경 때문에 고아가 된 import·함수는 지운다.
판정 기준: **바뀐 모든 줄이 사용자의 요청으로 곧장 추적되는가.**

**목표 기반 실행.** 성공 기준을 먼저 정의하고 검증될 때까지 돈다. "버그 수정"은
재현 테스트를 먼저 쓰는 것이고, "리팩터링"은 전후로 테스트가 다 통과하는 것이다.
여러 단계짜리 작업은 각 단계의 **검증 방법**을 포함한 짧은 계획을 먼저 밝힌다.

### 이 프로젝트에서는

- **큰 작업(모델 아키텍처 등)은 Plan Mode로 설계 검토 후 승인받고 진행**
- 데이터 수집/전처리처럼 반복적이고 리스크 낮은 작업은 자동 진행 가능
- **학습/백테스트는 결과를 직접 확인하면서 진행** — 자동 반복 실행으로 GPU/시간 낭비 금지
- 커밋은 작업 단위로. 커밋 메시지가 곧 작업 이력이다

## 상세 문서 (작업할 폴더의 것만 읽으면 된다)

| 문서 | 내용 |
|---|---|
| [`src/data/CLAUDE.md`](src/data/CLAUDE.md) | 키움 API 제약, 수집 전략, TR 정의 규칙 |
| [`src/features/CLAUDE.md`](src/features/CLAUDE.md) | 피처 설계표, look-ahead 방지, 스케일 정규화 |
| [`src/models/CLAUDE.md`](src/models/CLAUDE.md) | Phase 1/2 아키텍처 상세 |
| [`src/training/CLAUDE.md`](src/training/CLAUDE.md) | 분할 규칙, 손실, anti-overfitting |
| [`src/evaluation/CLAUDE.md`](src/evaluation/CLAUDE.md) | 평가 지표, walk-forward 백테스트 |
| [`src/trading/CLAUDE.md`](src/trading/CLAUDE.md) | 매매 판단 5단계, 기권 로직, 모의투자 실행 |
| [`src/webapp/CLAUDE.md`](src/webapp/CLAUDE.md) | 대시보드 — 읽기 전용 원칙 |
| [`docs/KIWOOM_VERIFY.md`](docs/KIWOOM_VERIFY.md) | 실제 API 응답 검증 결과와 함정 |
| [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) | 한계 — KCMI 4대 모델리스크 대입, 실측 근거 |
| [`docs/KAGGLE_SETUP.md`](docs/KAGGLE_SETUP.md) | 클라우드 GPU 학습 절차 (수집은 로컬, 학습만 외부) |
| [`README.md`](README.md) | 현재 진행 상태, 다음 할 일, 세션 시작 절차 |

**진행 상황은 이 문서가 아니라 `README.md`에 적는다.** 여기는 규칙만 담는다.

> Phase 1 구현에서 확정된 설계 변경 하나: **VSN 은 인코더 앞에 둔다**(TFT 원논문 순서).
> 성능상 5배 차이가 나고 해석가능성은 유지된다. 근거는 `src/models/CLAUDE.md`.

## Claude Code 세팅 (`.claude/`)

규칙이 글로만 있으면 언젠가 지나친다. 아래는 그걸 **강제하는 장치**다.

| 무엇 | 어디 | 하는 일 |
|---|---|---|
| 훅 | `guard_bash.sh` | `KIWOOM_ENV=live` **차단**(규칙 1) / 저장소 밖 git 명령 차단 / `paper_trade --execute` 확인(규칙 9) |
| 훅 | `guard_edit.sh` | `.env` 편집 차단(규칙 2) / `live` 대입 차단 / `config.py`·`signal.py`·`risk.py`·`inference.py`·`configs/config.yaml` 편집 시 확인(규칙 1·7) |
| 훅 | `ruff_check.sh` | 파이썬 편집 직후 ruff. 막지 않고 결과만 돌려준다 |
| 훅 | `test_before_commit.sh` | `git commit` 전에 pytest. 깨졌으면 **커밋 거부** |
| 훅 | `session_brief.sh` | 세션 시작 시 패널 최신성·실거래 현황·리밸런싱 주기·수집 정상 여부를 한 줄로 |
| 스킬 | `/brief` | 세션 시작 브리핑 |
| 스킬 | `/lookahead` | 피처·분할·백테스트를 건드린 뒤 도는 look-ahead 감사(규칙 5·6) |
| 에이전트 | `backtest-auditor` | 매매 코드 변경을 독립 감사 — look-ahead + 단일 구현(규칙 7) |

훅을 새로 넣거나 고친 뒤에는 `/hooks` 를 한 번 열어야 적용된다.

> ⚠️ **이 장치들은 Claude Code 에서만 돈다.** 다른 에이전트(Codex 등)로 작업할 때는
> 아무것도 막히지 않는다 — 훅이 막던 것을 손으로 지켜야 한다. 그 차이는
> [`AGENTS.md`](AGENTS.md) 에 정리해두었다. 규칙 자체는 이 문서가 정본이다.

## 개발 환경

- 파이썬 3.11, 가상환경 `.venv/`, 의존성은 `requirements.txt` 단일 관리
- Mac (MPS 백엔드), VS Code + Claude Code
- 설정은 `configs/config.yaml` 단일 소스 — 코드에 숫자를 하드코딩하지 않는다
