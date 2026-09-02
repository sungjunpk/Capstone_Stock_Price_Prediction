#!/usr/bin/env bash
# CLAUDE.md 절대 규칙을 명령 실행 **전에** 강제한다.
# 규칙이 글로만 있으면 언젠가 지나친다 — 여기서 막는다.
set -uo pipefail

cmd=$(cat | jq -r '.tool_input.command // ""')

emit() {  # $1=deny|ask  $2=사유
  jq -Rn --arg d "$1" --arg s "$2" \
    '{hookSpecificOutput:{hookEventName:"PreToolUse",
      permissionDecision:$d, permissionDecisionReason:$s}}'
  exit 0
}

# --- 규칙 1: 실전 환경 금지 -------------------------------------------------
if printf '%s' "$cmd" | grep -Eq "(^|[;&| ])(export[[:space:]]+)?KIWOOM_ENV[[:space:]]*=[[:space:]]*[\"']?live"; then
  emit deny "절대 규칙 1 위반 — KIWOOM_ENV=live. 이 프로젝트는 mock 만 쓴다."
fi

# --- git 은 이 프로젝트 저장소 안에서만 -------------------------------------
# 홈(/Users/mac)에도 **별개의** git 저장소가 있다 — sungjunhouse 연습용 28커밋.
# 내용물은 859바이트인데 .git 이 15GB 다: 홈에서 git add 를 한 번 쓸어서
# loose object 가 59,892개 쌓였다. 캡스톤 작업이 그쪽에 닿을 이유가 전혀 없다.
PROJ="${CLAUDE_PROJECT_DIR:-/Users/mac/Desktop/Capstone_Stock_Price_Prediction}"
if printf '%s' "$cmd" | grep -Eq '(^|[;&|[:space:]])git([[:space:]]|$)'; then
  # 명령이 대놓고 홈을 가리키는 경우
  if printf '%s' "$cmd" | grep -Eq 'git[[:space:]]+-C[[:space:]]+(~|\$HOME|/Users/mac)([[:space:]]|$)' \
     || printf '%s' "$cmd" | grep -Eq 'cd[[:space:]]+(~|\$HOME|/Users/mac)([[:space:]]|;|&|$)'; then
    emit deny "홈(/Users/mac)에는 캡스톤과 무관한 별개 git 저장소가 있다. git 은 프로젝트 저장소에서만 쓴다."
  fi
  # 프로젝트 밖에서 도는 경우
  case "$PWD" in
    "$PROJ"|"$PROJ"/*) ;;
    *) emit deny "여기는 프로젝트 저장소 밖이다 ($PWD). git 은 $PROJ 안에서만 쓴다." ;;
  esac
fi

# --- 규칙 9: 주문은 사람이 확인한다 -----------------------------------------
if printf '%s' "$cmd" | grep -q 'paper_trade' && printf '%s' "$cmd" | grep -q -- '--execute'; then
  emit ask "⚠️ 실제 모의투자 주문이 나간다 (절대 규칙 9). --execute 없이 계획을 먼저 확인했는지 볼 것."
fi

exit 0
