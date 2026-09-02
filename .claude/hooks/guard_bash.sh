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

# --- 홈 디렉토리는 그 자체가 git 저장소다 (SSH 키가 커밋된 전례) ------------
if printf '%s' "$cmd" | grep -q 'git add'; then
  if [ "$PWD" = "$HOME" ] \
     || printf '%s' "$cmd" | grep -Eq 'cd[[:space:]]+(~|\$HOME|/Users/mac)([[:space:]]|;|&|$)' \
     || printf '%s' "$cmd" | grep -Eq 'git[[:space:]]+-C[[:space:]]+(~|\$HOME|/Users/mac)([[:space:]]|$)'; then
    emit deny "홈 디렉토리(/Users/mac)는 그 자체가 git 저장소다 — 여기서 git add 하면 SSH 키가 커밋된다. 프로젝트 폴더에서 실행할 것."
  fi
fi

# --- 규칙 9: 주문은 사람이 확인한다 -----------------------------------------
if printf '%s' "$cmd" | grep -q 'paper_trade' && printf '%s' "$cmd" | grep -q -- '--execute'; then
  emit ask "⚠️ 실제 모의투자 주문이 나간다 (절대 규칙 9). --execute 없이 계획을 먼저 확인했는지 볼 것."
fi

exit 0
