#!/usr/bin/env bash
# 파일 편집으로 절대 규칙이 풀리는 것을 막는다.
set -uo pipefail

input=$(cat)
path=$(printf '%s' "$input" | jq -r '.tool_input.file_path // ""')
body=$(printf '%s' "$input" | jq -r \
  '[.tool_input.content, .tool_input.new_string] | map(select(. != null)) | join("\n")')

emit() {
  jq -Rn --arg d "$1" --arg s "$2" \
    '{hookSpecificOutput:{hookEventName:"PreToolUse",
      permissionDecision:$d, permissionDecisionReason:$s}}'
  exit 0
}

# --- 규칙 2: 비밀값 -----------------------------------------------------------
case "$path" in
  */.env|*/.env.local)
    emit deny "절대 규칙 2 — .env 는 코드로 건드리지 않는다. 값이 대화·커밋에 남는다." ;;
esac

# --- 규칙 1: live 대입 --------------------------------------------------------
# ⚠️ 줄 맨 앞의 대입만 본다. config.py 의 에러 메시지 안에 있는 같은 문자열은
#    앞에 따옴표가 있어 걸리지 않는다 (오탐 방지).
if printf '%s' "$body" | grep -Eq "^[[:space:]]*(export[[:space:]]+)?KIWOOM_ENV[[:space:]]*=[[:space:]]*[\"']?live"; then
  emit deny "절대 규칙 1 위반 — KIWOOM_ENV=live 대입."
fi

# --- 방어선 자체를 건드릴 때는 사람이 본다 ------------------------------------
case "$path" in
  */src/utils/config.py)
    emit ask "config.py 는 KIWOOM_ENV=live 를 막는 방어선이다 (규칙 1). 그 방어를 푸는 변경이 아닌지 확인할 것." ;;
  */src/trading/signal.py|*/src/models/inference.py)
    emit ask "규칙 7 — 백테스트와 모의투자가 공유하는 단일 구현이다. 여기 바꾸면 진행 중인 실거래 기록(8/26~)의 체제가 바뀐다." ;;
esac

exit 0
