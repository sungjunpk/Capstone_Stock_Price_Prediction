#!/usr/bin/env bash
# 커밋 직전에 테스트를 돌린다. 깨진 채로 커밋되면 이력이 거짓말을 한다.
# CLAUDE.md: "커밋 메시지가 곧 작업 이력이다" — 이력이 믿을 만하려면 그 시점이
# 실제로 통과 상태여야 한다.
set -uo pipefail

cmd=$(cat | jq -r '.tool_input.command // ""')
printf '%s' "$cmd" | grep -Eq '(^|[;&|[:space:]])git[[:space:]]+commit' || exit 0

root="${CLAUDE_PROJECT_DIR:-$PWD}"
[ -x "$root/.venv/bin/python" ] || exit 0

# 코드가 안 바뀐 커밋(문서만)에는 돌리지 않는다 — 8초를 낭비할 이유가 없다.
staged=$(git -C "$root" diff --cached --name-only 2>/dev/null)
printf '%s' "$staged" | grep -Eq '^(src|scripts|tests)/.*\.py$' || exit 0

if ! out=$(cd "$root" && .venv/bin/python -m pytest -q 2>&1); then
  jq -Rn --arg s "$(printf '%s' "$out" | tail -25)" \
    '{hookSpecificOutput:{hookEventName:"PreToolUse",permissionDecision:"deny",
      permissionDecisionReason:("테스트가 깨진 상태로 커밋하려 한다. 고치고 다시 커밋할 것.\n\n"+$s)}}'
  exit 0
fi
exit 0
