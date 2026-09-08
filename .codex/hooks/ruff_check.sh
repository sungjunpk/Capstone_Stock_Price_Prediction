#!/usr/bin/env bash
# 편집한 파이썬 파일만 즉시 린트한다. 막지 않고 결과만 돌려준다.
set -uo pipefail

f=$(cat | jq -r '.tool_input.file_path // .tool_response.filePath // ""')
case "$f" in *.py) ;; *) exit 0 ;; esac
[ -f "$f" ] || exit 0

root="${CLAUDE_PROJECT_DIR:-$PWD}"
[ -x "$root/.venv/bin/ruff" ] || exit 0

if ! out=$("$root/.venv/bin/ruff" check --quiet "$f" 2>&1); then
  jq -Rn --arg s "$out" \
    '{hookSpecificOutput:{hookEventName:"PostToolUse",additionalContext:("ruff:\n"+$s)}}'
fi
exit 0
