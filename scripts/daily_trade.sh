#!/bin/bash
# 평일 15:15 자동 매매. launchd 가 호출한다.
#
#   설치:   bash scripts/install_daily_trade.sh
#   해제:   bash scripts/install_daily_trade.sh --uninstall
#   로그:   outputs/logs/daily/trade_YYYY-MM-DD.log
#
# ⚠️ 이 스크립트는 **실제 모의투자 주문을 낸다.**
#    주문 경로는 여전히 scripts/paper_trade.py --execute 하나다 —
#    자동화가 새 주문 경로를 만들지 않는다(CLAUDE.md 절대 규칙 9).

set -uo pipefail

PROJECT="/Users/mac/Desktop/Capstone_Stock_Price_Prediction"
PY="$PROJECT/.venv/bin/python"
LOGDIR="$PROJECT/outputs/logs/daily"

mkdir -p "$LOGDIR"
LOG="$LOGDIR/trade_$(date +%F).log"

exec >>"$LOG" 2>&1
echo "=========== $(date '+%F %T') 매매 시작 ==========="

cd "$PROJECT" || exit 1

# 휴장일이면 여기서 끝. 공휴일 테이블 대신 지수 일봉에 물어본다.
if ! "$PY" -c "
import sys
from datetime import date
sys.path.insert(0, '$PROJECT')
from src.data.kiwoom.client import KiwoomClient
from src.data.kiwoom.collect import is_trading_day
with KiwoomClient() as c:
    sys.exit(0 if is_trading_day(c, date.today()) else 1)
"; then
    echo "휴장일 — 주문 없이 종료"
    exit 0
fi

echo "--- 주문 전송 ---"
"$PY" scripts/paper_trade.py --execute || { echo "매매 실패"; exit 1; }

echo "=========== $(date '+%F %T') 매매 완료 ==========="
