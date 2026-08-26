#!/bin/bash
# 자동 매매 설치/해제 — ⚠️ 실제 모의투자 주문을 낸다.
#
#   bash scripts/install_daily_trade.sh            # 설치
#   bash scripts/install_daily_trade.sh --uninstall # 해제
#   bash scripts/install_daily_trade.sh --status    # 등록 상태 확인
#   bash scripts/install_daily_trade.sh --now       # 지금 한 번 강제 실행

set -euo pipefail

LABEL="com.capstone.stock.trade"
PROJECT="/Users/mac/Desktop/Capstone_Stock_Price_Prediction"
SRC="$PROJECT/scripts/$LABEL.plist"
DST="$HOME/Library/LaunchAgents/$LABEL.plist"
DOMAIN="gui/$(id -u)"

case "${1:-}" in
  --uninstall)
    launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null || true
    rm -f "$DST"
    echo "해제됨: $LABEL"
    ;;
  --status)
    launchctl print "$DOMAIN/$LABEL" 2>/dev/null | head -20 || echo "등록 안 됨"
    ;;
  --now)
    launchctl kickstart -p "$DOMAIN/$LABEL"
    echo "지금 실행. 로그: outputs/logs/daily/trade_$(date +%F).log"
    ;;
  *)
    mkdir -p "$HOME/Library/LaunchAgents" "$PROJECT/outputs/logs/daily"
    cp "$SRC" "$DST"
    launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null || true
    launchctl bootstrap "$DOMAIN" "$DST"
    echo "설치됨: $LABEL — 평일 15:15 — ⚠️ 실제 주문이 나간다"
    echo "  확인: bash scripts/install_daily_trade.sh --status"
    echo "  로그: outputs/logs/daily/"
    ;;
esac
