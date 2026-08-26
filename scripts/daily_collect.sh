#!/bin/bash
# 매일 장 마감 후 도는 수집 + 피처 갱신. launchd 가 호출한다.
#
#   설치:   bash scripts/install_daily_collect.sh
#   로그:   outputs/logs/daily/YYYY-MM-DD.log
#   해제:   launchctl bootout gui/$(id -u)/com.capstone.stock.collect
#
# 주문은 여기서 내지 않는다. 자동 매매는 scripts/daily_trade.sh 가 15:15 에 따로 돈다 —
# 주문을 내는 자동화는 스스로 켜는 동작이어야 해서 설치를 분리했다.

set -uo pipefail

PROJECT="/Users/mac/Desktop/Capstone_Stock_Price_Prediction"
PY="$PROJECT/.venv/bin/python"
LOGDIR="$PROJECT/outputs/logs/daily"

mkdir -p "$LOGDIR"
LOG="$LOGDIR/$(date +%F).log"

exec >>"$LOG" 2>&1
echo "=========== $(date '+%F %T') 시작 ==========="

# 맥이 자고 있었으면 launchd 가 깨어난 뒤 밀린 실행을 한 번 돌린다.
# 그게 주말에 걸릴 수 있어 여기서 막는다 (증분이라 무해하지만 로그가 지저분해진다).
DOW=$(date +%u)   # 1=월 ... 7=일
if [ "$DOW" -ge 6 ]; then
    echo "주말($DOW) — 건너뛴다"
    exit 0
fi

cd "$PROJECT" || exit 1

echo "--- 일봉 수집 ---"
"$PY" scripts/collect.py --tr chart || { echo "수집 실패"; exit 1; }

echo "--- 피처 재생성 ---"
"$PY" scripts/build_features.py || { echo "피처 생성 실패"; exit 1; }

# 장 마감 후 확정 총자산을 하루 한 줄 남긴다. 거래가 없는 날에도 찍어야
# 누적 수익률 곡선에 구멍이 안 생긴다. 읽기 전용 — 주문 경로 없음.
echo "--- 계좌 스냅샷 ---"
"$PY" scripts/snapshot_account.py || echo "스냅샷 실패 (수집은 성공했다)"

echo "=========== $(date '+%F %T') 완료 ==========="
