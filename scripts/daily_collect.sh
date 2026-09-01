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

# 결과를 알림으로 띄운다. 대시보드를 열어봐야만 알 수 있으면 실패를 모르고 지나간다.
# 알림 권한이 막혀 있어도 수집 자체는 계속 가야 하므로 실패를 삼킨다.
notify() {   # $1=제목 $2=본문
    osascript -e "display notification \"$2\" with title \"$1\"" >/dev/null 2>&1 || true
}

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

# 도는 동안 맥이 잠들지 못하게 붙잡는다. 배터리에서는 1분만 유휴해도 잠드는데,
# 수집 중 잠들면 그대로 중단됐다가 깨어날 때 이어서 돈다 — 2026-08-28 에 4분짜리
# 수집이 그렇게 3일에 걸쳐 끌렸고, 그 사이 토큰이 두 번 죽었다.
# -w $$ 로 이 스크립트가 끝나면 같이 풀린다. 화면은 계속 꺼져도 된다(-i = 유휴수면만).
caffeinate -i -w $$ &

# 장 마감 후 확정 총자산을 하루 한 줄 남긴다. 거래가 없는 날에도 찍어야
# 누적 수익률 곡선에 구멍이 안 생긴다. 읽기 전용 — 주문 경로 없음.
#
# ⚠️ 수집보다 **먼저** 돈다. 예전엔 맨 뒤에 있었는데, 2026-08-28 에 일봉 수집이
#    2종목 인증 실패로 exit 1 하면서 스냅샷까지 통째로 건너뛰었다. 그날 곡선에
#    구멍이 났고 대시보드가 증권사와 어긋났다. 스냅샷은 계좌만 조회하므로
#    수집·피처와 아무 의존이 없다 — 남의 실패에 끌려갈 이유가 없다.
echo "--- 계좌 스냅샷 ---"
"$PY" scripts/snapshot_account.py || echo "스냅샷 실패 (수집은 계속한다)"

# 장중에 --end-date 없이 받으면 **오늘의 미완성 일봉**이 종가 자리에 들어간다.
# 이 스크립트는 16:00 에 도니 평소엔 문제없지만, launchd 는 놓친 실행을 깨어난
# 뒤에 띄우고 그게 다음날 장 마감 전일 수 있다 — 2026-08-31 에 그렇게 75종목이
# 거래량 0짜리 가짜 봉으로 오염됐다. 마감 전이면 전 거래일까지만 받는다.
# 밀린 날짜는 그대로 메우면서 당일 봉만 빠진다.
END_ARG=""
HM=$((10#$(date +%H%M)))
# 09:00 부터가 아니라 **마감 전 전체**를 막는다. 키움은 개장 전에도 당일 봉을
# 내보내는데 거래량 0 에 OHLC 가 전부 전일 종가다 — 8/31 05·07·08시에 받은
# 파일들이 정확히 그 모양이었다. 15:35 이후에만 당일 봉이 확정이다.
if [ "$HM" -lt 1535 ]; then
    # 전 거래일 = 직전 평일. 공휴일은 따지지 않는다 — collect.py 가 end-date
    # **이하**만 남기므로(collect.py 의 df["date"] <= end_date) 휴일을 줘도
    # 그 이전 마지막 봉까지 들어온다.
    PREV=$(date -j -v-1d -f "%Y-%m-%d" "$(date +%F)" "+%Y-%m-%d")
    while [ "$(date -j -f "%Y-%m-%d" "$PREV" "+%u")" -ge 6 ]; do
        PREV=$(date -j -v-1d -f "%Y-%m-%d" "$PREV" "+%Y-%m-%d")
    done
    echo "마감 전($HM) 실행 — 당일 봉을 막으려 --end-date $PREV 로 받는다"
    END_ARG="--end-date $PREV"
fi

echo "--- 일봉 수집 ---"
"$PY" scripts/collect.py --tr chart $END_ARG
RC=$?
# collect.py 종료코드: 0 전부 성공 / 1 일부 종목 실패 / 2 수집 실패
# 1 에서 멈추지 않는다 — 2026-08-28 에 149종목 중 2종목이 토큰 만료로 깨졌다고
# 피처 재생성이 통째로 스킵됐다. 증분이라 빠진 종목은 다음 실행이 메운다.
if [ "$RC" -ge 2 ]; then
    echo "수집 실패 (exit $RC)"
    notify "수집 실패" "일봉 수집이 엎어졌다 — 로그 $(date +%F).log"
    exit 1
fi
if [ "$RC" -eq 1 ]; then
    echo "일부 종목 실패 — 다음 실행이 메운다. 피처는 계속 만든다"
fi

echo "--- 피처 재생성 ---"
if ! "$PY" scripts/build_features.py; then
    echo "피처 생성 실패"
    notify "피처 생성 실패" "수집은 됐지만 피처가 안 만들어졌다 — 로그 $(date +%F).log"
    exit 1
fi

SUMMARY=$(grep -oE "완료: 성공 [0-9]+ / 실패 [0-9]+" "$LOG" | tail -1)
notify "수집 완료" "${SUMMARY:-일봉 수집·피처 갱신 완료}"
echo "=========== $(date '+%F %T') 완료 ==========="
