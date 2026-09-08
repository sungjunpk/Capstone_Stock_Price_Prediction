"""static covariate 의 look-ahead 감사 (2026-09-08).

`market_cap_bucket` 이 **조회시점** 시총으로 계산되고 있었다. static 은 종목당 한 값이라
2015년 행까지 "이 종목이 나중에 커진다"를 들고 있었다 — 전형적인 hindsight bias 다.

실측: 201종목 중 **106종목(56%)의 시총구간이 바뀌었다.** 효성중공업은 train_end 시점
182위인데 현재 26위다. 무해한 차이가 아니었다.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from src.features.build import _market_cap_at

TRAIN_END = date(2022, 12, 31)


def _panel(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"code": c, "date": date.fromisoformat(d), "close": px} for c, d, px in rows]
    )


def test_price_move_after_train_end_does_not_inflate_size():
    """train_end 이후에 오른 종목이 그때 더 컸던 것으로 취급되면 안 된다.

    A 와 B 는 train_end 시점 종가가 같고 주식수도 같다 = 그때 시총이 같다.
    A 만 그 뒤 10배 올랐다. 고치기 전 코드는 A 를 10배 큰 종목으로 봤다.
    """
    panel = _panel([
        ("A", "2022-12-30", 100.0), ("A", "2026-09-08", 1000.0),   # 10배
        ("B", "2022-12-30", 100.0), ("B", "2026-09-08", 100.0),    # 그대로
    ])
    # 현재 시총 = 주식수(1,000주 동일) x 현재가
    info = pd.DataFrame({"code": ["A", "B"], "market_cap": [1_000_000.0, 100_000.0]})

    cap = _market_cap_at(info, panel, TRAIN_END)
    assert cap["A"] == cap["B"], "train_end 시점 시총이 같아야 한다"


def test_uses_last_close_at_or_before_train_end():
    """train_end 당일이 휴장이면 그 이전 마지막 거래일 종가를 쓴다. 이후는 안 본다."""
    panel = _panel([
        ("A", "2022-12-29", 50.0),
        ("A", "2022-12-30", 60.0),   # train_end 이전 마지막 → 이 값을 써야 한다
        ("A", "2023-01-02", 900.0),  # train_end 이후 — 봐서는 안 된다
        ("A", "2026-09-08", 120.0),
    ])
    info = pd.DataFrame({"code": ["A"], "market_cap": [1200.0]})  # 주식수 10주

    cap = _market_cap_at(info, panel, TRAIN_END)
    assert cap["A"] == 600.0          # 10주 x 60원, 900원은 안 봤다


def test_not_listed_at_train_end_is_na():
    """train_end 시점에 상장 전이던 종목은 '규모 미상'이다.

    억지로 값을 만들면 그 종목의 2024년 규모가 2015년 학습에 새어든다.
    NA 로 두면 코드북에서 0번(미등록) 슬롯으로 떨어진다.
    """
    panel = _panel([("A", "2024-05-08", 100.0), ("A", "2026-09-08", 200.0)])
    info = pd.DataFrame({"code": ["A"], "market_cap": [200.0]})

    cap = _market_cap_at(info, panel, TRAIN_END)
    assert pd.isna(cap["A"])


def test_ranking_follows_train_end_not_today():
    """순위가 현재가 아니라 train_end 시점을 따른다 — 이게 고친 것의 전부다."""
    panel = _panel([
        ("큰놈", "2022-12-30", 200.0), ("큰놈", "2026-09-08", 200.0),
        ("뜬놈", "2022-12-30", 10.0), ("뜬놈", "2026-09-08", 500.0),
    ])
    info = pd.DataFrame({"code": ["큰놈", "뜬놈"], "market_cap": [200.0, 500.0]})

    cap = _market_cap_at(info, panel, TRAIN_END)
    # 현재 시총은 '뜬놈'이 크지만, train_end 시점엔 '큰놈'이 컸다
    assert cap["큰놈"] > cap["뜬놈"]
