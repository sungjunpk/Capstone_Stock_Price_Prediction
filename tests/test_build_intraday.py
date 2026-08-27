"""분봉 패널 조립 — 봉 종류가 바뀌어도 라벨 규약이 그대로인지.

여기서 지키려는 것:
  1) raw 의 'datetime' 이 패널에서 'date' 가 된다 (하류가 봉 종류를 몰라도 되게)
  2) 타깃은 **뒤쪽 h봉**만 본다 — 봉 단위가 바뀌어도 look-ahead 규약은 같다
  3) 일 단위 수급이 봉 단위 패널에 섞이지 않는다
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data import storage
from src.features.build import build_panel


@pytest.fixture
def minute_raw(tmp_path, monkeypatch):
    """2종목 × 400봉짜리 가짜 60분봉 raw 를 만든다."""
    monkeypatch.setattr(storage, "RAW_DIR", tmp_path)
    rng = np.random.default_rng(0)
    stamps = pd.date_range("2026-01-02 09:00", periods=400, freq="h")

    for code in ("000001", "000002"):
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(stamps))))
        df = pd.DataFrame({
            "datetime": stamps,
            "open": close * 0.999, "high": close * 1.01,
            "low": close * 0.99, "close": close,
            "volume": rng.integers(1000, 9999, len(stamps)),
        })
        storage.upsert(df, storage.raw_path("minute60", code),
                       key=["datetime"], sort_by=["datetime"])

    return {
        "data": {
            "chart_kind": "minute60",
            "universe": [{"code": "000001", "sector": "반도체"},
                         {"code": "000002", "sector": "화학"}],
        },
        "features": {"return_horizon": 7, "technical": {}},
    }


def test_datetime_becomes_the_date_column(minute_raw):
    panel = build_panel(minute_raw)
    assert "datetime" not in panel.columns
    d = pd.to_datetime(panel["date"])
    assert d.dt.hour.nunique() > 1, "시각 정보가 날짜로 뭉개지면 안 된다"


def test_target_is_forward_return_over_h_bars(minute_raw):
    panel = build_panel(minute_raw)
    one = panel[panel.code == "000001"].sort_values("date").reset_index(drop=True)

    i = 100
    expected = np.log(one.close.iloc[i + 7] / one.close.iloc[i])
    assert one.target.iloc[i] == pytest.approx(expected)
    # 마지막 h봉은 미래가 없으므로 결측이어야 한다 — 여기가 채워져 있으면 look-ahead 다
    assert one.target.tail(7).isna().all()


def test_daily_flow_is_not_joined_to_bar_panel(minute_raw):
    panel = build_panel(minute_raw)
    assert not [c for c in panel.columns if c.startswith("flow_")]


# --------------------------------------------------- 시장 대비 초과수익 타깃
def test_market_relative_target_removes_the_daily_cross_section_mean():
    from src.features.build import _apply_target_mode

    panel = pd.DataFrame({
        "date": ["d1", "d1", "d1", "d2", "d2", "d2"],
        "code": ["a", "b", "c", "a", "b", "c"],
        "target": [0.03, 0.01, -0.01, -0.02, 0.00, 0.02],
    })
    out = _apply_target_mode(panel.copy(), {"target_mode": "market_relative"})

    # 날짜별 평균이 0 이 된다 = 시장 공통성분이 사라졌다
    assert out.groupby("date")["target"].mean().abs().max() < 1e-12
    # 같은 날 안에서의 **순서**는 보존된다 — 순위 전략이 쓰는 정보다
    assert list(out[out.date == "d1"]["target"].rank()) == [3.0, 2.0, 1.0]


def test_raw_target_mode_is_the_default_and_changes_nothing():
    from src.features.build import _apply_target_mode

    panel = pd.DataFrame({"date": ["d1", "d1"], "code": ["a", "b"],
                          "target": [0.03, 0.01]})
    assert _apply_target_mode(panel.copy(), {})["target"].tolist() == [0.03, 0.01]


def test_unknown_target_mode_is_refused():
    from src.features.build import _apply_target_mode
    with pytest.raises(ValueError, match="target_mode"):
        _apply_target_mode(pd.DataFrame({"date": ["d"], "target": [0.1]}),
                           {"target_mode": "oops"})


# --------------------------------------------------- 횡단면 피처
def test_cross_sectional_features_are_within_date_ranks():
    from src.features.build import _add_cross_sectional

    panel = pd.DataFrame({
        "date": ["d1"] * 4 + ["d2"] * 4,
        "code": list("abcd") * 2,
        "ret_5d": [0.10, 0.05, -0.05, -0.10, -0.10, -0.05, 0.05, 0.10],
    })
    out = _add_cross_sectional(panel.copy(), {"cross_sectional": ["ret_5d"]})

    # 순위는 그 날 안에서만 매겨진다 — d1 의 a 는 1등, d2 의 a 는 꼴찌
    d1 = out[out.date == "d1"].set_index("code")["xs_ret_5d"]
    d2 = out[out.date == "d2"].set_index("code")["xs_ret_5d"]
    assert d1["a"] == pytest.approx(0.5) and d1["d"] == pytest.approx(-0.25)
    assert d2["a"] == pytest.approx(-0.25) and d2["d"] == pytest.approx(0.5)
    # 원본은 건드리지 않는다
    assert out["ret_5d"].tolist() == panel["ret_5d"].tolist()


def test_cross_sectional_is_off_by_default():
    from src.features.build import _add_cross_sectional
    panel = pd.DataFrame({"date": ["d"], "ret_5d": [0.1]})
    assert list(_add_cross_sectional(panel.copy(), {}).columns) == ["date", "ret_5d"]


def test_missing_cross_sectional_source_is_refused():
    from src.features.build import _add_cross_sectional
    with pytest.raises(ValueError, match="없다"):
        _add_cross_sectional(pd.DataFrame({"date": ["d"]}),
                             {"cross_sectional": ["nope"]})
