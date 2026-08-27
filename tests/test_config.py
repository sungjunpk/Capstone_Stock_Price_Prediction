"""설정 프로파일 — 분봉 트랙이 일봉 트랙을 건드리지 않는다는 보장.

프로파일은 **차이만** 적는다. base 를 덮어쓰는 범위가 넓어지면
"어느 값이 진짜인가"를 알 수 없게 되므로 여기서 경계를 지킨다.
"""

from __future__ import annotations

import pytest

from src.utils.config import _deep_merge, load_config


def test_deep_merge_keeps_untouched_branches():
    base = {"a": {"x": 1, "y": 2}, "b": 3, "list": [1, 2]}
    over = {"a": {"y": 9}, "list": [7]}
    out = _deep_merge(base, over)
    assert out == {"a": {"x": 1, "y": 9}, "b": 3, "list": [7]}
    assert base["a"]["y"] == 2, "원본을 변형하면 안 된다"


def test_default_load_has_no_profile_applied():
    """프로파일을 안 주면 일봉 트랙 값 그대로여야 한다."""
    cfg = load_config()
    assert cfg["features"]["return_horizon"] == 5
    assert cfg["data"].get("chart_kind") is None
    assert cfg.get("active_profile") is None


def test_intraday_profile_overrides_only_what_it_declares():
    base, intra = load_config(), load_config(profile="intraday")

    assert intra["data"]["chart_kind"] == "minute60"
    assert intra["features"]["return_horizon"] == 7
    assert intra.get("active_profile") == "intraday"

    # 건드리지 않은 가지는 base 와 같아야 한다
    assert intra["model"] == base["model"]
    assert intra["trading"]["costs"] == base["trading"]["costs"]
    assert intra["data"]["universe"] == base["data"]["universe"]

    # 선언한 가지 안에서도 **선언하지 않은 키는 살아남아야 한다** (얕은 교체 금지)
    assert intra["trading"]["risk"]["max_holding_bars"] == 7          # 프로파일이 추가
    assert (intra["trading"]["risk"]["max_gross_exposure"]
            == base["trading"]["risk"]["max_gross_exposure"])          # base 유지


def test_unknown_profile_is_refused():
    with pytest.raises(ValueError, match="알 수 없는 프로파일"):
        load_config(profile="없는프로파일")


def test_mktrel_profile_only_changes_the_target_definition():
    """대조군이 성립하려면 타깃과 저장경로 말고는 base 와 같아야 한다."""
    base = load_config().raw
    mr = load_config(profile="mktrel").raw

    assert mr["features"]["target_mode"] == "market_relative"
    assert base["features"]["target_mode"] == "raw"
    assert mr["data"]["processed_suffix"] == "_mr"

    # 모델·학습·매매 규칙은 손대지 않는다
    assert mr["model"] == base["model"]
    assert mr["training"] == base["training"]
    assert mr["trading"] == base["trading"]
    assert mr["backtest"] == base["backtest"]
    # features 도 target_mode 외에는 동일해야 한다
    assert {k: v for k, v in mr["features"].items() if k != "target_mode"} == \
           {k: v for k, v in base["features"].items() if k != "target_mode"}
