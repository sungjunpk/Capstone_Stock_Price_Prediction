"""체크포인트 자동 선택 — 트랙이 섞이면 **조용히 틀린다**.

일봉 트랙과 60분봉 트랙이 같은 디렉터리를 쓴다. '가장 최근 것'으로 고르면
분봉 모델로 일봉 예측을 하고, 그 결과가 에러가 아니라 그럴듯한 숫자로 나온다.
`paper_trade.py` 쪽은 그게 **실주문**이 된다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import backtest as bt  # noqa: E402
from scripts import paper_trade as pt  # noqa: E402
from src.utils.config import load_config  # noqa: E402


@pytest.fixture
def ckpts(tmp_path, monkeypatch):
    """일봉 것보다 분봉 것이 **더 최신**인 상황 — 실제로 이렇게 만들어졌다."""
    (tmp_path / "phase1_0db568ae.pt").write_bytes(b"daily")
    (tmp_path / "phase1_f2a3026d_60m.pt").write_bytes(b"intraday")
    (tmp_path / "phase1_f2a3026d_60m_smoke.pt").write_bytes(b"smoke")
    import os
    import time
    now = time.time()
    os.utime(tmp_path / "phase1_0db568ae.pt", (now - 100, now - 100))
    monkeypatch.setattr(bt, "CKPT_DIR", tmp_path)
    monkeypatch.setattr(pt, "CKPT_DIR", tmp_path)
    return tmp_path


def test_daily_backtest_never_picks_an_intraday_checkpoint(ckpts):
    cfg = load_config().raw
    assert bt.find_checkpoint(None, cfg).name == "phase1_0db568ae.pt"


def test_intraday_backtest_picks_its_own(ckpts):
    cfg = load_config(profile="intraday").raw
    assert bt.find_checkpoint(None, cfg).name == "phase1_f2a3026d_60m.pt"


def test_order_path_is_restricted_to_the_daily_track(ckpts):
    """실주문 경로가 다른 트랙 모델을 집으면 잘못된 주문이 나간다."""
    assert pt.find_checkpoint(None).name == "phase1_0db568ae.pt"


def test_missing_track_checkpoint_fails_loudly(tmp_path, monkeypatch):
    """없으면 조용히 남의 것을 쓰지 말고 멈춰야 한다."""
    (tmp_path / "phase1_0db568ae.pt").write_bytes(b"daily")
    monkeypatch.setattr(bt, "CKPT_DIR", tmp_path)
    with pytest.raises(SystemExit):
        bt.find_checkpoint(None, load_config(profile="intraday").raw)
