"""RevIN 우회 경로 — 횡단면 피처가 모델까지 살아서 도착하는가.

이게 깨지면 모델은 '오늘 시장에서 몇 등인가'를 못 본다. 조용히 성능만 나빠지므로
테스트로 못 박아 둔다.
"""

from __future__ import annotations

import pandas as pd
import pytest
import torch

from src.models.phase1 import Phase1Config, Phase1Model
from src.training.dataset import dynamic_feature_columns, n_passthrough_columns


def _cfg(n_dynamic: int, n_passthrough: int) -> Phase1Config:
    return Phase1Config(
        n_dynamic=n_dynamic, n_macro=3,
        static_vocab={"sector": 4, "size_class": 3, "market_cap_bucket": 6,
                      "day_of_week": 6},
        lookback=20, patch_len=5, stride=5, d_model=16, n_heads=2, n_layers=1,
        d_ff=32, n_passthrough=n_passthrough,
    )


class TestColumnOrder:
    def test_xs_columns_are_pushed_to_the_end(self):
        panel = pd.DataFrame(columns=["code", "date", "close", "xs_rsi", "rsi",
                                      "xs_ret_5d", "atr", "target"])
        cols = dynamic_feature_columns(panel)
        assert cols == ["rsi", "atr", "xs_rsi", "xs_ret_5d"]
        assert n_passthrough_columns(cols) == 2

    def test_no_xs_columns_means_no_passthrough(self):
        panel = pd.DataFrame(columns=["code", "date", "close", "rsi", "target"])
        assert n_passthrough_columns(dynamic_feature_columns(panel)) == 0


class TestPassthrough:
    def test_passthrough_channels_survive_untouched(self):
        """뒤쪽 채널은 RevIN 을 통과하지 않고 값이 그대로 남아야 한다."""
        m = Phase1Model(_cfg(5, 2)).eval()
        x = torch.randn(2, 20, 5)
        seen = {}
        orig = m.patch_dyn.forward

        def spy(t):
            seen["x"] = t.clone()
            return orig(t)

        m.patch_dyn.forward = spy
        m(x, torch.randn(2, 20, 3), torch.zeros(2, 4, dtype=torch.long))

        # 앞 3채널은 정규화되어 창 평균이 0 근처
        assert seen["x"][..., :3].mean().abs() < 0.5
        # 뒤 2채널은 입력 그대로
        torch.testing.assert_close(seen["x"][..., 3:], x[..., 3:])

    def test_revin_only_sizes_the_normalised_channels(self):
        m = Phase1Model(_cfg(5, 2))
        assert m.n_revin == 3
        assert m.revin_dyn.n_features == 3

    def test_zero_passthrough_keeps_old_behaviour(self):
        m = Phase1Model(_cfg(5, 0))
        assert m.n_revin == 5 and m.revin_dyn.n_features == 5

    def test_all_channels_passthrough_is_refused(self):
        with pytest.raises(ValueError, match="RevIN 채널이 없다"):
            Phase1Model(_cfg(3, 3))
