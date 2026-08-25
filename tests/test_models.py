"""모델 shape 스모크 + 계약 검증."""

import pytest
import torch

from src.models.patch_embed import PatchEmbedding, num_patches
from src.models.phase1 import Phase1Config, Phase1Model
from src.models.quantile_head import QuantileHead
from src.models.revin import RevIN

VOCAB = {"sector": 22, "size_class": 3, "market_cap_bucket": 6, "day_of_week": 6}


@pytest.fixture
def model():
    return Phase1Model(Phase1Config(n_dynamic=17, n_macro=13, static_vocab=VOCAB))


def _batch(b=4, lookback=120):
    return (
        torch.randn(b, lookback, 17),
        torch.randn(b, lookback, 13),
        torch.randint(0, 3, (b, 4)),
    )


def test_forward_shapes(model):
    out = model(*_batch(), need_cross_weights=True)
    assert out.quantiles.shape == (4, 3)
    assert out.dynamic_weights.shape == (4, 24, 17)   # (B, n_patch, n_channel)
    assert out.static_weights.shape == (4, 4)
    assert out.cross_weights.shape == (4, 24, 24)


def test_quantiles_always_monotonic():
    """분위 교차가 생기면 trading/signal.py 가 예외를 던져 백테스트가 죽는다.

    손실 페널티가 아니라 **구조**로 막혀 있어야 한다 — 학습 전 랜덤 초기화에서도 성립.
    """
    head = QuantileHead(d_model=32, n_quantiles=3)
    q = head(torch.randn(256, 32) * 100)   # 극단 입력에도
    assert (q[:, 1:] >= q[:, :-1]).all()


def test_model_output_feeds_signal_without_error(model):
    """모델 출력이 매매 신호 함수에 그대로 들어가야 한다."""
    from src.trading.signal import QuantilePrediction, generate_signal

    cfg = {
        "abstain": {"max_interval_width": 0.05},
        "direction": {"long_threshold": 0.004, "short_threshold": -0.004},
        "sizing": {"method": "inverse_width", "max_position_pct": 0.10},
        "costs": {"commission_bps": 1.5, "tax_bps": 18.0, "slippage_bps": 5.0},
    }
    q = model(*_batch(b=8)).quantiles.detach()
    for row in q:
        pred = QuantilePrediction("005930", *(float(v) for v in row))
        generate_signal(pred, cfg)   # 예외 없이 통과해야 한다


def test_vsn_weights_sum_to_one(model):
    out = model(*_batch())
    assert torch.allclose(out.dynamic_weights.sum(-1), torch.ones(4, 24), atol=1e-5)
    assert torch.allclose(out.static_weights.sum(-1), torch.ones(4), atol=1e-5)


def test_revin_roundtrip():
    revin = RevIN(3, affine=False)
    x = torch.randn(4, 60, 3) * 5 + 10
    normed = revin(x)
    assert normed.mean().abs() < 0.1
    back = revin.denormalize(normed[:, :, 0].transpose(0, 1).mean(0), channel=0)
    assert back.shape == (4,)


def test_patch_count_matches_config():
    assert num_patches(120, 5, 5) == 24
    pe = PatchEmbedding(5, 5, 16, 120)
    assert pe(torch.randn(2, 120, 7)).shape == (2, 7, 24, 16)


def test_patch_rejects_bad_lookback():
    with pytest.raises(ValueError, match="lookback"):
        num_patches(3, 5, 5)
