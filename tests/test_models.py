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


def test_encoder_runs_once_not_per_channel():
    """VSN 이 인코더 **앞**에 있어야 한다 (TFT 순서).

    뒤에 두면 인코더가 채널 수만큼 반복 실행돼 비용이 17배가 된다 —
    실측 259ms → 52ms 차이였다. 인코더 입력이 3차원(B,N,d)이면 채널이
    이미 합쳐진 것이고, 4차원(B,C,N,d)이면 채널별로 도는 것이다.
    """
    model = Phase1Model(Phase1Config(n_dynamic=17, n_macro=13, static_vocab=VOCAB))
    seen = []

    orig = model.encoder.forward
    model.encoder.forward = lambda x: (seen.append(x.dim()), orig(x))[1]
    model(*_batch())

    assert seen == [3], f"인코더가 채널별로 돌고 있다 (입력 차원 {seen})"


def test_grouped_grn_matches_per_channel_semantics():
    """묶음 연산이 채널별 독립 가중치를 유지하는지 — 채널이 섞이면 안 된다."""
    from src.models.vsn import GroupedGRN

    g = GroupedGRN(n_vars=4, d_model=8, hidden=6, dropout=0.0).eval()
    x = torch.zeros(2, 3, 4, 8)
    x[:, :, 1] = 1.0                      # 1번 채널에만 신호

    with torch.no_grad():
        out = g(x)

    # 0/2/3번 채널 출력은 서로 같아야 한다(입력이 같으므로)
    assert torch.allclose(out[:, :, 0], out[:, :, 2], atol=1e-6)
    # 1번 채널은 달라야 한다(입력이 다르므로)
    assert not torch.allclose(out[:, :, 0], out[:, :, 1], atol=1e-4)


# ── 최신 아키텍처 대조 트랙 (timexer / itrans 프로필) ──────────────────────
#
# 두 스위치는 서로 독립이고 기본값은 현재 동작이다. 여기서 고정하는 계약은
# "기본 경로가 변하지 않는다" 와 "새 경로의 해석 가중치 shape" 두 가지다.


def _model(**modes):
    return Phase1Model(
        Phase1Config(n_dynamic=17, n_macro=13, static_vocab=VOCAB, **modes)
    )


def test_default_modes_are_the_patch_track(model):
    """기본값을 건드리면 실거래 모델이 조용히 바뀐다 — 기본은 항상 patch 다."""
    assert model.cfg.endog_mode == "patch"
    assert model.cfg.exog_mode == "patch_cross"
    assert hasattr(model, "patch_dyn") and hasattr(model, "encoder_mac")
    assert not hasattr(model, "embed_dyn") and not hasattr(model, "embed_mac")


def test_timexer_exog_gives_per_macro_weights():
    """TimeXer: 매크로 채널당 토큰 하나 → 가중치가 '어느 지수를 봤나' 가 된다."""
    m = _model(exog_mode="variate_token")
    out = m(*_batch(), need_cross_weights=True)

    assert out.cross_weights.shape == (4, 24, 13)   # (B, n_patch, n_macro)
    assert out.dynamic_weights.shape == (4, 24, 17)  # 종목 경로는 그대로
    assert not hasattr(m, "encoder_mac"), "매크로 인코더가 남아 파라미터를 낭비한다"


def test_itransformer_endog_runs_encoder_over_variates():
    """iTransformer: 변수축이 시퀀스축이다.

    인코더 입력은 여전히 3차원(B,C,d)이고 forward 당 한 번만 돈다 —
    patch 트랙의 계약(test_encoder_runs_once_not_per_channel)과 같다.
    VSN 은 인코더 **뒤**로 가므로 가중치의 패치축이 1 이 된다.
    """
    m = _model(endog_mode="variate")
    seen = []
    orig = m.encoder.forward
    m.encoder.forward = lambda x: (seen.append(tuple(x.shape)), orig(x))[1]

    out = m(*_batch(), need_cross_weights=True)

    assert seen == [(4, 17, m.cfg.d_model)], f"인코더가 변수축으로 돌지 않는다 ({seen})"
    assert out.dynamic_weights.shape == (4, 1, 17)
    assert out.cross_weights.shape == (4, 1, 24)


@pytest.mark.parametrize(
    "modes",
    [{"exog_mode": "variate_token"}, {"endog_mode": "variate"}],
)
def test_new_modes_keep_the_output_contract(modes):
    """분위 단조성과 VSN 가중치 합 = 1 은 어느 트랙에서도 깨지면 안 된다."""
    out = _model(**modes)(*_batch())

    assert out.quantiles.shape == (4, 3)
    assert (out.quantiles[:, 1:] >= out.quantiles[:, :-1]).all()
    assert torch.allclose(out.dynamic_weights.sum(-1), torch.ones(1), atol=1e-5)
    assert torch.allclose(out.static_weights.sum(-1), torch.ones(1), atol=1e-5)


@pytest.mark.parametrize("bad", [{"endog_mode": "itransformer"}, {"exog_mode": "timexer"}])
def test_unknown_mode_fails_loudly(bad):
    """오타가 조용히 기본값으로 떨어지면 대조 실험이 통째로 무효가 된다."""
    with pytest.raises(ValueError):
        _model(**bad)
