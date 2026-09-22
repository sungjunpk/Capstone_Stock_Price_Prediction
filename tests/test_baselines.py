"""대조군(원본 PatchTST · TFT)이 Phase 1 과 **같은 출력 계약**을 지키는지.

계약이 깨지면 비교가 성립하지 않는다 — 학습 루프·추론·매매 경로를 공유하기 때문이다.
"""

import pytest
import torch

from src.models.baselines import (
    ARCHS,
    VARIANT_ARCHS,
    BaselineConfig,
    build_baseline,
    build_variant,
)
from src.models.phase1 import Phase1Config

VOCAB = {"sector": 22, "market_cap_bucket": 6, "day_of_week": 6}
N_DYN, N_MAC, LOOKBACK = 17, 13, 120


def _cfg(arch: str, **kw) -> BaselineConfig:
    return BaselineConfig(arch=arch, n_dynamic=N_DYN, n_macro=N_MAC,
                          static_vocab=VOCAB, **kw)


def _batch(b=4):
    return (torch.randn(b, LOOKBACK, N_DYN),
            torch.randn(b, LOOKBACK, N_MAC),
            torch.randint(0, 3, (b, len(VOCAB))))


@pytest.mark.parametrize("arch", ARCHS)
def test_output_contract(arch):
    """(B,3) 분위 + 단조성 + 해석 가중치 — signal.py 가 기대하는 모양 그대로."""
    out = build_baseline(_cfg(arch))(*_batch())
    assert out.quantiles.shape == (4, 3)
    assert torch.all(out.quantiles[:, 1:] >= out.quantiles[:, :-1]), "분위 교차"
    assert out.static_weights.shape == (4, len(VOCAB))
    assert out.dynamic_weights.dim() == 3 and out.dynamic_weights.shape[0] == 4


def test_patchtst_point_has_zero_width():
    """원본 PatchTST 는 폭이 **정확히 0** 이다 — 기권 판정이 성립하지 않는다.

    이 프로젝트 차별점 1(기권 로직)의 정량적 근거라서 테스트로 고정한다.
    """
    out = build_baseline(_cfg("patchtst", init_quantiles=(-0.05, 0.0, 0.05)))(*_batch())
    width = out.quantiles[:, 2] - out.quantiles[:, 0]
    assert torch.allclose(width, torch.zeros_like(width))

    # 분위 헤드를 달면 폭이 생긴다 — 같은 백본인데 여기서 갈린다
    q = build_baseline(_cfg("patchtst_q", init_quantiles=(-0.05, 0.0, 0.05)))(*_batch())
    assert torch.all(q.quantiles[:, 2] - q.quantiles[:, 0] > 0)


def test_patchtst_ignores_macro_and_static():
    """PatchTST 원본에는 외생변수·정적 공변량 개념이 없다. 그 사실을 고정한다.

    비교표가 "입력 정보량이 다르다"고 적는 근거다 — 구현 실수로 몰래 보고 있으면
    그 서술이 거짓이 된다.
    """
    model = build_baseline(_cfg("patchtst_q")).eval()
    dyn, mac, stat = _batch()
    with torch.no_grad():
        a = model(dyn, mac, stat).quantiles
        b = model(dyn, torch.randn_like(mac), torch.zeros_like(stat)).quantiles
    assert torch.allclose(a, b)


def test_tft_uses_macro_as_observed_input():
    """TFT 원본은 매크로를 dynamic 에 concat 한다 — 변수선택 축이 30+13 이다."""
    model = build_baseline(_cfg("tft")).eval()
    dyn, mac, stat = _batch()
    with torch.no_grad():
        out = model(dyn, mac, stat)
        changed = model(dyn, torch.randn_like(mac), stat).quantiles
    # 시점축(패치가 아니라 120시점) × 전체 변수
    assert out.dynamic_weights.shape == (4, LOOKBACK, N_DYN + N_MAC)
    assert not torch.allclose(out.quantiles, changed), "매크로를 안 보고 있다"


def test_unknown_arch_raises():
    with pytest.raises(ValueError, match="모르는 아키텍처"):
        build_baseline(_cfg("mamba"))


# ── 모듈 교체 변형 (itrans / timexer) ─────────────────────────────────────
# 원본 셋과 달리 **우리 모델에서 모듈 하나만 바뀐 것**이라 Phase1Config 를 쓴다.
# 우리 모델과 같은 설정을 주고, 바뀐 모듈 하나만 실제로 바뀌었는지 고정한다.

def _p1(**kw) -> Phase1Config:
    return Phase1Config(n_dynamic=N_DYN, n_macro=N_MAC, static_vocab=VOCAB,
                        d_model=32, n_heads=2, n_layers=1, d_ff=64, **kw)


@pytest.mark.parametrize("arch", VARIANT_ARCHS)
def test_variant_output_contract(arch):
    """교체판도 (B,3) 분위 + 단조성 + 해석 가중치를 그대로 낸다."""
    out = build_variant(arch, _p1())(*_batch())
    assert out.quantiles.shape == (4, 3)
    assert torch.all(out.quantiles[:, 1:] >= out.quantiles[:, :-1]), "분위 교차"
    assert out.static_weights.shape == (4, len(VOCAB))
    assert out.dynamic_weights.shape[0] == 4 and out.dynamic_weights.shape[2] == N_DYN


@pytest.mark.parametrize("arch", VARIANT_ARCHS)
def test_variant_has_interval(arch):
    """분위 헤드가 살아 있다 — 기권 로직이 성립한다(레이더의 리스크 제어 축)."""
    out = build_variant(arch, _p1(init_quantiles=(-0.05, 0.0, 0.05)))(*_batch())
    assert torch.all(out.quantiles[:, 2] - out.quantiles[:, 0] > 0)


def test_itrans_collapses_time_axis():
    """iTransformer 는 토큰이 변수라서 **시간 해상도를 버린다** — VSN 가중치가 (B,1,C).

    우리 모델은 패치마다 가중치를 내므로 (B,24,C) 다. 이 차이가 '무엇을 잃었나' 의
    정량적 근거라서 테스트로 고정한다.
    """
    out = build_variant("itrans", _p1())(*_batch())
    assert out.dynamic_weights.shape == (4, 1, N_DYN)
    assert build_variant("phase1", _p1())(*_batch()).dynamic_weights.shape == (4, 24, N_DYN)


def test_timexer_swaps_only_macro_path():
    """TimeXer 는 매크로 처리만 바뀐다 — 종목 경로는 우리 모델과 같은 패치다."""
    out = build_variant("timexer", _p1())(*_batch())
    assert out.dynamic_weights.shape == (4, 24, N_DYN)      # 패치 24개 그대로

    model = build_variant("timexer", _p1()).eval()
    dyn, mac, stat = _batch()
    with torch.no_grad():
        a = model(dyn, mac, stat).quantiles
        b = model(dyn, torch.randn_like(mac), stat).quantiles
    assert not torch.allclose(a, b), "매크로를 안 보고 있다"


@pytest.mark.parametrize("arch,gone", [("itrans", "patch_dyn"), ("timexer", "patch_mac")])
def test_variant_drops_replaced_module(arch, gone):
    """갈아끼운 자리의 옛 모듈은 **파라미터에 남으면 안 된다** — 파라미터 수 비교가 틀어진다."""
    names = dict(build_variant(arch, _p1()).named_parameters())
    assert not any(n.startswith(gone + ".") for n in names)


def test_unknown_variant_raises():
    with pytest.raises(ValueError, match="모르는 아키텍처"):
        build_variant("mamba", _p1())
