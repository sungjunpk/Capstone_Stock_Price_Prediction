"""원본 PatchTST · TFT 베이스라인 — Phase 1 과 정면 비교하기 위한 대조군.

**이 파일은 실거래 경로에 올라가지 않는다.** 체크포인트에 `_patchtst` / `_tft` 태그가
붙고 `scripts/paper_trade.py` 는 무태그만 집는다.

왜 새로 조립하나: 우리 모델은 PatchTST 백본에 TFT 의 변수선택·정적문맥을 얹고
매크로 크로스어텐션을 더한 것이다. "그 결합이 실제로 이득인가"를 말하려면 **같은
데이터·같은 split·같은 학습 루프**에서 원본 둘을 직접 돌린 숫자가 있어야 한다.

세 아키텍처가 한 사다리를 이룬다:
    patchtst    PatchTST 원본 — 점 예측(MSE). 불확실성이 없어 **기권이 불가능하다**
    patchtst_q  + 분위 헤드   — 여기서부터 기권 로직이 성립한다
    tft         TFT 원본      — 변수선택·정적문맥은 있으나 패치가 없다

전부 `Phase1Output` 을 내고 `forward(dynamic, macro, static)` 시그니처를 맞춘다.
그래야 DataLoader·학습 루프·`models/inference.py`·백테스트를 **그대로 공유**한다
(CLAUDE.md 절대 규칙 7 — 비교가 성립하려면 매매 경로가 같아야 한다).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from src.models.encoder import TransformerEncoder
from src.models.patch_embed import PatchEmbedding, num_patches
from src.models.phase1 import Phase1Config, Phase1Model, Phase1Output
from src.models.quantile_head import QuantileHead
from src.models.revin import RevIN
from src.models.vsn import DynamicVSN, GatedResidualNetwork, StaticVSN

ARCHS = ("patchtst", "patchtst_q", "tft")


@dataclass
class BaselineConfig:
    """Phase1Config 와 같은 필드를 쓰되 아키텍처별로 일부만 의미가 있다.

    하이퍼파라미터(d_model/n_layers/dropout)는 **우리 모델과 같은 값을 준다.**
    그것이 베이스라인에 최적이라는 보장은 없고, 그 사실은 리포트에 적는다.
    """

    arch: str
    n_dynamic: int
    n_macro: int
    static_vocab: dict[str, int]
    lookback: int = 120
    patch_len: int = 5
    stride: int = 5
    d_model: int = 32
    n_heads: int = 2
    n_layers: int = 1
    d_ff: int = 64
    dropout: float = 0.5
    vsn_hidden: int = 64
    vsn_dropout: float = 0.5
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)
    revin_affine: bool = True
    revin_eps: float = 1e-5
    init_quantiles: tuple[float, ...] | None = None

    @classmethod
    def from_config(cls, cfg: dict, *, arch: str, n_dynamic: int, n_macro: int,
                    static_vocab: dict[str, int]) -> BaselineConfig:
        m = cfg["model"]
        return cls(
            arch=arch, n_dynamic=n_dynamic, n_macro=n_macro, static_vocab=static_vocab,
            lookback=int(cfg["features"]["lookback"]),
            patch_len=int(m["patch"]["patch_len"]),
            stride=int(m["patch"]["stride"]),
            d_model=int(m["encoder"]["d_model"]),
            n_heads=int(m["encoder"]["n_heads"]),
            n_layers=int(m["encoder"]["n_layers"]),
            d_ff=int(m["encoder"]["d_ff"]),
            dropout=float(m["encoder"]["dropout"]),
            vsn_hidden=int(m["vsn"]["hidden_size"]),
            vsn_dropout=float(m["vsn"]["dropout"]),
            quantiles=tuple(m["head"]["quantiles"]),
            revin_affine=bool(m["revin"]["affine"]),
            revin_eps=float(m["revin"]["eps"]),
        )


def _uniform_weights(b: int, n: int, c: int, device, dtype) -> torch.Tensor:
    """VSN 이 없는 모델의 `dynamic_weights`. 전 채널 균등 = 해석 근거 없음.

    학습 루프가 이 값을 평균 내 `feature_importance` 로 남기므로, 균등하게 두면
    리포트에 1/C 가 그대로 찍힌다 — "이 모델은 어느 피처를 왜 썼는지 말할 수 없다"는
    사실이 숫자로 남는다. 0 으로 두면 '중요도 0'으로 오독된다.
    """
    return torch.full((b, n, c), 1.0 / c, device=device, dtype=dtype)


class PatchTSTBaseline(nn.Module):
    """PatchTST 원본 (Nie et al., ICLR 2023).

        RevIN → 패치 토큰화 → **채널 독립** Transformer → flatten 헤드

    원본에 없어서 **의도적으로 빼는 것**:
      - 매크로 시퀀스(외생변수). PatchTST 는 대상 시계열만 본다
      - static covariate(섹터·규모). 정적 공변량 개념 자체가 없다
      - 변수선택망. 채널 결합이 **학습된 고정 가중**이라 날짜·종목에 따라 변하지 않는다

    마지막 채널 결합(`self.mix`)이 우리 VSN 의 정적 대응물이다. VSN 은 같은 자리에서
    **입력마다 다른** 가중치를 내고 그 값을 밖으로 내보낸다 — 그것이 해석가능성이다.

    `point=True` 는 원본 그대로의 점 예측이다. 출력 계약을 맞추려고 (B,3) 으로 펼치되
    세 분위가 전부 같다 → **구간 폭이 0 이라 기권 판정이 성립하지 않는다.**
    """

    def __init__(self, cfg: BaselineConfig, *, point: bool):
        super().__init__()
        self.cfg = cfg
        self.point = point
        self.n_patches = num_patches(cfg.lookback, cfg.patch_len, cfg.stride)

        self.revin = RevIN(cfg.n_dynamic, cfg.revin_eps, cfg.revin_affine)
        self.patch = PatchEmbedding(
            cfg.patch_len, cfg.stride, cfg.d_model, cfg.lookback, cfg.dropout
        )
        self.encoder = TransformerEncoder(
            cfg.d_model, cfg.n_heads, cfg.n_layers, cfg.d_ff, cfg.dropout
        )
        # 원본 헤드: 채널마다 flatten → 선형. 가중치는 채널 간 공유한다.
        self.flatten_head = nn.Linear(self.n_patches * cfg.d_model, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)
        # 채널 결합 — 스칼라 하나를 내야 하므로 필요하다. 고정 가중이다.
        self.mix = nn.Linear(cfg.n_dynamic, 1)
        self.head = (
            nn.Linear(cfg.d_model, 1) if point
            else QuantileHead(cfg.d_model, len(cfg.quantiles), dropout=cfg.dropout,
                              init_quantiles=cfg.init_quantiles)
        )
        if point and cfg.init_quantiles is not None:
            # 분위 헤드와 같은 출발점을 준다 — 중앙값 기준선에서 시작한다.
            # 안 맞추면 스케일 줄이기에 학습 예산을 쓰는 문제를 이 모델만 받는다.
            with torch.no_grad():
                nn.init.normal_(self.head.weight, std=1e-4)
                mid = sorted(cfg.init_quantiles)[len(cfg.init_quantiles) // 2]
                self.head.bias.fill_(float(mid))

    def forward(
        self, dynamic: torch.Tensor, macro: torch.Tensor, static: torch.Tensor,
        *, need_cross_weights: bool = False,
    ) -> Phase1Output:
        """macro/static 은 받되 쓰지 않는다 — 같은 DataLoader 를 타기 위한 시그니처."""
        x = self.revin(dynamic)                       # (B,L,C)
        x = self.patch(x)                             # (B,C,N,d)
        x = self.encoder(x)                           # (B,C,N,d) 채널을 배치로 접어 처리
        b, c, n, d = x.shape

        x = self.dropout(self.flatten_head(x.reshape(b, c, n * d)))   # (B,C,d)
        pooled = self.mix(x.transpose(1, 2)).squeeze(-1)              # (B,d)

        q = self.head(pooled)
        if self.point:
            q = q.expand(-1, len(self.cfg.quantiles))  # 폭 0 — 기권이 불가능하다

        n_static = len(self.cfg.static_vocab)
        return Phase1Output(
            quantiles=q,
            dynamic_weights=_uniform_weights(b, n, c, q.device, q.dtype),
            static_weights=_uniform_weights(b, 1, n_static, q.device, q.dtype)[:, 0],
        )


class TFTBaseline(nn.Module):
    """TFT 원본 (Lim et al., IJF 2021) 의 핵심 경로.

        변수별 임베딩 → static 인코더 → **시점별** 변수선택 → LSTM
        → static enrichment → temporal self-attention → 분위 헤드

    우리 모델과 갈리는 지점 셋:
      - **패치가 없다.** 120시점을 그대로 처리한다 → 시퀀스가 24 가 아니라 120 이고
        시점별 VSN 텐서가 5배다. 비용 차이를 리포트에 남긴다
      - **RevIN 이 없다.** 원본은 인스턴스 정규화를 안 쓴다 (입력은 train 통계로만
        정규화된 상태다 — 그건 우리 파이프라인 공통이라 그대로 둔다)
      - **매크로가 별도 경로가 아니다.** observed input 으로 dynamic 에 concat 한다.
        원본 TFT 의 방식이고, 우리의 크로스어텐션이 무엇을 바꿨는지 보는 기준이 된다

    원본에서 빼는 것: 미래 known input(달력 등)을 받는 디코더. 우리 과제는 t 에서
    스칼라 하나를 예측하는 단일 스텝이라 디코더가 할 일이 없다.
    """

    def __init__(self, cfg: BaselineConfig):
        super().__init__()
        self.cfg = cfg
        self.n_vars = cfg.n_dynamic + cfg.n_macro
        d = cfg.d_model

        # 변수별 선형 임베딩 = 변수마다 독립인 Linear(1 → d). TFT 의 입력 변환.
        self.var_w = nn.Parameter(torch.empty(self.n_vars, d))
        self.var_b = nn.Parameter(torch.zeros(self.n_vars, d))
        nn.init.normal_(self.var_w, std=0.02)

        self.static_vsn = StaticVSN(cfg.static_vocab, d, cfg.vsn_hidden, cfg.vsn_dropout)
        self.dynamic_vsn = DynamicVSN(
            self.n_vars, d, cfg.vsn_hidden, cfg.vsn_dropout, context_size=d
        )
        # locality enhancement — 어텐션 전에 지역 패턴을 잡는 자리다
        self.lstm = nn.LSTM(d, d, num_layers=1, batch_first=True)
        self.lstm_gate = nn.Linear(d, d * 2)
        self.lstm_norm = nn.LayerNorm(d)
        # static enrichment — 정적 문맥을 시점마다 주입한다
        self.enrich = GatedResidualNetwork(
            d, cfg.vsn_hidden, d, dropout=cfg.dropout, context_size=d
        )
        self.attn = TransformerEncoder(d, cfg.n_heads, cfg.n_layers, cfg.d_ff, cfg.dropout)
        self.head = QuantileHead(d, len(cfg.quantiles), dropout=cfg.dropout,
                                 init_quantiles=cfg.init_quantiles)

    def forward(
        self, dynamic: torch.Tensor, macro: torch.Tensor, static: torch.Tensor,
        *, need_cross_weights: bool = False,
    ) -> Phase1Output:
        x = torch.cat([dynamic, macro], dim=-1)            # (B,L,V) 매크로도 관측 입력이다
        b, length, v = x.shape
        if v != self.n_vars:
            raise ValueError(f"변수 수 불일치: {v} != {self.n_vars}")

        emb = x.unsqueeze(-1) * self.var_w + self.var_b    # (B,L,V,d)
        ctx, w_static = self.static_vsn(static)            # (B,d), (B,n_static)

        # DynamicVSN 은 (B,C,N,d) 를 받는다. 여기서 N 은 패치가 아니라 시점이다.
        sel, w_dyn = self.dynamic_vsn(emb.permute(0, 2, 1, 3), ctx)   # (B,L,d), (B,L,V)

        h, _ = self.lstm(sel)
        a, gate = self.lstm_gate(h).chunk(2, dim=-1)
        h = self.lstm_norm(sel + a * torch.sigmoid(gate))   # 게이트 스킵 연결
        h = self.enrich(h, ctx.unsqueeze(1).expand(b, length, -1))
        h = self.attn(h)

        q = self.head(h[:, -1])                             # 마지막 시점 = 판단 시점
        return Phase1Output(quantiles=q, dynamic_weights=w_dyn, static_weights=w_static)


def build_baseline(cfg: BaselineConfig) -> nn.Module:
    if cfg.arch == "patchtst":
        return PatchTSTBaseline(cfg, point=True)
    if cfg.arch == "patchtst_q":
        return PatchTSTBaseline(cfg, point=False)
    if cfg.arch == "tft":
        return TFTBaseline(cfg)
    raise ValueError(f"모르는 아키텍처: {cfg.arch!r} (가능: {list(ARCHS)})")


# ──────────────────────────────────────────────────────────────────────────
# 모듈 교체 변형 — 위의 셋과 성격이 다르다
#
# 위 셋은 **논문 원본을 그대로 조립한 것**이고, 아래 둘은 **우리 모델에서 모듈
# 하나만 갈아끼운 것**이다. RevIN·변수선택망·정적문맥·분위헤드·횡단면 우회는
# 우리 모델 그대로 남으므로, 차이가 교체한 모듈 하나로만 설명된다.
#
#   itrans   종목 경로: 패치 → 변수 토큰            iTransformer (ICLR 2024)
#   timexer  매크로 경로: 패치 → 채널당 토큰 하나    TimeXer (NeurIPS 2024)
#
# 2026-09-10 실험 때는 `phase1.py` 안의 `endog_mode` / `exog_mode` 스위치였다.
# 그 분기는 2026-09-14 `71ca558` 에서 지웠고, 되살리지 않는다 — 실거래가 타는
# 경로에 대조군용 분기를 다시 넣지 않으려는 것이다.
# ──────────────────────────────────────────────────────────────────────────

VARIANT_ARCHS = ("itrans", "timexer")


class VariateEmbedding(nn.Module):
    """시계열 하나를 통째로 토큰 하나로. (B,L,C) → (B,C,d)

    `patch_embed.py` 와 정반대다 — 시간축을 잘라 토큰을 만드는 대신 lookback 전체를
    한 번에 투영해 **변수 하나당 토큰 하나**를 만든다. 어텐션이 시점이 아니라 변수
    사이에서 걸린다.
    """

    def __init__(self, lookback: int, n_vars: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.n_vars = n_vars
        # 투영은 변수끼리 공유한다. 변수를 구분하는 것은 아래 var 임베딩이다 —
        # 변수마다 별도 투영을 두면 파라미터가 n_vars 배가 되고 과적합한다.
        self.proj = nn.Linear(lookback, d_model)
        self.var = nn.Parameter(torch.zeros(1, n_vars, d_model))
        nn.init.trunc_normal_(self.var, std=0.02)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.n_vars:
            raise ValueError(f"변수 수 불일치: {x.shape[-1]} != {self.n_vars}")
        return self.dropout(self.proj(x.transpose(1, 2)) + self.var)


class _Phase1Variant(Phase1Model):
    """모듈 교체 변형의 공통 뼈대.

    ⚠️ `Phase1Model.forward` 를 **복사한 것**이다. 실거래 경로에 분기를 넣지 않으려고
    이렇게 뒀다 — 대신 `phase1.py` 의 forward 가 바뀌면 **여기도 같이 고쳐야 한다.**
    원본과 다른 곳은 종목 경로(`_endog`)와 매크로 경로(`_exog`) 두 훅뿐이고,
    기본 구현은 우리 모델과 완전히 같다.
    """

    def _endog(self, x: torch.Tensor, ctx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(B,L,C) → (B,N,d), (B,N,C). VSN 이 인코더 앞이다 (TFT 순서)."""
        h = self.patch_dyn(x)                            # (B,C,N,d)
        h, w = self.dynamic_vsn(h, ctx)                  # (B,N,d), (B,N,C)
        return self.encoder(h), w

    def _exog(self, m: torch.Tensor) -> torch.Tensor:
        """(B,L,M) → (B,N,d). 채널을 먼저 합치고 한 번만 인코딩한다."""
        m = self.patch_mac(m)                            # (B,M,N,d)
        b, n_mac, n, d = m.shape
        m = self.macro_merge(m.permute(0, 2, 1, 3).reshape(b, n, n_mac * d))
        return self.encoder_mac(m)

    def forward(
        self,
        dynamic: torch.Tensor,
        macro: torch.Tensor,
        static: torch.Tensor,
        *,
        need_cross_weights: bool = False,
    ) -> Phase1Output:
        """dynamic (B,L,C) / macro (B,L,M) / static (B,n_static) int64"""
        if self.cfg.n_passthrough:
            x = torch.cat(                               # (B,L,C)
                [self.revin_dyn(dynamic[..., : self.n_revin]),
                 dynamic[..., self.n_revin :]], dim=-1)
        else:
            x = self.revin_dyn(dynamic)                  # (B,L,C)
        if self.training and self.cfg.channel_dropout > 0:
            keep_p = 1.0 - self.cfg.channel_dropout
            keep = torch.rand(x.shape[0], 1, x.shape[-1], device=x.device) < keep_p
            x = x * keep / keep_p

        ctx, w_static = self.static_vsn(static)          # (B,d), (B,n_static)
        x, w_dyn = self._endog(x, ctx)                   # ← 교체 지점 1
        m = self._exog(self.revin_mac(macro))            # ← 교체 지점 2

        z, w_cross = self.cross(x, m, need_weights=need_cross_weights)
        z = z + ctx.unsqueeze(1)                         # static 문맥을 한 번 더 주입
        pooled = z[:, -1]                                # 마지막 토큰 = 가장 최근 구간

        q = self.head(pooled)                            # (B,Q) 정규화 공간
        if self.cfg.scale_target:
            scale = self.revin_dyn.scale_of(self.cfg.target_scale_channel)
            q = q * scale.unsqueeze(-1)

        aux = None
        if self.aux_heads is not None:
            aux = torch.stack([h(pooled) for h in self.aux_heads], dim=1)  # (B,A,Q)
            if self.cfg.scale_target:
                aux = aux * scale.unsqueeze(-1).unsqueeze(-1)

        return Phase1Output(
            quantiles=q, dynamic_weights=w_dyn, static_weights=w_static,
            cross_weights=w_cross, aux_quantiles=aux,
        )


class ItransVariant(_Phase1Variant):
    """iTransformer — 종목 경로를 변수 토큰으로.

    변수축이 곧 시퀀스축이라 인코더가 채널마다 돌지 않는다. 그래서 **VSN 을 인코더
    뒤에 둔다** — 앞에 두면 인코더가 볼 변수가 하나로 합쳐져 사라진다.
    잃는 것은 윈도우 안의 시간 해상도다(`dynamic_weights` 가 (B,N,C) → (B,1,C)).
    """

    def __init__(self, cfg: Phase1Config):
        super().__init__(cfg)
        self.embed_dyn = VariateEmbedding(
            cfg.lookback, cfg.n_dynamic, cfg.d_model, cfg.dropout
        )
        del self.patch_dyn          # 패치 경로가 없다 — 파라미터에도 잡히면 안 된다

    def _endog(self, x: torch.Tensor, ctx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(self.embed_dyn(x))              # (B,C,d) 변수 사이 어텐션
        return self.dynamic_vsn(h.unsqueeze(2), ctx)     # (B,1,d), (B,1,C)


class TimeXerVariant(_Phase1Variant):
    """TimeXer — 매크로를 채널당 토큰 하나로.

    우리 설계는 매크로를 패치로 잘라 채널을 합친 뒤 인코딩하는데, TimeXer 는 매크로
    변수 하나를 토큰 하나로 만들어 크로스어텐션의 Key/Value 로 준다. 종목 경로는
    우리 모델 그대로(패치)다.
    """

    def __init__(self, cfg: Phase1Config):
        super().__init__(cfg)
        self.embed_mac = VariateEmbedding(
            cfg.lookback, cfg.n_macro, cfg.d_model, cfg.cross_dropout
        )
        del self.patch_mac
        del self.macro_merge
        del self.encoder_mac

    def _exog(self, m: torch.Tensor) -> torch.Tensor:
        return self.embed_mac(m)                         # (B,M,d)


def build_variant(arch: str, cfg: Phase1Config) -> nn.Module:
    """`Phase1Config` 를 쓰는 모델을 세운다 — 우리 모델과 그 모듈 교체 변형들."""
    if arch == "phase1":
        return Phase1Model(cfg)
    if arch == "itrans":
        return ItransVariant(cfg)
    if arch == "timexer":
        return TimeXerVariant(cfg)
    raise ValueError(f"모르는 아키텍처: {arch!r} (가능: phase1, {list(VARIANT_ARCHS)})")
