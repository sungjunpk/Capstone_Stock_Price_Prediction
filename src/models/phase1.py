"""Phase 1 모델 조립.

    dynamic (B,L,C) ─ RevIN ─ patch ─ embed ─→ (B,C,N,d)
                                                 │ 채널 독립 인코더
    static ─ StaticVSN ─→ context (B,d) ─────────┤ (context 로 조건화)
                                                 ↓ DynamicVSN: 채널 가중합
                                              (B,N,d)
    macro (B,L,M) ─ RevIN ─ patch ─ 인코더 ────→ (B,N,d)
                                                 ↓ 크로스어텐션 (Q=종목, KV=매크로)
                                              (B,N,d) ─ pool ─→ (B,d) ─ 헤드 ─→ (B,3)

CLAUDE.md 의 Phase 1 구성 그대로다. Phase 2 는 encoder 만 교체하면 되도록
인코더를 주입 가능한 형태로 두었다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from src.models.cross_attention import CrossAttentionBlock
from src.models.encoder import TransformerEncoder
from src.models.patch_embed import PatchEmbedding, num_patches
from src.models.quantile_head import QuantileHead
from src.models.revin import RevIN
from src.models.variate_embed import VariateEmbedding
from src.models.vsn import DynamicVSN, StaticVSN


@dataclass
class Phase1Config:
    n_dynamic: int
    n_macro: int
    static_vocab: dict[str, int]
    lookback: int = 120
    patch_len: int = 5
    stride: int = 5
    d_model: int = 128
    n_heads: int = 8
    n_layers: int = 3
    d_ff: int = 256
    dropout: float = 0.2
    cross_heads: int = 4
    cross_dropout: float = 0.1
    vsn_hidden: int = 64
    vsn_dropout: float = 0.1
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)
    revin_affine: bool = True
    revin_eps: float = 1e-5
    # 윈도우 변동성으로 출력을 곱하면 상수(무조건부 분위수)조차 표현하기 어려워진다 —
    # 헤드가 1/scale 을 학습해야 하기 때문. 기본은 끈다.
    scale_target: bool = False
    # 학습 데이터의 무조건부 분위수. 헤드 bias 를 여기서 출발시킨다.
    init_quantiles: tuple[float, ...] | None = None
    target_scale_channel: int = 0   # panel 의 첫 피처(ret_1d)를 변동성 기준으로 쓴다
    # 시퀀스를 무엇으로 토큰화할지. 두 축은 서로 독립이다.
    #   endog "patch"       종목을 5일 단위로 자른다 (PatchTST, 기본)
    #         "variate"     종목 채널 하나가 토큰 하나 (iTransformer)
    #   exog  "patch_cross" 매크로도 패치로 잘라 시점끼리 붙인다 (기본)
    #         "variate_token" 매크로 채널 하나가 토큰 하나 (TimeXer)
    #         "both"        둘을 병렬 expert 로 돌려 토큰 축에서 합친다
    endog_mode: str = "patch"
    exog_mode: str = "patch_cross"
    # 채널별 GRN 30벌 대신 한 벌을 공유한다. select(해석 근거)는 그대로다.
    vsn_shared_transform: bool = False
    # 학습 중 입력 채널을 통째로 무작위 마스킹한다(0 이면 끈다).
    # 30채널에 상관 높은 쌍이 많아(rs_20↔xs_rs_20, macd↔macd_signal↔macd_hist)
    # 모델이 한 채널에 기대는 것을 막는다. 추론에서는 동작하지 않는다.
    channel_dropout: float = 0.0
    # 보조 지평 개수. 0 이면 보조 헤드를 안 만든다 — 기존 경로와 완전히 같다.
    # 주 출력 quantiles (B,Q) 의 모양은 어떤 경우에도 안 바뀐다 (절대 규칙 7).
    n_aux_horizons: int = 0
    # 뒤쪽 N개 채널은 RevIN 을 **건너뛴다**. 횡단면 순위 피처(`xs_`)가 여기 해당한다.
    # RevIN 은 종목별 윈도우 안에서 표준화하므로 "오늘 시장에서 몇 등인가"를 지운다 —
    # 정확히 매매 규칙이 쓰는 정보라서, 통과시키지 않으면 모델이 그걸 볼 수 없다.
    n_passthrough: int = 0

    @classmethod
    def from_config(cls, cfg: dict, *, n_dynamic: int, n_macro: int,
                    static_vocab: dict[str, int],
                    n_passthrough: int = 0) -> Phase1Config:
        m = cfg["model"]
        return cls(
            n_dynamic=n_dynamic, n_macro=n_macro, static_vocab=static_vocab,
            lookback=int(cfg["features"]["lookback"]),
            patch_len=int(m["patch"]["patch_len"]),
            stride=int(m["patch"]["stride"]),
            d_model=int(m["encoder"]["d_model"]),
            n_heads=int(m["encoder"]["n_heads"]),
            n_layers=int(m["encoder"]["n_layers"]),
            d_ff=int(m["encoder"]["d_ff"]),
            dropout=float(m["encoder"]["dropout"]),
            cross_heads=int(m["cross_attention"]["n_heads"]),
            cross_dropout=float(m["cross_attention"]["dropout"]),
            vsn_hidden=int(m["vsn"]["hidden_size"]),
            vsn_dropout=float(m["vsn"]["dropout"]),
            quantiles=tuple(m["head"]["quantiles"]),
            revin_affine=bool(m["revin"]["affine"]),
            revin_eps=float(m["revin"]["eps"]),
            scale_target=bool(m["revin"].get("scale_target", False)),
            endog_mode=str(m.get("endog", {}).get("mode", "patch")),
            exog_mode=str(m.get("exog", {}).get("mode", "patch_cross")),
            vsn_shared_transform=bool(m["vsn"].get("shared_transform", False)),
            channel_dropout=float(m.get("channel_dropout", 0.0)),
            # 데이터에 어떤 타깃 컬럼을 만들지는 features.aux_horizons 가 정하고,
            # **그걸 실제로 쓸지는 model.aux_horizons 가 정한다.** 둘을 갈라야
            # 같은 패널로 보조 지평 on/off 를 비교할 수 있다.
            n_aux_horizons=len([
                h for h in m.get("aux_horizons", [])
                if int(h) != int(cfg["features"]["return_horizon"])
            ]),
            n_passthrough=n_passthrough,
        )


@dataclass
class Phase1Output:
    quantiles: torch.Tensor                 # (B, Q) 원래 스케일
    dynamic_weights: torch.Tensor           # (B, N, C) 해석용
    static_weights: torch.Tensor            # (B, n_static) 해석용
    cross_weights: torch.Tensor | None = field(default=None)
    # 보조 지평 예측 (B, n_aux, Q). 학습 손실에만 쓰이고 매매 경로는 안 본다.
    aux_quantiles: torch.Tensor | None = field(default=None)


class Phase1Model(nn.Module):
    def __init__(self, cfg: Phase1Config):
        super().__init__()
        self.cfg = cfg
        self.n_patches = num_patches(cfg.lookback, cfg.patch_len, cfg.stride)

        # --- 종목 경로
        # RevIN 은 앞쪽 채널에만 건다. 뒤쪽 n_passthrough 개는 그대로 흘린다.
        self.n_revin = cfg.n_dynamic - cfg.n_passthrough
        if self.n_revin < 1:
            raise ValueError(
                f"RevIN 채널이 없다 (n_dynamic={cfg.n_dynamic}, "
                f"n_passthrough={cfg.n_passthrough})"
            )
        self.revin_dyn = RevIN(self.n_revin, cfg.revin_eps, cfg.revin_affine)
        if cfg.endog_mode not in ("patch", "variate", "both"):
            raise ValueError(f"endog_mode 를 모른다: {cfg.endog_mode!r}")
        if cfg.endog_mode in ("patch", "both"):
            self.patch_dyn = PatchEmbedding(
                cfg.patch_len, cfg.stride, cfg.d_model, cfg.lookback, cfg.dropout
            )
        if cfg.endog_mode in ("variate", "both"):
            # 변수축이 곧 시퀀스축이 된다 — 인코더가 채널 사이에서 어텐션한다.
            self.embed_dyn = VariateEmbedding(
                cfg.lookback, cfg.n_dynamic, cfg.d_model, cfg.dropout
            )
        self.encoder = TransformerEncoder(
            cfg.d_model, cfg.n_heads, cfg.n_layers, cfg.d_ff, cfg.dropout
        )
        if cfg.endog_mode == "both":
            # 토큰의 의미가 달라(시점 vs 변수) 인코더 가중치를 공유하면 안 된다.
            self.encoder_var = TransformerEncoder(
                cfg.d_model, cfg.n_heads, cfg.n_layers, cfg.d_ff, cfg.dropout
            )
            # 마지막 패치와 변수 토큰을 합쳐 하나의 표현으로 만든다.
            self.fuse = nn.Linear(cfg.d_model * 2, cfg.d_model)

        # --- 매크로 경로 (별도 인코더 — 성격이 다른 시퀀스라 가중치를 공유하지 않는다)
        self.revin_mac = RevIN(cfg.n_macro, cfg.revin_eps, cfg.revin_affine)
        if cfg.exog_mode == "variate_token":
            # 매크로 채널 하나가 토큰 하나 → 크로스어텐션 가중치가
            # "어느 지수를 봤나" 가 된다. 별도 인코더가 필요 없다.
            self.embed_mac = VariateEmbedding(
                cfg.lookback, cfg.n_macro, cfg.d_model, cfg.cross_dropout
            )
        elif cfg.exog_mode == "patch_cross":
            self.patch_mac = PatchEmbedding(
                cfg.patch_len, cfg.stride, cfg.d_model, cfg.lookback, cfg.dropout
            )
            # 매크로는 VSN 을 안 거치므로 채널을 살려둘 이유가 없다.
            # 채널별로 인코딩한 뒤 합치면 인코더가 13번 도는데, 합친 뒤 한 번 도는 것과
            # 표현력 차이는 없고 비용만 13배다.
            self.macro_merge = nn.Linear(cfg.n_macro * cfg.d_model, cfg.d_model)
            self.encoder_mac = TransformerEncoder(
                cfg.d_model, cfg.cross_heads, max(1, cfg.n_layers - 1), cfg.d_ff, cfg.dropout
            )
        else:
            raise ValueError(f"exog_mode 를 모른다: {cfg.exog_mode!r}")

        # --- VSN
        self.static_vsn = StaticVSN(
            cfg.static_vocab, cfg.d_model, cfg.vsn_hidden, cfg.vsn_dropout
        )
        self.dynamic_vsn = DynamicVSN(
            cfg.n_dynamic, cfg.d_model, cfg.vsn_hidden, cfg.vsn_dropout,
            context_size=cfg.d_model,
            shared_transform=cfg.vsn_shared_transform,
        )

        # --- 결합 + 출력
        self.cross = CrossAttentionBlock(cfg.d_model, cfg.cross_heads, cfg.cross_dropout)
        self.head = QuantileHead(
            cfg.d_model, len(cfg.quantiles), dropout=cfg.cross_dropout,
            init_quantiles=cfg.init_quantiles,
        )
        # 지평마다 별도 헤드. 주 헤드와 분리해 두면 보조 과제가 주 출력의
        # 스케일을 끌고 가지 않는다.
        self.aux_heads = nn.ModuleList(
            QuantileHead(cfg.d_model, len(cfg.quantiles), dropout=cfg.cross_dropout,
                         init_quantiles=cfg.init_quantiles)
            for _ in range(cfg.n_aux_horizons)
        ) if cfg.n_aux_horizons else None

    def _patch_branch(
        self, x: torch.Tensor, ctx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(B,L,C) → (B,N,d), (B,N,C). VSN 이 인코더 앞이다 (TFT 순서)."""
        h = self.patch_dyn(x)                            # (B,C,N,d)
        h, w = self.dynamic_vsn(h, ctx)                  # (B,N,d), (B,N,C)
        return self.encoder(h), w

    def _variate_branch(
        self, x: torch.Tensor, ctx: torch.Tensor, encoder: nn.Module
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(B,L,C) → (B,1,d), (B,1,C). 변수축 어텐션 뒤에 VSN 이 온다."""
        h = encoder(self.embed_dyn(x))                   # (B,C,d)
        return self.dynamic_vsn(h.unsqueeze(2), ctx)     # (B,1,d), (B,1,C)

    def forward(
        self,
        dynamic: torch.Tensor,
        macro: torch.Tensor,
        static: torch.Tensor,
        *,
        need_cross_weights: bool = False,
    ) -> Phase1Output:
        """dynamic (B,L,C) / macro (B,L,M) / static (B,n_static) int64"""
        # 종목 경로. patch 모드에서는 TFT 순서대로 **변수 선택을 인코더 앞에서** 한다 —
        # 뒤에 두면 인코더가 채널 수만큼 반복 실행돼 비용이 그만큼 늘어난다.
        if self.cfg.n_passthrough:
            x = torch.cat(                               # (B,L,C)
                [self.revin_dyn(dynamic[..., : self.n_revin]),
                 dynamic[..., self.n_revin :]], dim=-1)
        else:
            x = self.revin_dyn(dynamic)                  # (B,L,C)
        if self.training and self.cfg.channel_dropout > 0:
            # 표본마다 다른 채널을 끈다. 남은 채널을 키워 기댓값을 보존한다.
            keep_p = 1.0 - self.cfg.channel_dropout
            keep = torch.rand(x.shape[0], 1, x.shape[-1], device=x.device) < keep_p
            x = x * keep / keep_p

        ctx, w_static = self.static_vsn(static)          # (B,d), (B,n_static)

        mode = self.cfg.endog_mode
        if mode == "variate":
            # 변수축 어텐션이 먼저다. VSN 을 앞에 두면 인코더가 볼 변수가 없어진다 —
            # 여기서는 인코더가 채널마다 도는 문제가 없으므로 순서를 뒤집어도 된다.
            x, w_dyn = self._variate_branch(x, ctx, self.encoder)   # (B,1,d), (B,1,C)
        elif mode == "patch":
            x, w_dyn = self._patch_branch(x, ctx)        # (B,N,d), (B,N,C)
        else:
            # 병렬 expert — 두 축이 버리는 것이 서로 다르다. 토큰 축에서 합친다.
            xp, wp = self._patch_branch(x, ctx)          # (B,N,d), (B,N,C)
            xv, wv = self._variate_branch(x, ctx, self.encoder_var)  # (B,1,d), (B,1,C)
            x = torch.cat([xp, xv], dim=1)               # (B,N+1,d)
            w_dyn = torch.cat([wp, wv], dim=1)           # (B,N+1,C)

        # 매크로 경로
        m = self.revin_mac(macro)
        if self.cfg.exog_mode == "variate_token":
            m = self.embed_mac(m)                        # (B,M,d) — 채널당 토큰 하나
        else:
            # 채널을 먼저 합치고 한 번만 인코딩
            m = self.patch_mac(m)                        # (B,M,N,d)
            b, n_mac, n, d = m.shape
            m = self.macro_merge(m.permute(0, 2, 1, 3).reshape(b, n, n_mac * d))  # (B,N,d)
            m = self.encoder_mac(m)

        # 결합
        z, w_cross = self.cross(x, m, need_weights=need_cross_weights)
        z = z + ctx.unsqueeze(1)                         # static 문맥을 한 번 더 주입
        if self.cfg.endog_mode == "both":
            # 마지막 패치(가장 최근 구간) + 변수 토큰(전 채널 요약)
            pooled = self.fuse(torch.cat([z[:, -2], z[:, -1]], dim=-1))
        else:
            pooled = z[:, -1]                            # 마지막 패치 = 가장 최근 구간

        q = self.head(pooled)                            # (B,Q) 정규화 공간

        if self.cfg.scale_target:
            # 윈도우 변동성으로 되돌린다 — RevIN 의 "출력 시 역변환"
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

    def target_scale(self) -> torch.Tensor:
        """손실 계산 시 타깃도 같은 스케일로 맞추고 싶을 때 쓴다. (B,)"""
        return self.revin_dyn.scale_of(self.cfg.target_scale_channel)
