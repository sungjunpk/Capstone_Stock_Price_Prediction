"""Phase 1 학습 루프.

로컬(Mac/MPS)과 클라우드 GPU(Colab/Kaggle, CUDA) 양쪽에서 같은 코드로 돈다.
디바이스별 차이는 여기서 흡수한다:
  - num_workers: macOS + MPS 는 0 이 안전, CUDA 는 병렬로 올린다
  - AMP(혼합정밀): CUDA 에서만 켠다. MPS 는 아직 불안정
  - pin_memory: CUDA 에서만
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.models.phase1 import Phase1Config
from src.training.dataset import (
    StaticVocab,
    WindowDataset,
    dynamic_feature_columns,
    n_passthrough_columns,
)
from src.training.losses import QuantileLoss, pinball_loss
from src.training.split import SplitSpec, apply_normalizer, fit_normalizer, split_by_date
from src.utils.config import PROJECT_ROOT
from src.utils.logging import get_logger
from src.utils.seed import get_device, set_seed

log = get_logger(__name__)

CKPT_DIR = PROJECT_ROOT / "outputs" / "checkpoints"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"


def _config_hash(cfg: dict) -> str:
    """모델/학습/피처/유니버스 설정의 짧은 해시. 체크포인트·리포트 이름에 쓴다.

    ⚠️ **유니버스도 넣는다.** 예전엔 model/training/features 만 봤는데, 그러면
    같은 하이퍼파라미터로 다른 종목 구성을 학습했을 때 파일명이 같아져
    이전 체크포인트와 리포트를 덮어쓴다(CLAUDE.md 규칙 8 위반).
    코스피200 은 정기변경(6·12월)이 있어 이 상황이 반기마다 온다.
    """
    payload = {k: cfg[k] for k in ("model", "training", "features")}
    payload["universe"] = sorted(str(u["code"]) for u in cfg["data"]["universe"])
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()[:8]


# ------------------------------------------------------------ 디바이스 설정
def loader_settings(device: torch.device, requested_workers: int) -> dict:
    """디바이스에 맞는 DataLoader 설정. 클라우드 GPU 에서 자동으로 올라간다."""
    if device.type == "cuda":
        workers = requested_workers if requested_workers > 0 else 4
        return {"num_workers": workers, "pin_memory": True,
                "persistent_workers": workers > 0}
    # MPS/CPU: 워커를 늘리면 오히려 느려지고 macOS 에서 종종 멈춘다
    return {"num_workers": 0, "pin_memory": False, "persistent_workers": False}


def use_amp(device: torch.device) -> bool:
    """혼합정밀은 CUDA 에서만. MPS 의 autocast 는 아직 신뢰하기 어렵다."""
    return device.type == "cuda"


# ------------------------------------------------------------ 데이터 준비
def build_loaders(cfg: dict, *, smoke: bool = False):
    """panel/macro/static → train/val/test DataLoader 와 메타정보."""
    from src.data.storage import PROCESSED_DIR

    # 프로파일마다 산출물이 다르다 (일봉: panel.parquet / 60분봉: panel_60m.parquet).
    # 체크포인트 이름은 _config_hash 가 이미 갈라준다 — features 가 다르기 때문이다.
    sfx = cfg["data"].get("processed_suffix", "")
    panel = pd.read_parquet(PROCESSED_DIR / f"panel{sfx}.parquet")
    macro = pd.read_parquet(PROCESSED_DIR / f"macro{sfx}.parquet")
    static = pd.read_parquet(PROCESSED_DIR / f"static{sfx}.parquet")

    if smoke:
        codes = sorted(panel["code"].unique())[:6]
        panel = panel[panel["code"].isin(codes)]
        static = static[static["code"].isin(codes)]
        log.info("[smoke] 종목 %d개로 축소", len(codes))

    feature_cols = dynamic_feature_columns(panel)
    # 보조 지평 타깃. dynamic_feature_columns 가 target* 을 전부 걸러내므로
    # 피처로는 절대 안 들어간다 (절대 규칙 5, tests/test_dataset.py 가 고정).
    hz = int(cfg["features"]["return_horizon"])
    # 모델이 쓰겠다고 선언한 것만 싣는다 (패널에는 더 많이 있을 수 있다).
    aux_hz = [int(h) for h in cfg["model"].get("aux_horizons", []) if int(h) != hz]
    aux_target_cols = [
        f"target_h{h}" for h in aux_hz if f"target_h{h}" in panel.columns
    ]
    spec = SplitSpec.from_config(cfg)
    # ⚠️ embargo 는 **가장 긴 지평**을 덮어야 한다. 보조 지평이 20일인데 embargo 가
    #    5일이면 train 마지막 샘플의 t+20 라벨이 val 구간 15일치를 훔쳐본다
    #    (절대 규칙 5). 에러가 아니라 val 이 조용히 좋아지는 실패라 여기서 막는다.
    longest = max([hz, *aux_hz])
    if aux_target_cols and spec.embargo_days < longest:
        raise ValueError(
            f"embargo_days({spec.embargo_days}) 가 가장 긴 지평({longest})보다 짧다 — "
            f"보조 지평 {aux_hz} 의 라벨이 구간 경계를 넘어 샌다. "
            "split.embargo_days 를 늘리거나 aux_horizons 를 줄일 것."
        )
    parts = split_by_date(panel, spec)

    # 정규화 통계는 **train 구간에서만** 계산한다 (CLAUDE.md 절대 규칙)
    train_usable = parts["train"].dropna(subset=feature_cols + ["target"])
    stats = fit_normalizer(train_usable, feature_cols)
    macro_cols = [c for c in macro.columns if c != "date"]
    macro_train = macro[pd.to_datetime(macro["date"]).dt.date <= spec.train_end]
    macro_stats = fit_normalizer(macro_train.dropna(subset=macro_cols), macro_cols)

    macro_n = apply_normalizer(macro.fillna(0.0), macro_stats)
    vocab = StaticVocab.build(static)

    loaders, sizes = {}, {}
    train_cfg = cfg["training"]
    device = get_device()
    ls = loader_settings(device, int(train_cfg.get("num_workers", 0)))

    for name in ("train", "val", "test"):
        part = parts[name]
        if part.empty:
            continue
        ds = WindowDataset(
            apply_normalizer(part, stats), macro_n, static,
            lookback=int(cfg["features"]["lookback"]),
            feature_cols=feature_cols, vocab=vocab,
            aux_target_cols=aux_target_cols,
        )
        loaders[name] = DataLoader(
            ds, batch_size=int(train_cfg["batch_size"]),
            shuffle=(name == "train"), drop_last=(name == "train"), **ls,
        )
        sizes[name] = len(ds)

    # 무조건부 분위수 = 아무것도 학습하지 않은 모델. 두 곳에 쓴다:
    #   1) 헤드 bias 초기화 — 기준선에서 출발시켜 스케일 맞추기에 시간을 안 쓰게 한다
    #   2) 리포트의 기준선 — 이걸 못 이기면 조건부 신호를 못 찾은 것이다
    quantiles = tuple(float(q) for q in cfg["model"]["head"]["quantiles"])
    y_train = torch.tensor(train_usable["target"].to_numpy(), dtype=torch.float32)
    base_q = torch.quantile(y_train, torch.tensor(quantiles))

    meta = {
        "aux_target_cols": aux_target_cols,
        "feature_cols": feature_cols, "macro_cols": macro_cols,
        "vocab_sizes": vocab.sizes, "sizes": sizes, "split": str(spec),
        "baseline_quantiles": [float(v) for v in base_q],
        # ⚠️ **코드북 자체**를 남긴다. 크기만 남기면 추론이 그때그때 static.parquet 에서
        #    다시 만드는데, 유니버스가 바뀌면 같은 범주가 다른 인덱스를 받는다.
        #    2026-09-08 에 146→201종목으로 넓히면서 실제로 그랬다:
        #    size_class 에 '소형주'가 생겨 중형주 인덱스가 표 크기를 넘었고,
        #    MPS 는 예외 없이 0 벡터를 돌려줘 **조용히 틀렸다**(CPU 는 IndexError).
        "vocab": {"sector": vocab.sector,
                  "market_cap_bucket": {str(k): v
                                        for k, v in vocab.market_cap_bucket.items()}},
        # 이 체크포인트가 무엇으로 학습됐는지. 나중에 되짚을 유일한 단서다.
        "universe": {"n_stocks": int(static["code"].nunique()),
                     **{k: v for k, v in (cfg["data"].get("universe_meta") or {})
                        .get("criteria", {}).items()}},
    }
    return loaders, meta


# ------------------------------------------------------------ 모델·손실 선택
def _build_model(cfg: dict, meta: dict, init_quantiles: tuple[float, ...]):
    """`model.arch` 가 고르는 아키텍처. 기본값 phase1 — **기존 경로는 안 바뀐다.**

    베이스라인(원본 PatchTST/TFT)은 대조군이다. 같은 학습 루프를 타야 비교가
    성립하므로 여기서만 갈라지고, 이후 경로는 전부 공유한다.
    """
    arch = str(cfg["model"].get("arch", "phase1"))
    common = {
        "n_dynamic": len(meta["feature_cols"]),
        "n_macro": len(meta["macro_cols"]),
        "static_vocab": meta["vocab_sizes"],
    }
    from src.models.baselines import VARIANT_ARCHS, build_variant

    if arch == "phase1" or arch in VARIANT_ARCHS:
        # 모듈 교체 변형(itrans/timexer)은 우리 모델에서 모듈 하나만 바뀐 것이라
        # Phase1Config 를 그대로 쓴다 — 나머지 설정이 전부 같아야 비교가 성립한다.
        mcfg = Phase1Config.from_config(
            cfg, n_passthrough=n_passthrough_columns(meta["feature_cols"]), **common
        )
        mcfg.init_quantiles = init_quantiles
        return arch, mcfg, build_variant(arch, mcfg)

    from src.models.baselines import BaselineConfig, build_baseline

    mcfg = BaselineConfig.from_config(cfg, arch=arch, **common)
    mcfg.init_quantiles = init_quantiles
    return arch, mcfg, build_baseline(mcfg)


def _weight_names(meta: dict, n: int) -> list[str]:
    """채널 가중치 벡터에 붙일 이름.

    TFT 대조군은 매크로를 관측입력으로 dynamic 에 concat 하므로(원본 방식) 변수선택
    축이 `dynamic + macro` 다. 우리 모델은 매크로가 별도 크로스어텐션 경로라 dynamic 뿐이다.
    """
    names = list(meta["feature_cols"])
    if n == len(names) + len(meta["macro_cols"]):
        return names + list(meta["macro_cols"])
    return names


class _MedianMSE(torch.nn.Module):
    """점 예측 모델(PatchTST 원본)의 손실. 분위 축에서 중앙값 하나만 본다.

    원본은 MSE 로 학습한다 — 그래서 불확실성이 없고 기권 판정이 성립하지 않는다.
    그 사실을 재현하는 것이 이 대조군의 목적이므로 손실도 원본대로 둔다.
    """

    def __init__(self, n_quantiles: int):
        super().__init__()
        self.mid = n_quantiles // 2

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.mse_loss(pred[..., self.mid], target)


# ------------------------------------------------------------ 학습
def _lr_at(step: int, total: int, warmup: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(warmup, 1)
    prog = (step - warmup) / max(total - warmup, 1)
    return base_lr * 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))


@torch.no_grad()
def _baseline_loss(loader, base_q, quantiles, device) -> float:
    """무조건부 분위수를 상수로 예측했을 때의 손실. 모든 실험의 하한선."""
    q = torch.tensor(base_q, dtype=torch.float32, device=device)
    qs = torch.tensor(quantiles, dtype=torch.float32, device=device)
    total, n = 0.0, 0
    for *_, y in loader:
        # 보조 지평이 켜지면 y 가 (B, 1+A) 다. 기준선은 **주 지평만** 잰다 —
        # 이 값이 모든 실험의 비교 축이라 정의가 흔들리면 안 된다.
        y = _split_y(y.to(device))[0]
        total += pinball_loss(q.expand(y.size(0), -1), y, qs).item() * y.size(0)
        n += y.size(0)
    return total / max(n, 1)


def _split_y(y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
    """(B,) 또는 (B, 1+A) 를 (주 타깃, 보조 타깃) 으로 가른다.

    보조 지평이 꺼져 있으면 y 가 그냥 (B,) 라 예전과 같은 경로다.
    """
    if y.dim() == 1:
        return y, None
    return y[:, 0], (y[:, 1:] if y.shape[1] > 1 else None)


@torch.no_grad()
def evaluate(model, loader, criterion, device,
             quantiles=None) -> tuple[float, float, np.ndarray]:
    """(감시 손실, val pinball, 채널 가중치 평균).

    감시 손실은 early stopping 과 체크포인트 선택에 쓴다 — 점 예측 모델이면 MSE 다.
    **pinball 은 어느 아키텍처든 항상 같이 잰다.** 무조건부 분위수 기준선과 여태
    기록한 모든 실험 수치가 pinball 축이라, 이걸 빼면 대조군을 표에 못 올린다.
    """
    model.eval()
    total, pin_total, n = 0.0, 0.0, 0
    weight_sum = None
    qs = None if quantiles is None else torch.tensor(
        [float(q) for q in quantiles], dtype=torch.float32, device=device)
    for dyn, mac, stat, y in loader:
        dyn, mac, stat, y = (t.to(device, non_blocking=True) for t in (dyn, mac, stat, y))
        out = model(dyn, mac, stat)
        # ⚠️ 검증 손실은 **주 지평만** 잰다. 보조를 섞으면 무조건부 분위수 기준선과
        #    지금까지의 모든 실험 수치와 비교가 안 된다.
        y_main, _ = _split_y(y)
        loss = criterion(out.quantiles, y_main)
        bs = y.size(0)
        total += loss.item() * bs
        pin_total += (
            loss.item() if qs is None
            else pinball_loss(out.quantiles, y_main, qs).item()
        ) * bs
        n += bs
        w = out.dynamic_weights.mean(dim=(0, 1)).float().cpu().numpy()
        weight_sum = w * bs if weight_sum is None else weight_sum + w * bs
    return total / max(n, 1), pin_total / max(n, 1), (weight_sum / max(n, 1))


def train(cfg: dict, *, smoke: bool = False, max_epochs: int | None = None) -> dict:
    set_seed(int(cfg["project"]["seed"]))
    device = get_device()
    log.info("디바이스: %s | AMP: %s", device, use_amp(device))

    loaders, meta = build_loaders(cfg, smoke=smoke)
    log.info("샘플 수: %s", meta["sizes"])

    arch, mcfg, model = _build_model(cfg, meta, tuple(meta["baseline_quantiles"]))
    model = model.to(device)
    meta["arch"] = arch          # 추론이 어느 클래스를 세울지 여기서만 알 수 있다
    n_params = sum(p.numel() for p in model.parameters())
    log.info("아키텍처 %s | 파라미터 %.2fM", arch, n_params / 1e6)

    t = cfg["training"]
    loss_name = str(t.get("loss", "pinball"))
    if loss_name == "mse":
        criterion = _MedianMSE(len(mcfg.quantiles)).to(device)
    elif loss_name == "pinball":
        criterion = QuantileLoss(mcfg.quantiles, crossing_weight=0.01).to(device)
    else:
        raise ValueError(f"training.loss 는 pinball|mse 다: {loss_name!r}")
    # 보조 지평 손실의 비중. 주 과제를 밀어내지 않을 정도로만 준다.
    aux_weight = float(cfg["training"].get("aux_weight", 0.3))

    # 기준선 손실을 먼저 재둔다. 학습이 이걸 못 이기면 의미가 없다.
    baseline_loss = _baseline_loss(loaders["val"], meta["baseline_quantiles"],
                                   mcfg.quantiles, device)
    log.info("기준선(무조건부 분위수) val pinball = %.6f", baseline_loss)
    opt = torch.optim.AdamW(
        model.parameters(), lr=float(t["lr"]), weight_decay=float(t["weight_decay"])
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp(device))

    epochs = max_epochs or int(t["epochs"])
    if smoke:
        epochs = min(epochs, 2)
    steps_per_epoch = len(loaders["train"])
    total_steps = epochs * steps_per_epoch
    warmup = int(t.get("warmup_epochs", 0)) * steps_per_epoch
    base_lr = float(t["lr"])
    grad_clip = float(t.get("grad_clip", 1.0))

    es = t.get("early_stopping", {})
    patience, min_delta = int(es.get("patience", 12)), float(es.get("min_delta", 1e-5))

    # 설정 해시를 이름에 넣는다. 스윕이 여러 설정을 돌 때 서로 덮어쓰지 않아야
    # 나중에 승자 체크포인트를 골라 쓸 수 있다 (CLAUDE.md: 결과를 덮어쓰지 않는다).
    cfg_hash = _config_hash(cfg)
    best, best_epoch, bad, step = float("inf"), -1, 0, 0
    best_pinball = float("nan")
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    # 트랙 태그를 이름에 넣는다. 해시만으로는 부족하다 — 학습 당시 설정과
    # 지금 설정의 해시가 어긋나면(캐글에서 받은 체크포인트 등) 자동 선택이
    # 다른 트랙 것을 집어간다. 그러면 **조용히 틀린 숫자**가 나온다.
    # 파일명 태그는 보통 데이터 접미사와 같지만, **갈라 쓸 수 있어야 한다.**
    # 스윕은 base 데이터(panel.parquet)를 그대로 읽으면서 체크포인트만 따로
    # 떨어져야 한다 — 무태그로 떨어지면 scripts/paper_trade.py 가 실험 모델을
    # 실주문에 집어간다(2026-09-14 실제 발생). 데이터 접미사를 바꾸면 없는
    # panel<suffix>.parquet 을 찾으므로 그 방법은 못 쓴다.
    tag = cfg["data"].get("checkpoint_suffix", cfg["data"].get("processed_suffix", ""))
    suffix = "_smoke" if smoke else ""
    ckpt_path = CKPT_DIR / f"phase1_{cfg_hash}{tag}{suffix}.pt"
    history = []

    for epoch in range(epochs):
        model.train()
        t0, run, seen = time.time(), 0.0, 0
        for dyn, mac, stat, y in loaders["train"]:
            for g in opt.param_groups:
                g["lr"] = _lr_at(step, total_steps, warmup, base_lr)

            dyn, mac, stat, y = (x.to(device, non_blocking=True) for x in (dyn, mac, stat, y))
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp(device)):
                out = model(dyn, mac, stat)
                y_main, y_aux = _split_y(y)
                loss = criterion(out.quantiles, y_main)
                if out.aux_quantiles is not None and y_aux is not None:
                    n_aux = min(out.aux_quantiles.shape[1], y_aux.shape[1])
                    aux = sum(
                        criterion(out.aux_quantiles[:, i], y_aux[:, i])
                        for i in range(n_aux)
                    ) / max(n_aux, 1)
                    loss = loss + aux_weight * aux
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()

            run += loss.item() * y.size(0)
            seen += y.size(0)
            step += 1

        tr_loss = run / max(seen, 1)
        val_loss, val_pinball, val_w = evaluate(
            model, loaders["val"], criterion, device, quantiles=mcfg.quantiles
        )
        dt = time.time() - t0
        log.info(
            "epoch %2d/%d | train %.6f | val %.6f | pinball %.6f | %.0fs | lr %.2e",
            epoch + 1, epochs, tr_loss, val_loss, val_pinball, dt,
            opt.param_groups[0]["lr"],
        )
        history.append({"epoch": epoch + 1, "train": tr_loss, "val": val_loss,
                        "val_pinball": val_pinball, "sec": round(dt, 1)})

        if val_loss < best - min_delta:
            best, best_epoch, bad = val_loss, epoch + 1, 0
            best_pinball = val_pinball
            torch.save(
                {"model": model.state_dict(), "config": asdict(mcfg), "meta": meta,
                 "val_loss": best, "epoch": best_epoch},
                ckpt_path,
            )
        else:
            bad += 1
            if bad >= patience:
                log.info("early stopping (patience %d) — best epoch %d", patience, best_epoch)
                break

    # --- 실험 리포트 (덮어쓰지 않는다: 날짜 + 설정 해시)
    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config_hash": cfg_hash, "device": str(device), "smoke": smoke,
        "arch": arch, "loss": loss_name,
        "n_params": n_params, "sizes": meta["sizes"],
        "universe": meta["universe"],
        "best_val_loss": best, "best_epoch": best_epoch,
        # 감시 손실이 MSE 면 best 와 척도가 다르다. 비교표는 **항상 이 pinball** 을 쓴다.
        "best_val_pinball": best_pinball,
        "baseline_val_loss": baseline_loss,
        "improvement_vs_baseline_pct": round(
            100 * (baseline_loss - best_pinball) / baseline_loss, 3),
        "beats_baseline": bool(best_pinball < baseline_loss),
        "history": history,
        # VSN 채널 가중치 = 해석가능성 리포트의 근거
        "feature_importance": dict(
            sorted(zip(_weight_names(meta, len(val_w)), [float(x) for x in val_w],
                       strict=True),
                   key=lambda kv: -kv[1])
        ),
        "checkpoint": str(ckpt_path.relative_to(PROJECT_ROOT)),
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    name = f"{datetime.now():%Y%m%d_%H%M%S}_{cfg_hash}{'_smoke' if smoke else ''}.json"
    Path(REPORT_DIR / name).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log.info("리포트: outputs/reports/%s", name)

    imp = report["improvement_vs_baseline_pct"]
    if best_pinball >= baseline_loss:
        log.warning(
            "❌ 기준선(%.6f)을 못 이겼다 (%.2f%%). 조건부 신호를 못 찾았다는 뜻 — "
            "하이퍼파라미터보다 피처/타깃 설계를 먼저 볼 것.", baseline_loss, imp,
        )
    elif imp < 1.0:
        log.warning("⚠️ 기준선 대비 %.2f%% 개선에 그쳤다 — 사실상 무조건부 분포만 학습했다", imp)
    else:
        log.info("✅ 기준선 대비 %.2f%% 개선", imp)
    return report
