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

from src.models.phase1 import Phase1Config, Phase1Model
from src.training.dataset import (
    StaticVocab,
    WindowDataset,
    dynamic_feature_columns,
)
from src.training.losses import QuantileLoss
from src.training.split import SplitSpec, apply_normalizer, fit_normalizer, split_by_date
from src.utils.config import PROJECT_ROOT
from src.utils.logging import get_logger
from src.utils.seed import get_device, set_seed

log = get_logger(__name__)

CKPT_DIR = PROJECT_ROOT / "outputs" / "checkpoints"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"


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

    panel = pd.read_parquet(PROCESSED_DIR / "panel.parquet")
    macro = pd.read_parquet(PROCESSED_DIR / "macro.parquet")
    static = pd.read_parquet(PROCESSED_DIR / "static.parquet")

    if smoke:
        codes = sorted(panel["code"].unique())[:6]
        panel = panel[panel["code"].isin(codes)]
        static = static[static["code"].isin(codes)]
        log.info("[smoke] 종목 %d개로 축소", len(codes))

    feature_cols = dynamic_feature_columns(panel)
    spec = SplitSpec.from_config(cfg)
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
        )
        loaders[name] = DataLoader(
            ds, batch_size=int(train_cfg["batch_size"]),
            shuffle=(name == "train"), drop_last=(name == "train"), **ls,
        )
        sizes[name] = len(ds)

    meta = {
        "feature_cols": feature_cols, "macro_cols": macro_cols,
        "vocab_sizes": vocab.sizes, "sizes": sizes, "split": str(spec),
    }
    return loaders, meta


# ------------------------------------------------------------ 학습
def _lr_at(step: int, total: int, warmup: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(warmup, 1)
    prog = (step - warmup) / max(total - warmup, 1)
    return base_lr * 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> tuple[float, np.ndarray]:
    model.eval()
    total, n = 0.0, 0
    weight_sum = None
    for dyn, mac, stat, y in loader:
        dyn, mac, stat, y = (t.to(device, non_blocking=True) for t in (dyn, mac, stat, y))
        out = model(dyn, mac, stat)
        loss = criterion(out.quantiles, y)
        bs = y.size(0)
        total += loss.item() * bs
        n += bs
        w = out.dynamic_weights.mean(dim=(0, 1)).float().cpu().numpy()
        weight_sum = w * bs if weight_sum is None else weight_sum + w * bs
    return total / max(n, 1), (weight_sum / max(n, 1))


def train(cfg: dict, *, smoke: bool = False, max_epochs: int | None = None) -> dict:
    set_seed(int(cfg["project"]["seed"]))
    device = get_device()
    log.info("디바이스: %s | AMP: %s", device, use_amp(device))

    loaders, meta = build_loaders(cfg, smoke=smoke)
    log.info("샘플 수: %s", meta["sizes"])

    mcfg = Phase1Config.from_config(
        cfg, n_dynamic=len(meta["feature_cols"]),
        n_macro=len(meta["macro_cols"]), static_vocab=meta["vocab_sizes"],
    )
    model = Phase1Model(mcfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("파라미터 %.2fM", n_params / 1e6)

    t = cfg["training"]
    criterion = QuantileLoss(mcfg.quantiles, crossing_weight=0.01).to(device)
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

    best, best_epoch, bad, step = float("inf"), -1, 0, 0
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CKPT_DIR / ("phase1_smoke.pt" if smoke else "phase1_best.pt")
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
                loss = criterion(out.quantiles, y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()

            run += loss.item() * y.size(0)
            seen += y.size(0)
            step += 1

        tr_loss = run / max(seen, 1)
        val_loss, val_w = evaluate(model, loaders["val"], criterion, device)
        dt = time.time() - t0
        log.info(
            "epoch %2d/%d | train %.6f | val %.6f | %.0fs | lr %.2e",
            epoch + 1, epochs, tr_loss, val_loss, dt, opt.param_groups[0]["lr"],
        )
        history.append({"epoch": epoch + 1, "train": tr_loss, "val": val_loss,
                        "sec": round(dt, 1)})

        if val_loss < best - min_delta:
            best, best_epoch, bad = val_loss, epoch + 1, 0
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
    cfg_hash = hashlib.sha256(
        json.dumps({k: cfg[k] for k in ("model", "training", "features")},
                   sort_keys=True, default=str).encode()
    ).hexdigest()[:8]
    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config_hash": cfg_hash, "device": str(device), "smoke": smoke,
        "n_params": n_params, "sizes": meta["sizes"],
        "best_val_loss": best, "best_epoch": best_epoch,
        "history": history,
        # VSN 채널 가중치 = 해석가능성 리포트의 근거
        "feature_importance": dict(
            sorted(zip(meta["feature_cols"], [float(x) for x in val_w], strict=True),
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
    return report
