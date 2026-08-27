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
    """모델/학습/피처 설정의 짧은 해시. 체크포인트·리포트 이름에 쓴다."""
    return hashlib.sha256(
        json.dumps({k: cfg[k] for k in ("model", "training", "features")},
                   sort_keys=True, default=str).encode()
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

    # 무조건부 분위수 = 아무것도 학습하지 않은 모델. 두 곳에 쓴다:
    #   1) 헤드 bias 초기화 — 기준선에서 출발시켜 스케일 맞추기에 시간을 안 쓰게 한다
    #   2) 리포트의 기준선 — 이걸 못 이기면 조건부 신호를 못 찾은 것이다
    quantiles = tuple(float(q) for q in cfg["model"]["head"]["quantiles"])
    y_train = torch.tensor(train_usable["target"].to_numpy(), dtype=torch.float32)
    base_q = torch.quantile(y_train, torch.tensor(quantiles))

    meta = {
        "feature_cols": feature_cols, "macro_cols": macro_cols,
        "vocab_sizes": vocab.sizes, "sizes": sizes, "split": str(spec),
        "baseline_quantiles": [float(v) for v in base_q],
    }
    return loaders, meta


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
        y = y.to(device)
        total += pinball_loss(q.expand(y.size(0), -1), y, qs).item() * y.size(0)
        n += y.size(0)
    return total / max(n, 1)


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
        cfg, n_passthrough=n_passthrough_columns(meta["feature_cols"]),
        n_dynamic=len(meta["feature_cols"]),
        n_macro=len(meta["macro_cols"]), static_vocab=meta["vocab_sizes"],
    )
    mcfg.init_quantiles = tuple(meta["baseline_quantiles"])
    model = Phase1Model(mcfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("파라미터 %.2fM", n_params / 1e6)

    t = cfg["training"]
    criterion = QuantileLoss(mcfg.quantiles, crossing_weight=0.01).to(device)

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
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    # 트랙 태그를 이름에 넣는다. 해시만으로는 부족하다 — 학습 당시 설정과
    # 지금 설정의 해시가 어긋나면(캐글에서 받은 체크포인트 등) 자동 선택이
    # 다른 트랙 것을 집어간다. 그러면 **조용히 틀린 숫자**가 나온다.
    tag = cfg["data"].get("processed_suffix", "")
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
    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config_hash": cfg_hash, "device": str(device), "smoke": smoke,
        "n_params": n_params, "sizes": meta["sizes"],
        "best_val_loss": best, "best_epoch": best_epoch,
        "baseline_val_loss": baseline_loss,
        "improvement_vs_baseline_pct": round(100 * (baseline_loss - best) / baseline_loss, 3),
        "beats_baseline": bool(best < baseline_loss),
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

    imp = report["improvement_vs_baseline_pct"]
    if best >= baseline_loss:
        log.warning(
            "❌ 기준선(%.6f)을 못 이겼다 (%.2f%%). 조건부 신호를 못 찾았다는 뜻 — "
            "하이퍼파라미터보다 피처/타깃 설계를 먼저 볼 것.", baseline_loss, imp,
        )
    elif imp < 1.0:
        log.warning("⚠️ 기준선 대비 %.2f%% 개선에 그쳤다 — 사실상 무조건부 분포만 학습했다", imp)
    else:
        log.info("✅ 기준선 대비 %.2f%% 개선", imp)
    return report
