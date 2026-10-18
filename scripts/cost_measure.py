#!/usr/bin/env python
"""세 구조의 계산 비용을 **한 장비에서** 잰다 — 논문 Table 5 의 출처.

왜 필요한가: Table 5 의 수치는 임시로 측정한 값이고 리포트가 남지 않았다.
재현 요청에 답할 수 없으므로 같은 장비·같은 배치로 다시 잰다.

무엇을 재는가 (전부 같은 DataLoader 배치 위에서):
    s/epoch      학습 스텝(순전파+역전파+옵티마이저) 배치 시간의 중앙값 × 에폭 배치 수.
                 전체 에폭을 돌리지 않는다 — 배치 시간이 균일하므로 외삽이 더 싸고,
                 세 모형이 **같은 배치 수**를 쓰므로 비교는 그대로 성립한다.
    universe     201종목 1회 추론(순전파만). 종목당 지연은 이 값 / 201 이다.
    activation   배치 64 순전파+역전파 중 늘어난 할당량 - 모수 메모리.
    param        모수 수 × 4바이트.

가중치는 무작위 초기화를 쓴다. 비용은 구조가 정하고 학습된 값과 무관하다.

사용:
    .venv/bin/python scripts/cost_measure.py
    .venv/bin/python scripts/cost_measure.py --batches 30 --universe-reps 5
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from src.training.train import _build_model, build_loaders  # noqa: E402
from src.utils.config import PROJECT_ROOT, load_config  # noqa: E402
from src.utils.logging import get_logger, setup_logging  # noqa: E402
from src.utils.seed import get_device, set_seed  # noqa: E402

log = get_logger("cost_measure")
REPORTS = PROJECT_ROOT / "outputs" / "reports"
ARCHS = {"patchtst_q": "PatchTST (+ quantile head)", "tft": "TFT, original",
         "phase1": "Ours"}
UNIVERSE = 201


def allocated(device: torch.device) -> int:
    if device.type == "mps":
        return int(torch.mps.current_allocated_memory())
    if device.type == "cuda":
        return int(torch.cuda.memory_allocated())
    return 0


def sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def measure(arch: str, cfg: dict, meta: dict, batches: list, device: torch.device,
            *, universe_reps: int, n_epoch_batches: int) -> dict:
    set_seed(42)
    acfg = copy.deepcopy(cfg)
    acfg["model"]["arch"] = arch
    _, _, model = _build_model(acfg, meta, tuple(meta["baseline_quantiles"]))
    model.to(device).train()
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    def step(batch):
        dyn, mac, stat, y = (t.to(device) for t in batch)
        opt.zero_grad(set_to_none=True)
        out = model(dyn, mac, stat)
        loss = out.quantiles.float().mean()
        loss.backward()
        opt.step()

    step(batches[0])                                   # 워밍업 (커널 컴파일 제외)
    sync(device)

    step_times = []
    for b in batches:
        sync(device)
        t0 = time.perf_counter()
        step(b)
        sync(device)
        step_times.append(time.perf_counter() - t0)
    per_batch = statistics.median(step_times)

    # 활성값: 역전파 그래프가 살아 있는 동안의 증가분
    dyn, mac, stat, y = (t.to(device) for t in batches[0])
    opt.zero_grad(set_to_none=True)
    if device.type == "mps":
        torch.mps.empty_cache()
    sync(device)
    base = allocated(device)
    out = model(dyn, mac, stat)
    sync(device)
    activation = max(allocated(device) - base, 0)
    del out

    # 201종목 1회 추론 (순전파만)
    model.eval()
    u_dyn, u_mac, u_stat = dyn[:1].repeat(UNIVERSE, 1, 1), mac[:1].repeat(UNIVERSE, 1, 1), \
        stat[:1].repeat(UNIVERSE, 1)
    with torch.no_grad():
        model(u_dyn, u_mac, u_stat)
        sync(device)
        reps = []
        for _ in range(universe_reps):
            t0 = time.perf_counter()
            model(u_dyn, u_mac, u_stat)
            sync(device)
            reps.append(time.perf_counter() - t0)
    universe = statistics.median(reps)

    return {
        "label": ARCHS[arch],
        "n_params": n_params,
        "sec_per_epoch": round(per_batch * n_epoch_batches, 1),
        "sec_per_batch_median": round(per_batch, 4),
        "universe_pass_ms": round(universe * 1e3, 1),
        "latency_per_stock_ms": round(universe * 1e3 / UNIVERSE, 3),
        "activation_mb": round(activation / 1e6, 1),
        "param_mb": round(n_params * 4 / 1e6, 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, default=20, help="시간을 재는 학습 배치 수")
    ap.add_argument("--universe-reps", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=64,
                    help="비용 비교용 배치. 논문 Table 5 는 64 다(설정 기본값과 다르다)")
    ap.add_argument("--epoch-batches", type=int, default=None,
                    help="에폭 배치 수를 직접 지정. 기본은 현재 로더 길이지만, 논문 4.3절의 "
                         "동결 표본(260,339 / 64 = 4,068)과 맞추려면 그 값을 준다")
    args = ap.parse_args()

    setup_logging(run_name="cost_measure")
    cfg = load_config()
    cfg["training"]["batch_size"] = args.batch_size
    device = get_device()
    loaders, meta = build_loaders(cfg)
    train = loaders["train"]
    n_epoch_batches = args.epoch_batches or len(train)

    picked = []
    for i, b in enumerate(train):
        if i >= args.batches:
            break
        picked.append(b)
    log.info("디바이스 %s | 배치 %d개 측정 | 에폭 배치 수 %d",
             device, len(picked), n_epoch_batches)

    out = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "device": device.type,
        "batch_size": int(cfg["training"]["batch_size"]),
        "lookback": int(cfg["features"]["lookback"]),
        "n_epoch_batches": n_epoch_batches,
        "loader_batches": len(train),
        "timed_batches": len(picked),
        "universe": UNIVERSE,
        "note": "s/epoch 은 배치 시간 중앙값 × 에폭 배치 수. 종목당 지연은 universe/201.",
        "models": {},
    }
    for arch in ARCHS:
        out["models"][arch] = measure(arch, cfg, meta, picked, device,
                                      universe_reps=args.universe_reps,
                                      n_epoch_batches=n_epoch_batches)
        m = out["models"][arch]
        log.info("%-12s %8.1f s/epoch | %6.1f ms/universe | %6.1f MB activation",
                 arch, m["sec_per_epoch"], m["universe_pass_ms"], m["activation_mb"])

    REPORTS.mkdir(parents=True, exist_ok=True)
    path = REPORTS / f"cost_measure_{datetime.now():%Y%m%d_%H%M%S}.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {path.relative_to(PROJECT_ROOT)}")
    print(f"{'model':<14}{'s/epoch':>10}{'lat/stock':>11}{'universe':>10}"
          f"{'activation':>12}{'param':>8}")
    for arch, m in out["models"].items():
        print(f"{arch:<14}{m['sec_per_epoch']:>10.1f}{m['latency_per_stock_ms']:>11.3f}"
              f"{m['universe_pass_ms']:>10.1f}{m['activation_mb']:>12.1f}"
              f"{m['param_mb']:>8.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
