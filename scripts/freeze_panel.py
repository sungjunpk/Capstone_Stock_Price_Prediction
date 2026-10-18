#!/usr/bin/env python
"""학습에 들어가는 자료를 **동결**한다 — 어느 시점 자료로 돌린 결과인지 남긴다.

왜 필요한가: 논문 4.3절의 학습 윈도우 260,339 와 현재 로더의 264,653 이 다르다.
학습 구간은 2022년까지라 검정구간 연장으로는 설명되지 않는다 — 수집·전처리가 중간에
바뀌었다는 뜻이고, 그러면 〈Table 4〉와 〈Table 6〉이 같은 자료에서 나왔다고 보장할 수 없다.

무엇을 남기는가: panel/macro/static 의 sha256 과 행·종목 수·기간, 그리고 train/val/test
윈도우 수. 캐글에서 같은 스크립트를 `--verify` 로 돌려 대조한다.

⚠️ 대조 기준은 **윈도우 수·행 수·기간·피처 수**다. sha256 은 비교하지 않는다 —
`package_data.py` 가 학습에 안 쓰는 열을 떼고 float32 로 다시 쓰므로 캐글 쪽 파일은
바이트가 달라진다. 해시는 '같은 로컬 파일인지' 확인할 때만 쓴다.

사용:
    python scripts/freeze_panel.py                                  # 기록
    python scripts/freeze_panel.py --verify outputs/reports/freeze_*.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.data.storage import PROCESSED_DIR  # noqa: E402
from src.training.train import build_loaders  # noqa: E402
from src.utils.config import PROJECT_ROOT, load_config  # noqa: E402
from src.utils.logging import get_logger, setup_logging  # noqa: E402

log = get_logger("freeze_panel")
REPORTS = PROJECT_ROOT / "outputs" / "reports"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot(cfg: dict) -> dict:
    sfx = cfg["data"].get("processed_suffix", "")
    files = {}
    for name in ("panel", "macro", "static"):
        p = PROCESSED_DIR / f"{name}{sfx}.parquet"
        df = pd.read_parquet(p)
        files[p.name] = {
            "sha256": sha256(p),
            "bytes": p.stat().st_size,
            "rows": int(len(df)),
            "cols": int(df.shape[1]),
            "codes": int(df["code"].nunique()) if "code" in df else None,
            "date_min": str(df["date"].min()) if "date" in df else None,
            "date_max": str(df["date"].max()) if "date" in df else None,
        }
    loaders, meta = build_loaders(cfg)
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "profile_suffix": sfx,
        "files": files,
        "windows": {k: int(len(v.dataset)) for k, v in loaders.items()},
        "batch_size": int(cfg["training"]["batch_size"]),
        "split": meta["split"] if isinstance(meta.get("split"), dict) else str(meta.get("split")),
        "n_features": len(meta["feature_cols"]),
        "n_macro": len(meta["macro_cols"]),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", type=Path, help="이 기록과 대조만 하고 저장하지 않는다")
    args = ap.parse_args()

    setup_logging(run_name="freeze_panel")
    snap = snapshot(load_config())

    for name, f in snap["files"].items():
        print(f"  {name:<22} {f['sha256'][:16]}…  {f['rows']:>9,}행 {f['cols']:>3}열 "
              f"{f['date_min']}~{f['date_max']}")
    print(f"  윈도우  train {snap['windows']['train']:,} / "
          f"val {snap['windows']['val']:,} / test {snap['windows']['test']:,}")

    if args.verify:
        ref = json.loads(args.verify.read_text(encoding="utf-8"))
        bad = [f"windows.{k}" for k, v in snap["windows"].items()
               if ref["windows"].get(k) != v]
        for n, f in snap["files"].items():
            r = ref["files"].get(n, {})
            bad += [f"{n}.{key}" for key in ("rows", "date_min", "date_max")
                    if r.get(key) != f[key]]
        bad += [k for k in ("n_features", "n_macro") if ref.get(k) != snap[k]]
        if bad:
            print(f"\n❌ 자료가 다르다: {', '.join(bad)}")
            return 1
        # sha256 은 다를 수 있다 — package_data.py 가 열을 떼고 float32 로 다시 쓴다.
        same_hash = all(ref["files"].get(n, {}).get("sha256") == f["sha256"]
                        for n, f in snap["files"].items())
        print(f"\n✅ {args.verify.name} 과 같은 자료다"
              f"{'' if same_hash else ' (sha256 은 다르다 — 재패키징 때문이며 정상이다)'}")
        return 0

    REPORTS.mkdir(parents=True, exist_ok=True)
    path = REPORTS / f"freeze_{datetime.now():%Y%m%d_%H%M%S}.json"
    path.write_text(json.dumps(snap, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
