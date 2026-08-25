"""윈도우 Dataset — 슬라이딩 윈도우를 **복제하지 않고** 뷰로 낸다.

왜 중요한가:
    학습샘플 35만 × lookback 120 × 피처 17 × 4바이트 = 2.9GB.
    윈도우를 미리 펼쳐 만들면 메모리가 터진다.
    종목별 연속 배열은 다 합쳐도 28MB뿐이므로, 배열은 한 번만 만들고
    __getitem__ 에서 구간을 잘라 준다.

leakage 방지:
    - 윈도우는 **분할된 구간 안에서만** 만든다. train 윈도우가 val 행을 보지 않는다.
    - 정규화 통계는 src/training/split.py 의 fit_normalizer 로 train 에서만 계산한다.
    - 종목마다 거래일이 다르므로(공통 1,887 / 전체 2,857) 공통 캘린더를 강요하지 않고
      각 종목의 자기 행 위에서 윈도우를 만든다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.utils.logging import get_logger

log = get_logger(__name__)

BASE_COLS = ("code", "date", "open", "high", "low", "close", "volume", "value")
TARGET_COL = "target"


def dynamic_feature_columns(panel: pd.DataFrame) -> list[str]:
    """패널에서 모델 입력이 되는 동적 피처 컬럼만 고른다."""
    return [c for c in panel.columns if c not in set(BASE_COLS) | {TARGET_COL}]


@dataclass(frozen=True)
class StaticVocab:
    """범주형 static covariate 의 코드북. **train 기준으로 한 번 만들어 공유한다.**"""

    sector: dict[str, int]
    size_class: dict[str, int]
    market_cap_bucket: dict[int, int]

    @property
    def sizes(self) -> dict[str, int]:
        # +1 은 미등록/결측용 0번 슬롯
        return {
            "sector": len(self.sector) + 1,
            "size_class": len(self.size_class) + 1,
            "market_cap_bucket": len(self.market_cap_bucket) + 1,
            "day_of_week": 6,  # 월~금 + 결측
        }

    @classmethod
    def build(cls, static: pd.DataFrame) -> StaticVocab:
        def codebook(values) -> dict:
            uniq = sorted({v for v in values if pd.notna(v)}, key=str)
            return {v: i + 1 for i, v in enumerate(uniq)}

        return cls(
            sector=codebook(static["sector"]),
            size_class=codebook(static.get("size_class", pd.Series(dtype=object))),
            market_cap_bucket=codebook(static.get("market_cap_bucket", pd.Series(dtype=object))),
        )


class WindowDataset(Dataset):
    """(dynamic, macro, static, target) 을 내는 윈도우 데이터셋.

    반환:
        dynamic (L, C_dyn)  float32
        macro   (L, C_mac)  float32
        static  (4,)        int64  — sector/size/mcap/dow
        target  ()          float32
    """

    def __init__(
        self,
        panel: pd.DataFrame,
        macro: pd.DataFrame,
        static: pd.DataFrame,
        *,
        lookback: int,
        feature_cols: list[str],
        vocab: StaticVocab,
    ):
        self.lookback = int(lookback)
        self.feature_cols = list(feature_cols)
        self.vocab = vocab

        # --- 매크로: 날짜 → 행 인덱스 맵. 윈도우마다 날짜로 조회한다
        macro = macro.sort_values("date").reset_index(drop=True)
        self.macro_cols = [c for c in macro.columns if c != "date"]
        self._macro = macro[self.macro_cols].to_numpy(dtype=np.float32)
        self._macro_row = {d: i for i, d in enumerate(macro["date"])}

        # --- static: 종목 → 범주 인덱스
        self._static_by_code = self._encode_static(static, vocab)

        # --- 종목별 연속 배열 + 윈도우 인덱스 테이블
        self._arrays: list[np.ndarray] = []
        self._targets: list[np.ndarray] = []
        self._macro_rows: list[np.ndarray] = []
        self._dow: list[np.ndarray] = []
        self._static: list[np.ndarray] = []
        index: list[tuple[int, int]] = []

        skipped = 0
        for code, part in panel.groupby("code", sort=True):
            part = part.sort_values("date")
            usable = part.dropna(subset=self.feature_cols + [TARGET_COL])
            if len(usable) <= self.lookback:
                skipped += 1
                continue

            si = len(self._arrays)
            self._arrays.append(usable[self.feature_cols].to_numpy(dtype=np.float32))
            self._targets.append(usable[TARGET_COL].to_numpy(dtype=np.float32))

            dates = pd.to_datetime(usable["date"])
            # 매크로에 없는 날짜는 -1 → 0벡터로 채운다(정상 상황에선 발생하지 않는다)
            self._macro_rows.append(
                np.array([self._macro_row.get(d.date(), -1) for d in dates], dtype=np.int64)
            )
            self._dow.append(dates.dt.dayofweek.to_numpy(dtype=np.int64))
            self._static.append(self._static_by_code.get(code, np.zeros(3, dtype=np.int64)))

            # end 는 윈도우의 마지막 행(포함). 그 행의 target 을 맞춘다.
            index.extend((si, end) for end in range(self.lookback - 1, len(usable)))

        self._index = np.asarray(index, dtype=np.int64)
        if skipped:
            log.warning("행이 lookback(%d) 이하라 제외된 종목 %d개", self.lookback, skipped)

        n_bytes = sum(a.nbytes for a in self._arrays)
        log.info(
            "WindowDataset: 샘플 %s개 / 종목 %d개 / 배열 %.0fMB (윈도우 복제 없음)",
            f"{len(self._index):,}", len(self._arrays), n_bytes / 1e6,
        )

    @staticmethod
    def _encode_static(static: pd.DataFrame, vocab: StaticVocab) -> dict[str, np.ndarray]:
        out = {}
        for r in static.itertuples():
            out[r.code] = np.array(
                [
                    vocab.sector.get(getattr(r, "sector", None), 0),
                    vocab.size_class.get(getattr(r, "size_class", None), 0),
                    vocab.market_cap_bucket.get(getattr(r, "market_cap_bucket", None), 0),
                ],
                dtype=np.int64,
            )
        return out

    @property
    def n_dynamic(self) -> int:
        return len(self.feature_cols)

    @property
    def n_macro(self) -> int:
        return len(self.macro_cols)

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, i: int):
        si, end = self._index[i]
        lo = end - self.lookback + 1

        dyn = self._arrays[si][lo : end + 1]          # 뷰 — 복사 아님
        rows = self._macro_rows[si][lo : end + 1]
        mac = np.where(
            (rows >= 0)[:, None], self._macro[np.clip(rows, 0, None)], 0.0
        ).astype(np.float32)

        stat = np.concatenate([self._static[si], [self._dow[si][end] + 1]])
        y = self._targets[si][end]

        return (
            torch.from_numpy(np.ascontiguousarray(dyn)),
            torch.from_numpy(mac),
            torch.from_numpy(stat),
            torch.tensor(y, dtype=torch.float32),
        )
