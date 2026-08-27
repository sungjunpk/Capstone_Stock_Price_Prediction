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


XS_PREFIX = "xs_"


def dynamic_feature_columns(panel: pd.DataFrame) -> list[str]:
    """패널에서 모델 입력이 되는 동적 피처 컬럼만 고른다.

    횡단면 피처(`xs_`)는 **항상 맨 뒤로** 보낸다. 모델이 뒤에서 N개를 잘라
    RevIN 을 건너뛰기 때문에, 순서가 곧 계약이다 (Phase1Config.n_passthrough).
    """
    cols = [c for c in panel.columns if c not in set(BASE_COLS) | {TARGET_COL}]
    return ([c for c in cols if not c.startswith(XS_PREFIX)]
            + [c for c in cols if c.startswith(XS_PREFIX)])


def n_passthrough_columns(feature_cols: list[str]) -> int:
    """RevIN 을 건너뛸 채널 수 = 뒤쪽 `xs_` 컬럼 개수."""
    return sum(1 for c in feature_cols if c.startswith(XS_PREFIX))


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
        require_target: bool = True,
    ):
        self.lookback = int(lookback)
        self.feature_cols = list(feature_cols)
        self.vocab = vocab
        # 학습/백테스트는 타깃이 있는 행만 쓴다.
        # 모의투자는 **가장 최근 윈도우**를 써야 하는데 그 구간은 아직 t+h 가 오지 않아
        # 타깃이 NaN 이다. require_target=False 로 그 행들을 살린다.
        # ⚠️ 이때 나오는 target 값은 의미 없는 0 이다 — 학습에 쓰면 안 된다.
        self.require_target = bool(require_target)

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
        self._codes: list[str] = []
        self._dates: list[np.ndarray] = []
        index: list[tuple[int, int]] = []

        skipped = 0
        for code, part in panel.groupby("code", sort=True):
            part = part.sort_values("date")
            required = self.feature_cols + ([TARGET_COL] if self.require_target else [])
            usable = part.dropna(subset=required)
            if len(usable) <= self.lookback:
                skipped += 1
                continue

            si = len(self._arrays)
            self._arrays.append(usable[self.feature_cols].to_numpy(dtype=np.float32))
            self._targets.append(
                usable[TARGET_COL].fillna(0.0).to_numpy(dtype=np.float32)
                if TARGET_COL in usable.columns
                else np.zeros(len(usable), dtype=np.float32)
            )
            self._codes.append(str(code))

            # ⚠️ 원본 dtype 그대로 보관한다. pd.to_datetime 으로 바꿔 담으면
            #    sample_keys 의 date 가 Timestamp 가 되어 주가 테이블(date 객체)과
            #    키가 안 맞고, 백테스트가 조용히 리밸런싱을 0회 한다(실측).
            self._dates.append(usable["date"].to_numpy())
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

    def sample_keys(self) -> pd.DataFrame:
        """윈도우 i → (code, date, target). 예측을 종목·날짜에 붙일 때 쓴다.

        예측 텐서의 i 번째 행이 어느 종목의 어느 날짜인지는 **여기 하나로만** 복원한다.
        호출자가 필터·정렬을 따라 재현하면 조건이 한 곳만 어긋나도 예측이
        엉뚱한 종목에 붙고, 지표는 그럴듯하게 나온다(조용히 틀린다).

        ⚠️ target 은 사후 평가(랭크 IC) 전용이다. 매매 판단에 넣으면 look-ahead 다.
        """
        si = self._index[:, 0]
        end = self._index[:, 1]
        return pd.DataFrame({
            "code": [self._codes[i] for i in si],
            "date": [self._dates[i][e] for i, e in zip(si, end, strict=True)],
            "target": [self._targets[i][e] for i, e in zip(si, end, strict=True)],
        })

    def latest_rows(self) -> list[int]:
        """종목별 **마지막** 윈도우의 샘플 인덱스. 모의투자는 이것만 필요하다."""
        last: dict[int, int] = {}
        for row, (si, _) in enumerate(self._index):
            last[int(si)] = row          # _index 는 종목별로 오름차순이라 마지막이 최신
        return [last[k] for k in sorted(last)]

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
