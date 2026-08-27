"""분봉 수집 — 네트워크 없이 증분·미완성봉 방어만 검증한다.

여기서 지키려는 것:
  1) 진행 중인 봉은 저장되지 않는다 (일봉의 '장중 미완성 봉' 문제와 같은 것)
  2) 같은 명령을 두 번 돌려도 중복 행이 생기지 않는다 (CLAUDE.md 절대 규칙 4)
  3) 틱범위가 다르면 저장 위치가 다르다 — 30분봉과 60분봉이 섞이면 안 된다
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from src.data import storage
from src.data.kiwoom import endpoints as ep
from src.data.kiwoom.collect import (
    collect_minute_chart,
    drop_incomplete_bars,
    minute_kind,
)


def _bars(day: str, hours: list[int]) -> list[dict]:
    """키움 응답 형태의 60분봉 레코드. 응답은 최신→과거 내림차순이다."""
    return [
        {
            "cntr_tm": f"{day}{h:02d}0000",
            "open_pric": "+100", "high_pric": "+110",
            "low_pric": "+90", "cur_prc": "+105", "trde_qty": "1,000",
        }
        for h in sorted(hours, reverse=True)
    ]


class FakePagedClient:
    """paginate 가 정해진 페이지들을 순서대로 내보내는 클라이언트."""

    def __init__(self, pages: list[list[dict]]):
        self.pages = pages
        self.bodies: list[dict] = []

    def paginate(self, spec, body=None):
        self.bodies.append(body or {})
        for page in self.pages:
            yield {spec.list_key: page}


class TestIncompleteBars:
    def test_in_progress_bar_is_dropped(self):
        """11:45 에 받으면 11:00 봉은 아직 진행 중이다 — 종가가 확정되지 않았다."""
        df = pd.DataFrame({"datetime": pd.to_datetime(
            ["2026-08-27 09:00", "2026-08-27 10:00", "2026-08-27 11:00"])})
        kept = drop_incomplete_bars(df, "60", now=datetime(2026, 8, 27, 11, 45))
        assert list(pd.to_datetime(kept["datetime"]).dt.hour) == [9, 10]

    def test_completed_bar_survives(self):
        df = pd.DataFrame({"datetime": pd.to_datetime(["2026-08-27 11:00"])})
        kept = drop_incomplete_bars(df, "60", now=datetime(2026, 8, 27, 12, 0))
        assert len(kept) == 1

    def test_shorter_tick_waits_less(self):
        """5분봉은 5분만 지나면 확정이다 — 틱범위를 그대로 쓴다."""
        df = pd.DataFrame({"datetime": pd.to_datetime(["2026-08-27 11:00"])})
        assert len(drop_incomplete_bars(df, "5", now=datetime(2026, 8, 27, 11, 6))) == 1
        assert len(drop_incomplete_bars(df, "5", now=datetime(2026, 8, 27, 11, 3))) == 0


class TestKind:
    def test_tick_scopes_do_not_share_a_directory(self):
        assert minute_kind("60") != minute_kind("30")
        assert minute_kind("60", index=True) != minute_kind("60")


class TestIncrementalCollect:
    def test_rerun_adds_no_duplicate_rows(self, tmp_path, monkeypatch):
        monkeypatch.setattr(storage, "RAW_DIR", tmp_path)
        now = datetime(2026, 8, 27, 16, 0)   # 장 마감 후 — 전 봉이 확정된 시점
        monkeypatch.setattr(
            "src.data.kiwoom.collect.datetime",
            type("D", (), {"now": staticmethod(lambda: now)}),
        )
        pages = [_bars("20260827", [9, 10, 11, 12, 13, 14, 15])]

        first = collect_minute_chart(FakePagedClient(pages), "005930")
        assert len(first) == 7

        second = collect_minute_chart(FakePagedClient(pages), "005930")
        assert len(second) == 7, "재실행이 행을 늘리면 증분 계약이 깨진 것이다"

        saved = storage.read_parquet(storage.raw_path(minute_kind("60"), "005930"))
        assert saved["datetime"].is_monotonic_increasing
        assert not saved["datetime"].duplicated().any()

    def test_stops_early_once_it_reaches_stored_bars(self, tmp_path, monkeypatch):
        """이미 가진 구간에 닿으면 다음 페이지를 받지 않는다."""
        monkeypatch.setattr(storage, "RAW_DIR", tmp_path)
        now = datetime(2026, 8, 27, 16, 0)
        monkeypatch.setattr(
            "src.data.kiwoom.collect.datetime",
            type("D", (), {"now": staticmethod(lambda: now)}),
        )
        path = storage.raw_path(minute_kind("60"), "005930")
        storage.upsert(
            pd.DataFrame({"datetime": pd.to_datetime(["2026-08-26 15:00"]),
                          "close": [100.0]}),
            path, key=["datetime"], sort_by=["datetime"],
        )

        client = FakePagedClient([
            _bars("20260827", [9, 10, 11]),   # 새 구간
            _bars("20260825", [9, 10, 11]),   # 보유 구간 아래 — 여기서 멈춰야 한다
            _bars("20260820", [9, 10, 11]),   # 받으면 안 되는 페이지
        ])
        out = collect_minute_chart(client, "005930")
        assert pd.Timestamp("2026-08-20 09:00") not in set(out["datetime"])

    def test_request_body_carries_tick_and_adjusted_price(self):
        client = FakePagedClient([[]])
        collect_minute_chart(client, "005930", tic_scope="30")
        body = client.bodies[0]
        assert body["tic_scope"] == "30"
        assert body["upd_stkpc_tp"] == "1"   # 끄면 분할 구간이 망가진다


def test_minute_spec_has_no_value_column():
    """일봉과 달리 거래대금이 없다 — 피처가 이걸 전제하면 안 된다."""
    assert "value" not in ep.MINUTE_CHART.schema
    assert ep.MINUTE_CHART.schema["datetime"] == ("cntr_tm", "datetime")
