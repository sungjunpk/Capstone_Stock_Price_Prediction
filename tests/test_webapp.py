"""대시보드 테스트 — 화면이 **없는 데이터로도 죽지 않는가**가 핵심이다.

대시보드는 파이프라인의 곁가지라, 리포트가 하나 비었다고 예외가 나면
정작 결과를 봐야 할 때 못 본다. 그래서 빈 상태·부분 상태를 먼저 검증한다.
"""

from __future__ import annotations

import json

import pytest

from src.webapp import collect as wc
from src.webapp.render import render_html


def test_renders_with_nothing():
    """실행 기록도 리포트도 없는 초기 상태."""
    html = render_html({"generated_at": "2026-08-25T10:00:00"})
    assert "<html" in html and "주가예측" in html
    assert "실행 기록 없음" in html


def test_renders_without_crashing_on_partial_data():
    html = render_html({
        "generated_at": "x",
        "run": {"dry_run": True, "plan": {}},
        "backtest": {},
        "training": None,
        "data": {"available": False},
    })
    assert "<html" in html


def test_escapes_untrusted_text():
    """리포트 문자열이 그대로 HTML 로 들어가면 안 된다."""
    html = render_html({
        "generated_at": "x",
        "run": {"dry_run": True, "plan": {
            "decision_date": "2026-08-24", "generated_at": "x",
            "orders": [], "notes": ["<script>alert(1)</script>"], "stats": {},
        }},
    })
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def _write(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


@pytest.fixture
def reports(tmp_path, monkeypatch):
    monkeypatch.setattr(wc, "REPORTS_DIR", tmp_path)
    return tmp_path


class TestReportSelection:
    def test_picks_newest_plain_run_not_oldest(self, reports):
        """같은 세션에 실행이 여러 번이면 **최근 것**이 대표다.

        예전 실행을 고르면 화면에 옛 숫자(때로는 0)가 뜨고, 그걸 최신으로 읽게 된다.
        """
        _write(reports / "backtest_a.json", {
            "checkpoint": "m.pt", "timestamp": "2026-08-25T17:23:00", "variant": "",
            "strategy": {"sharpe": 0.0},
        })
        _write(reports / "backtest_b.json", {
            "checkpoint": "m.pt", "timestamp": "2026-08-25T17:24:00", "variant": "",
            "strategy": {"sharpe": 1.1},
        })
        main = wc.latest_backtests()["main"]
        assert main["strategy"]["sharpe"] == 1.1

    def test_ignores_other_checkpoints(self, reports):
        _write(reports / "backtest_a.json", {
            "checkpoint": "old.pt", "timestamp": "2026-08-25T17:00:00", "variant": "",
            "strategy": {"sharpe": 9.9},
        })
        _write(reports / "backtest_b.json", {
            "checkpoint": "new.pt", "timestamp": "2026-08-25T17:24:00", "variant": "",
            "strategy": {"sharpe": 1.1},
        })
        session = wc.latest_backtests()
        assert {v["checkpoint"] for v in session["variants"]} == {"new.pt"}

    def test_flags_smoke_training_report(self, reports):
        """캐글 학습이면 로컬에 실제 리포트가 없다 — smoke 로 물러나되 표시해야 한다."""
        _write(reports / "20260825_142330_e592a778_smoke.json",
               {"smoke": True, "n_params": 1, "feature_importance": {"a": 1.0}})
        tr = wc.latest_training()
        assert tr["_smoke"] is True

        html = render_html({"generated_at": "x", "training": tr, "checkpoint": {"name": "real.pt"}})
        assert "스모크 실행" in html

    def test_warns_when_report_and_checkpoint_differ(self, reports):
        html = render_html({
            "generated_at": "x",
            "checkpoint": {"name": "phase1_new.pt"},
            "training": {"checkpoint": "outputs/checkpoints/phase1_old.pt",
                         "feature_importance": {"a": 1.0}, "_smoke": False},
        })
        assert "다르다" in html


class TestPerformanceSections:
    """성과 섹션은 기록이 없거나 하루뿐일 때도 죽지 않아야 한다.

    대시보드는 파이프라인의 곁가지라, 리포트 하나 비었다고 예외가 나면
    정작 결과를 봐야 할 때 못 본다.
    """

    def test_renders_with_no_records(self):
        html = render_html({"generated_at": "x"})
        assert "누적 수익률" in html and "기록 없음" in html

    def test_renders_with_single_day(self):
        """점이 하나면 곡선을 그리지 않지만 타일은 나온다."""
        html = render_html({
            "generated_at": "x",
            "performance": {
                "n_days": 1, "start_date": "2026-08-26", "end_date": "2026-08-26",
                "start_equity": 1e8, "equity": 1e8, "total_return": 0.0,
                "reliable": False, "min_days_for_metrics": 20,
                "baseline_seeded": True, "baseline_in_curve": False,
                "realized_pnl": 0.0, "fee_tax": 0.0,
            },
            "equity": [{"date": "2026-08-26", "equity": 1e8}],
        })
        assert "<svg" not in html            # 한 점으로는 곡선을 그리지 않는다
        assert "해석이 아니라 착시" in html    # 짧은 표본 경고가 뜬다

    def test_holdings_survive_an_empty_plan(self):
        """실행 계획이 비어도 보유 현황은 나와야 한다.

        2026-08-31 회귀: 마지막 실행에 신호가 없자 9종목을 들고 있는데도
        보유 현황 섹션이 통째로 사라졌다. 보유는 계좌 기록에서 온다.
        """
        html = render_html({
            "generated_at": "x",
            "run": {"plan": {"signals": []}},
            "holdings": [
                {"date": "2026-08-28", "code": "024110", "name": "기업은행",
                 "quantity": 496, "eval_amount": 10_341_600, "weight": 0.1027,
                 "pnl_rate": 0.0321},
                {"date": "2026-08-28", "code": "CASH", "name": "현금",
                 "quantity": None, "eval_amount": 16_063_315, "weight": 0.1595,
                 "pnl_rate": 0.0},
            ],
        })
        assert "보유 현황" in html
        assert "024110" in html and "기업은행" in html
        assert "10.3%" in html and "3.21%" in html
        assert "현금" in html          # 현금 행이 있어야 비중 합이 100% 로 읽힌다

    def test_daily_return_tile_shows_when_recorded(self):
        html = render_html({
            "generated_at": "x",
            "performance": {
                "n_days": 3, "start_date": "2026-08-25", "end_date": "2026-08-27",
                "start_equity": 1e8, "equity": 9.9e7, "total_return": -0.01,
                "reliable": False, "min_days_for_metrics": 20,
                "baseline_seeded": True, "baseline_in_curve": True,
                "realized_pnl": 0.0, "fee_tax": 0.0,
                "daily_return": 0.0198, "daily_date": "2026-08-27",
            },
        })
        assert "일간 수익률" in html
        assert "1.98%" in html and "2026-08-27 종가 기준" in html

    def test_daily_return_tile_absent_on_first_day(self):
        """전일이 없으면 타일을 띄우지 않는다 — 0.00% 는 보합과 구분되지 않는다."""
        html = render_html({
            "generated_at": "x",
            "performance": {
                "n_days": 1, "start_date": "2026-08-25", "end_date": "2026-08-25",
                "start_equity": 1e8, "equity": 1e8, "total_return": 0.0,
                "reliable": False, "min_days_for_metrics": 20,
                "baseline_seeded": True, "baseline_in_curve": False,
                "realized_pnl": 0.0, "fee_tax": 0.0,
            },
        })
        assert "일간 수익률" not in html

    def test_curve_is_drawn_from_two_points(self):
        """점이 둘이면 곡선을 그린다."""
        html = render_html({
            "generated_at": "x",
            "performance": {
                "n_days": 2, "start_date": "2026-08-25", "end_date": "2026-08-26",
                "start_equity": 1e8, "equity": 9.9e7, "total_return": -0.01,
                "reliable": False, "min_days_for_metrics": 20,
                "baseline_seeded": True, "baseline_in_curve": True,
                "realized_pnl": -3118.0, "fee_tax": 317358.0,
            },
            "equity": [
                {"date": "2026-08-25", "equity": 1e8},
                {"date": "2026-08-26", "equity": 9.9e7},
            ],
        })
        assert "<svg" in html and "polyline" in html

    def test_attribution_marks_unsold_as_holding(self):
        html = render_html({
            "generated_at": "x",
            "attribution": [{
                "code": "005930", "name": "삼성전자", "days": 1,
                "buy_qty": 10, "sell_qty": 0, "buy_amount": 2.6e6,
                "sell_amount": 0.0, "pnl_amount": 0.0, "fee_tax": 9100.0,
            }],
        })
        assert "보유중" in html and "삼성전자" in html
