#!/usr/bin/env python
"""창 앱 — 대시보드를 브라우저가 아니라 자체 창으로 연다.

    python scripts/desktop.py

**읽기 전용이다** (절대 규칙 9). 화면은 `scripts/dashboard.py` 와 같은
`render_html(collect_all())` 이다 — 창 껍데기만 다르고 그리는 코드는 하나다.
열 때마다 새로 모으므로 `--build` 로 만든 정적 파일처럼 낡지 않는다.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import webview
from AppKit import NSApplication, NSImage
from Foundation import NSBundle

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.webapp.collect import collect_all  # noqa: E402
from src.webapp.render import render_html  # noqa: E402

NAME = "그리드매매"


def _brand() -> None:
    """런처가 파이썬으로 exec 하면 Dock·메뉴에 'Python' 🚀 이 뜬다 — 이름과 아이콘을 되돌린다."""
    info = NSBundle.mainBundle().infoDictionary()
    info["CFBundleName"] = NAME
    icon = os.environ.get("APP_ICON")   # make_app.sh 의 런처가 넘긴다
    if icon and Path(icon).exists():
        NSApplication.sharedApplication().setApplicationIconImage_(
            NSImage.alloc().initWithContentsOfFile_(icon))


if __name__ == "__main__":
    _brand()
    webview.create_window(NAME, html=render_html(collect_all()), width=1280, height=900)
    webview.start(func=_brand)
