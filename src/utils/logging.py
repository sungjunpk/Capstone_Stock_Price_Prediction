"""프로젝트 공용 로거. 콘솔 + outputs/logs 파일 동시 출력."""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

from src.utils.config import PROJECT_ROOT

_FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_configured = False


def setup_logging(level: int = logging.INFO, run_name: str | None = None) -> Path:
    """루트 로거 설정. 로그 파일 경로를 반환."""
    global _configured

    log_dir = PROJECT_ROOT / "outputs" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{run_name or 'run'}_{stamp}.log"

    if _configured:
        return log_path

    root = logging.getLogger()
    root.setLevel(level)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(_FMT, datefmt="%H:%M:%S"))
    root.addHandler(console)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(_FMT))
    root.addHandler(file_handler)

    _configured = True
    return log_path


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
