"""설정 로딩: configs/config.yaml + .env 를 합쳐 하나의 객체로.

경로 하드코딩과 URL 하드코딩을 막기 위한 단일 진입점.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

# 문서 확인 후 다르면 .env 로 덮어쓴다. 코드 안에서 이 값을 직접 참조하지 말 것.
_DEFAULT_BASE_URLS = {
    "mock": "https://mockapi.kiwoom.com",  # UNVERIFIED — MCP 응답으로 검증 필요
    "live": "https://api.kiwoom.com",      # UNVERIFIED — 이 프로젝트에서는 사용 금지
}


@dataclass(frozen=True)
class KiwoomSettings:
    env: str
    base_url: str
    app_key: str | None
    app_secret: str | None
    account_no: str | None
    rate_limit_per_sec: float

    @property
    def is_mock(self) -> bool:
        return self.env == "mock"

    def require_credentials(self) -> None:
        missing = [
            name
            for name, val in (
                ("KIWOOM_APP_KEY", self.app_key),
                ("KIWOOM_APP_SECRET", self.app_secret),
            )
            if not val
        ]
        if missing:
            raise RuntimeError(
                f".env 에 다음 값이 없다: {', '.join(missing)} "
                f"(.env.example 참고)"
            )


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any] = field(repr=False)
    kiwoom: KiwoomSettings

    # 자주 쓰는 경로
    project_root: Path = PROJECT_ROOT

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    def path(self, *parts: str) -> Path:
        return self.project_root.joinpath(*parts)


def load_kiwoom_settings() -> KiwoomSettings:
    load_dotenv(PROJECT_ROOT / ".env")

    env = os.getenv("KIWOOM_ENV", "mock").strip().lower()
    if env not in _DEFAULT_BASE_URLS:
        raise ValueError(f"KIWOOM_ENV 는 'mock' 또는 'live' — 받은 값: {env!r}")
    if env == "live":
        # CLAUDE.md 규칙: 이 프로젝트는 실전투자를 사용하지 않는다.
        raise RuntimeError(
            "KIWOOM_ENV=live 는 이 프로젝트에서 허용되지 않는다. "
            "모의투자(mock)로만 실행할 것."
        )

    override = os.getenv(f"KIWOOM_BASE_URL_{env.upper()}")
    return KiwoomSettings(
        env=env,
        base_url=(override or _DEFAULT_BASE_URLS[env]).rstrip("/"),
        app_key=os.getenv("KIWOOM_APP_KEY") or None,
        app_secret=os.getenv("KIWOOM_APP_SECRET") or None,
        account_no=os.getenv("KIWOOM_ACCOUNT_NO") or None,
        rate_limit_per_sec=float(os.getenv("KIWOOM_RATE_LIMIT_PER_SEC", "3")),
    )


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Config:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return Config(raw=raw, kiwoom=load_kiwoom_settings())
