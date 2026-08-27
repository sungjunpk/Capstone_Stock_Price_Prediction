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


UNIVERSE_PATH = PROJECT_ROOT / "configs" / "universe.yaml"


def _deep_merge(base: dict, over: dict) -> dict:
    """over 를 base 위에 재귀 병합. 리스트는 통째로 교체한다."""
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(
    path: str | Path = DEFAULT_CONFIG_PATH, *, profile: str | None = None
) -> Config:
    """config.yaml 로드. profile 을 주면 `profiles.<이름>` 을 위에 덮어쓴다.

    설정 파일을 갈라놓지 않는 이유: 값이 두 곳에 흩어지면 어느 쪽이 진짜인지
    금방 모르게 된다. 프로파일은 **차이만** 적고 나머지는 base 를 그대로 쓴다.
    """
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    if profile:
        profiles = raw.get("profiles") or {}
        if profile not in profiles:
            raise ValueError(
                f"알 수 없는 프로파일: {profile} (있는 것: {sorted(profiles)})"
            )
        raw = _deep_merge(raw, profiles[profile] or {})
        raw["active_profile"] = profile

    # 유니버스는 scripts/build_universe.py 가 생성하는 별도 파일이 정본이다.
    # 종목이 수백 개라 config.yaml 에 인라인으로 두면 읽기 힘들어서 분리했다.
    # universe.yaml 이 없으면 config.yaml 의 인라인 목록으로 폴백한다.
    if UNIVERSE_PATH.exists():
        with open(UNIVERSE_PATH, encoding="utf-8") as f:
            uni = yaml.safe_load(f) or {}
        if uni.get("universe"):
            raw.setdefault("data", {})["universe"] = uni["universe"]
            raw["data"]["universe_meta"] = {
                k: v for k, v in uni.items() if k != "universe"
            }

    return Config(raw=raw, kiwoom=load_kiwoom_settings())
