"""송출 주소를 환경변수 또는 Git에서 제외한 로컬 설정에서 읽는다."""

import json
import os
from functools import lru_cache
from pathlib import Path


SETTINGS_PATH = Path(__file__).resolve().parent / "local_settings.json"


@lru_cache(maxsize=1)
def _local_settings():
    if not SETTINGS_PATH.exists():
        return {}
    with SETTINGS_PATH.open(encoding="utf-8") as source:
        settings = json.load(source)
    if not isinstance(settings, dict):
        raise ValueError("local_settings.json은 JSON 객체여야 합니다.")
    return settings


def get_setting(name, default=""):
    """환경변수 > local_settings.json > 기본값 순서로 문자열을 반환한다."""
    if name in os.environ:
        return os.environ[name]
    value = _local_settings().get(name, default)
    if not isinstance(value, str):
        raise ValueError(f"{name} 설정값은 문자열이어야 합니다.")
    return value
