from __future__ import annotations

import logging
from typing import Any, Optional, Type

import pytz
from pydantic import BaseModel, Field

from src.domain.tool import BaseTool

logger = logging.getLogger(__name__)


class NationToolInput(BaseModel):
    country: str = Field(
        ...,
        description="입력값. 예: ISO 국가코드(KR/US/GB), 국가명(United Kingdom/영국), 또는 타임존(Europe/London)",
    )


class NationTool(BaseTool):
    def __init__(self) -> None:
        # timezone -> country code 역인덱스 (가장 먼저 발견된 code를 사용)
        self._tz_to_country: dict[str, str] = {}
        for code, tzs in pytz.country_timezones.items():
            for tz in tzs:
                self._tz_to_country.setdefault(tz, code)

    @property
    def name(self) -> str:
        return "nation_tool"
    
    @property
    def description(self) -> str:
        return (
            "국가/도시/타임존 입력을 받아 국가코드/표준 타임존 등 '국가 정보'를 JSON으로 반환합니다. "
            "시간 계산은 하지 않으며, 시간은 time_tool로 조회하세요."
        )

    @property
    def input_model(self) -> Type[BaseModel]:
        return NationToolInput
	    
    def execute(self, country: str, **_: Any) -> dict:
        raw = (country or "").strip()
        if not raw:
            raise ValueError("country is required")

        def _resolve(value: str) -> tuple[Optional[str], Optional[str]]:
            # 1) ISO 3166-1 alpha-2 국가코드 (GB, KR, US 등)
            # pytz에는 "GB" 같은 timezone alias도 존재하므로, 2자리 코드는 국가코드로 우선 해석한다.
            code = value.upper()
            if len(code) == 2 and code in pytz.country_timezones:
                return pytz.country_timezones[code][0], code

            # 2) IANA timezone 직접 입력(Europe/London 등)
            if value in pytz.all_timezones_set:
                return value, self._tz_to_country.get(value)

            # 3) 국가명(영문) -> 코드 매핑(pytz 내장 country_names)
            #    pytz.country_names: { 'GB': 'United Kingdom', ... }
            normalized = value.casefold()
            for c, name in pytz.country_names.items():
                if name.casefold() == normalized:
                    tzs = pytz.country_timezones.get(c)
                    if tzs:
                        return tzs[0], c

            # 4) 자주 쓰는 별칭(한글/약어) 최소 지원
            aliases = {
                "uk": "GB",
                "u.k.": "GB",
                "united kingdom": "GB",
                "britain": "GB",
                "영국": "GB",
                "런던": "Europe/London",
                "london": "Europe/London",
                "korea": "KR",
                "south korea": "KR",
                "한국": "KR",
                "대한민국": "KR",
                "seoul": "Asia/Seoul",
                "서울": "Asia/Seoul",
                "미국": "US",
                "usa": "US",
                "united states": "US",
                "new york": "America/New_York",
                "뉴욕": "America/New_York",
                "일본": "JP",
                "japan": "JP",
                "tokyo": "Asia/Tokyo",
                "도쿄": "Asia/Tokyo",
            }
            mapped = aliases.get(normalized)
            if mapped:
                if mapped in pytz.all_timezones_set:
                    return mapped, self._tz_to_country.get(mapped)
                tzs = pytz.country_timezones.get(mapped)
                if tzs:
                    return tzs[0], mapped

            return None, None

        timezone, country_code = _resolve(raw)
        if not timezone:
            logger.warning("Failed to resolve timezone for input=%s", raw)
            return {
                "input": raw,
                "resolved": None,
                "error": "not_found",
                "hint": "예: 'GB', 'United Kingdom', '영국', 'Europe/London', 'London' 형태로 입력해 주세요.",
            }

        country_name = pytz.country_names.get(country_code) if country_code else None

        logger.info("Resolved nation input=%s timezone=%s country_code=%s", raw, timezone, country_code)
        return {
            "input": raw,
            "resolved": {
                "timezone": timezone,
                "country_code": country_code,
                "country_name": country_name,
            },
        }
