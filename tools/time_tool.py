"""向模型提供可靠的实时时间与时区转换能力。"""

from __future__ import annotations

import json
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from tools.registry import registry


def get_current_time(timezone_name: str = "local") -> dict[str, str | int]:
    """返回指定时区的当前时间。

    Args:
        timezone_name: IANA 时区名称；使用 ``local`` 时读取操作系统本地时区。
    """
    normalized_name = (timezone_name or "local").strip()

    if normalized_name.lower() == "local":
        current = datetime.now().astimezone()
        resolved_timezone = current.tzname() or str(current.tzinfo) or "local"
    else:
        try:
            timezone = ZoneInfo(normalized_name)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"Unknown IANA timezone: {normalized_name}") from exc
        current = datetime.now(timezone)
        resolved_timezone = getattr(timezone, "key", normalized_name)

    utc_offset = current.strftime("%z")
    formatted_offset = (
        f"{utc_offset[:3]}:{utc_offset[3:]}" if len(utc_offset) == 5 else utc_offset
    )

    return {
        "iso8601": current.isoformat(timespec="seconds"),
        "date": current.date().isoformat(),
        "time": current.time().isoformat(timespec="seconds"),
        "weekday": current.strftime("%A"),
        "iso_weekday": current.isoweekday(),
        "timezone": resolved_timezone,
        "utc_offset": formatted_offset,
        "unix_timestamp": int(current.timestamp()),
    }


def _handle_current_time(timezone: str = "local", **kwargs) -> str:
    """ToolRegistry handler：将结构化时间转换成模型易读取的 JSON。"""
    del kwargs
    try:
        result = get_current_time(timezone)
    except ValueError as exc:
        return json.dumps({"error": str(exc)}, ensure_ascii=False)
    return json.dumps(result, ensure_ascii=False, indent=2)


CURRENT_TIME_SCHEMA = {
    "description": (
        "Get the exact current date and time. Use this tool whenever the user asks "
        "for the current time, date, weekday, timezone, or a time-sensitive answer."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "timezone": {
                "type": "string",
                "description": (
                    "IANA timezone name such as Asia/Shanghai, America/New_York, "
                    "or UTC. Defaults to the computer's local timezone."
                ),
                "default": "local",
            }
        },
    },
}


registry.register(
    name="current_time",
    toolset="time",
    schema=CURRENT_TIME_SCHEMA,
    handler=_handle_current_time,
)
