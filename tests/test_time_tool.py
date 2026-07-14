"""Tests for the model-facing current-time tool."""

import json
import time

from tools.registry import registry
from tools.time_tool import get_current_time


def test_get_current_time_in_utc():
    before = int(time.time())

    result = get_current_time("UTC")

    after = int(time.time())
    assert result["timezone"] == "UTC"
    assert result["utc_offset"] == "+00:00"
    assert result["iso8601"].endswith("+00:00")
    assert 1 <= result["iso_weekday"] <= 7
    assert result["weekday"]
    assert before <= result["unix_timestamp"] <= after


def test_get_current_time_in_shanghai():
    result = get_current_time("Asia/Shanghai")

    assert result["timezone"] == "Asia/Shanghai"
    assert result["utc_offset"] == "+08:00"
    assert result["iso8601"].endswith("+08:00")


def test_registry_dispatches_current_time_tool():
    result = json.loads(registry.dispatch("current_time", {"timezone": "UTC"}))

    assert result["timezone"] == "UTC"
    assert "iso8601" in result
    assert "unix_timestamp" in result


def test_registry_returns_error_for_unknown_timezone():
    result = json.loads(
        registry.dispatch("current_time", {"timezone": "Mars/Olympus_Mons"})
    )

    assert result == {"error": "Unknown IANA timezone: Mars/Olympus_Mons"}
