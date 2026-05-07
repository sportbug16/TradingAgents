import pytest

from tradingagents.dataflows.interface import _limit_tool_output
from tradingagents.dataflows.config import set_config
from tradingagents.default_config import DEFAULT_CONFIG


@pytest.mark.unit
def test_limit_tool_output_truncates_configured_strings():
    set_config({"max_tool_output_chars": 10})
    result = _limit_tool_output("x" * 20)
    set_config(DEFAULT_CONFIG.copy())
    assert result.startswith("x" * 10)
    assert "TRUNCATED" in result


@pytest.mark.unit
def test_limit_tool_output_leaves_non_strings_unchanged():
    set_config({"max_tool_output_chars": 10})
    value = {"a": 1}
    result = _limit_tool_output(value)
    set_config(DEFAULT_CONFIG.copy())
    assert result is value
