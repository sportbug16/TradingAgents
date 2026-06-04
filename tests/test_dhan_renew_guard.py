from datetime import datetime, timedelta

import pytest

import scripts.dhan_renew_guard as guard
from tradingagents.india.auth import DhanAuthResult


def test_parse_dhan_token_validity():
    assert guard.parse_dhan_token_validity("13/05/2026 15:44") == datetime(2026, 5, 13, 15, 44)
    assert guard.parse_dhan_token_validity("bad") is None


def test_renew_guard_skips_when_valid_beyond_threshold(monkeypatch, tmp_path):
    future = (datetime.now() + timedelta(hours=24)).strftime("%d/%m/%Y %H:%M")

    class Client:
        client_id = "client-1"

        def __init__(self, *args, **kwargs):
            pass

        def profile(self):
            return DhanAuthResult(status="ok", token_validity=future, data_plan="Active")

        def renew_token(self):
            raise AssertionError("should not renew")

    monkeypatch.setattr(guard, "DhanAuthClient", Client)
    result = guard.renew_guard(renew_before_hours=18, write_env=True, env_file=str(tmp_path / ".env"), timeout=1)

    assert result["status"] == "valid_no_renewal_needed"
    assert result["env_updated"] is False


def test_renew_guard_renews_and_writes_env(monkeypatch, tmp_path):
    soon = (datetime.now() + timedelta(hours=2)).strftime("%d/%m/%Y %H:%M")

    class Client:
        client_id = "client-1"

        def __init__(self, *args, **kwargs):
            self.access_token = kwargs.get("access_token")

        def profile(self):
            if self.access_token == "new-token":
                return DhanAuthResult(status="ok", token_validity="14/05/2026 09:00", data_plan="Active")
            return DhanAuthResult(status="ok", token_validity=soon, data_plan="Active")

        def renew_token(self):
            return DhanAuthResult(status="renewed", expiry_time="2026-05-14T09:00:00"), "new-token"

    monkeypatch.setattr(guard, "DhanAuthClient", Client)
    env_file = tmp_path / ".env"
    env_file.write_text("DHAN_ACCESS_TOKEN=old-token\n", encoding="utf-8")

    result = guard.renew_guard(renew_before_hours=18, write_env=True, env_file=str(env_file), timeout=1)

    assert result["status"] == "renewed"
    assert result["env_updated"] is True
    assert env_file.read_text(encoding="utf-8") == "DHAN_ACCESS_TOKEN=new-token\n"
