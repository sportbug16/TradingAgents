import json

import pytest

from tradingagents.india.auth import DhanAuthClient, DhanAuthError, generate_totp, update_env_file


class _Response:
    def __init__(self, payload, status_error=None):
        self.payload = payload
        self.status_error = status_error

    def raise_for_status(self):
        if self.status_error:
            raise self.status_error

    def json(self):
        return self.payload


class _Session:
    def __init__(self):
        self.calls = []

    def get(self, url, headers, timeout):
        self.calls.append(("GET", url, headers, timeout))
        if url.endswith("/profile"):
            return _Response(
                {
                    "dhanClientId": "client-1",
                    "tokenValidity": "30/03/2026 15:37",
                    "dataPlan": "Active",
                    "dataValidity": "2026-05-30 09:37:52.0",
                }
            )
        return _Response({"dhanClientId": "client-1", "accessToken": "new-token", "expiryTime": "2026-05-10T10:00:00"})

    def post(self, url, timeout):
        self.calls.append(("POST", url, {}, timeout))
        return _Response({"dhanClientId": "client-1", "accessToken": "generated-token", "expiryTime": "2026-05-10T10:00:00"})


@pytest.mark.unit
def test_generate_totp_matches_rfc_vector():
    assert generate_totp("GEZDGNBVGY3TQOJQGEZDGNBVGY3TQOJQ", for_time=59, digits=8) == "94287082"


@pytest.mark.unit
def test_profile_returns_sanitized_health():
    result = DhanAuthClient(access_token="token", session=_Session()).profile()
    assert result.status == "ok"
    assert result.dhan_client_id == "client-1"
    assert result.data_plan == "Active"
    assert "ey" not in json.dumps(result.public_dict()).lower()


@pytest.mark.unit
def test_renew_token_returns_token_separately():
    result, token = DhanAuthClient(access_token="token", client_id="client-1", session=_Session()).renew_token()
    assert result.status == "renewed"
    assert token == "new-token"
    assert "new-token" not in json.dumps(result.public_dict())


@pytest.mark.unit
def test_generate_access_token_uses_pin_and_totp_secret():
    session = _Session()
    result, token = DhanAuthClient(
        client_id="client-1",
        pin="123456",
        totp_secret="GEZDGNBVGY3TQOJQGEZDGNBVGY3TQOJQ",
        session=session,
    ).generate_access_token()
    assert result.status == "generated"
    assert token == "generated-token"
    method, url, _, _ = session.calls[0]
    assert method == "POST"
    assert "dhanClientId=client-1" in url
    assert "pin=123456" in url
    assert "totp=" in url


@pytest.mark.unit
def test_missing_required_secret_raises():
    with pytest.raises(DhanAuthError, match="DHAN_PIN is required"):
        DhanAuthClient(client_id="client-1").generate_access_token()


@pytest.mark.unit
def test_update_env_file_replaces_existing_key(tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text("A=1\nDHAN_ACCESS_TOKEN=old\n", encoding="utf-8")
    update_env_file(env_path, {"DHAN_ACCESS_TOKEN": "new"})
    assert env_path.read_text(encoding="utf-8") == "A=1\nDHAN_ACCESS_TOKEN=new\n"
