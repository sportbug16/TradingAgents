"""DhanHQ authentication health, renewal, and token generation helpers."""

from __future__ import annotations

import base64
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import hmac
import os
import struct
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import requests


DHAN_API_BASE_URL = "https://api.dhan.co/v2"
DHAN_AUTH_BASE_URL = "https://auth.dhan.co/app"


class DhanAuthError(RuntimeError):
    """Raised when Dhan auth cannot be checked or refreshed."""


@dataclass(frozen=True)
class DhanAuthResult:
    status: str
    dhan_client_id: str | None = None
    token_validity: str | None = None
    data_plan: str | None = None
    data_validity: str | None = None
    expiry_time: str | None = None
    message: str | None = None

    def public_dict(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}


class DhanAuthClient:
    """Small REST client for Dhan auth endpoints.

    Tokens and PIN/TOTP secrets are intentionally never returned in public
    result objects. Callers that request a token receive it separately so they
    can write it to a local secret store without logging it.
    """

    def __init__(
        self,
        *,
        access_token: str | None = None,
        client_id: str | None = None,
        pin: str | None = None,
        totp_secret: str | None = None,
        totp: str | None = None,
        session: requests.Session | None = None,
        timeout: float = 30.0,
    ):
        self.access_token = access_token or os.getenv("DHAN_ACCESS_TOKEN")
        self.client_id = client_id or os.getenv("DHAN_CLIENT_ID")
        self.pin = pin or os.getenv("DHAN_PIN")
        self.totp_secret = totp_secret or os.getenv("DHAN_TOTP_SECRET")
        self.totp = totp or os.getenv("DHAN_TOTP")
        self.session = session or requests.Session()
        self.timeout = timeout

    def profile(self) -> DhanAuthResult:
        token = self._require(self.access_token, "DHAN_ACCESS_TOKEN")
        response = self.session.get(
            f"{DHAN_API_BASE_URL}/profile",
            headers={"access-token": token},
            timeout=self.timeout,
        )
        payload = self._json(response)
        return DhanAuthResult(
            status="ok",
            dhan_client_id=str(payload.get("dhanClientId") or self.client_id or ""),
            token_validity=payload.get("tokenValidity"),
            data_plan=payload.get("dataPlan"),
            data_validity=payload.get("dataValidity"),
            message="Dhan access token is valid",
        )

    def renew_token(self) -> tuple[DhanAuthResult, str]:
        token = self._require(self.access_token, "DHAN_ACCESS_TOKEN")
        client_id = self._require(self.client_id, "DHAN_CLIENT_ID")
        response = self.session.get(
            f"{DHAN_API_BASE_URL}/RenewToken",
            headers={"access-token": token, "dhanClientId": client_id},
            timeout=self.timeout,
        )
        payload = self._json(response)
        new_token = self._extract_token(payload)
        return (
            DhanAuthResult(
                status="renewed",
                dhan_client_id=str(payload.get("dhanClientId") or client_id),
                expiry_time=payload.get("expiryTime") or payload.get("tokenValidity"),
                message="Dhan access token renewed",
            ),
            new_token,
        )

    def generate_access_token(self) -> tuple[DhanAuthResult, str]:
        client_id = self._require(self.client_id, "DHAN_CLIENT_ID")
        pin = self._require(self.pin, "DHAN_PIN")
        totp = self.totp or generate_totp(self._require(self.totp_secret, "DHAN_TOTP_SECRET"))
        params = urlencode({"dhanClientId": client_id, "pin": pin, "totp": totp})
        response = self.session.post(
            f"{DHAN_AUTH_BASE_URL}/generateAccessToken?{params}",
            timeout=self.timeout,
        )
        payload = self._json(response)
        new_token = self._extract_token(payload)
        return (
            DhanAuthResult(
                status="generated",
                dhan_client_id=str(payload.get("dhanClientId") or client_id),
                expiry_time=payload.get("expiryTime"),
                message="Dhan access token generated",
            ),
            new_token,
        )

    def _json(self, response) -> dict[str, Any]:
        try:
            response.raise_for_status()
        except requests.RequestException as exc:
            raise DhanAuthError(str(exc)) from exc
        payload = response.json()
        if isinstance(payload, dict) and (payload.get("errorCode") or payload.get("errorMessage")):
            raise DhanAuthError(f"{payload.get('errorCode') or 'DHAN'}: {payload.get('errorMessage') or payload}")
        if not isinstance(payload, dict):
            raise DhanAuthError("Dhan auth response was not a JSON object")
        return payload

    @staticmethod
    def _extract_token(payload: dict[str, Any]) -> str:
        token = payload.get("accessToken") or payload.get("access_token") or payload.get("token")
        if not token:
            raise DhanAuthError("Dhan auth response did not include an access token")
        return str(token)

    @staticmethod
    def _require(value: str | None, env_name: str) -> str:
        if not value:
            raise DhanAuthError(f"{env_name} is required")
        return value


def generate_totp(secret: str, for_time: int | None = None, interval: int = 30, digits: int = 6) -> str:
    normalized = secret.replace(" ", "").upper()
    missing_padding = len(normalized) % 8
    if missing_padding:
        normalized += "=" * (8 - missing_padding)
    key = base64.b32decode(normalized)
    counter = int((for_time if for_time is not None else time.time()) // interval)
    digest = hmac.new(key, struct.pack(">Q", counter), hashlib.sha1).digest()
    offset = digest[-1] & 0x0F
    code = struct.unpack(">I", digest[offset : offset + 4])[0] & 0x7FFFFFFF
    return str(code % (10**digits)).zfill(digits)


def update_env_file(path: str | Path, updates: dict[str, str]) -> None:
    env_path = Path(path)
    lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
    seen = set()
    output = []
    for line in lines:
        key = line.split("=", 1)[0].strip() if "=" in line and not line.lstrip().startswith("#") else None
        if key in updates:
            output.append(f"{key}={updates[key]}")
            seen.add(key)
        else:
            output.append(line)
    for key, value in updates.items():
        if key not in seen:
            output.append(f"{key}={value}")
    env_path.write_text("\n".join(output) + "\n", encoding="utf-8")


def result_with_checked_at(result: DhanAuthResult) -> dict[str, Any]:
    payload = result.public_dict()
    payload["checked_at"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    return payload
