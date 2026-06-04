"""Renew Dhan access token before it reaches expiry.

This script is intended for automation. It never prints access tokens, PINs, or
TOTP secrets. If the current token is already invalid, it exits non-zero because
Dhan's renewal path requires a still-valid token.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import warnings

from dotenv import load_dotenv

warnings.filterwarnings("ignore", message="Pandas requires version .*", category=UserWarning)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tradingagents.india.auth import DhanAuthClient, DhanAuthError, result_with_checked_at, update_env_file


def main() -> None:
    load_dotenv()
    args = _parse_args()
    try:
        payload = renew_guard(
            renew_before_hours=args.renew_before_hours,
            write_env=args.write_env,
            env_file=args.env_file,
            timeout=args.timeout,
        )
        print(json.dumps(payload, indent=2, default=str))
    except DhanAuthError as exc:
        print(json.dumps({"status": "error", "message": str(exc)}, indent=2), file=sys.stderr)
        raise SystemExit(1) from exc


def renew_guard(
    *,
    renew_before_hours: float,
    write_env: bool,
    env_file: str,
    timeout: float,
) -> dict:
    client = DhanAuthClient(timeout=timeout)
    before = client.profile()
    expiry = parse_dhan_token_validity(before.token_validity)
    hours_remaining = None
    should_renew = True
    if expiry is not None:
        hours_remaining = (expiry - datetime.now()).total_seconds() / 3600
        should_renew = hours_remaining <= renew_before_hours

    if not should_renew:
        payload = result_with_checked_at(before)
        payload.update(
            {
                "status": "valid_no_renewal_needed",
                "hours_remaining": hours_remaining,
                "renew_before_hours": renew_before_hours,
                "env_updated": False,
            }
        )
        return payload

    renewed, token = client.renew_token()
    if write_env:
        update_env_file(env_file, {"DHAN_ACCESS_TOKEN": token})
    after = DhanAuthClient(access_token=token, client_id=client.client_id, timeout=timeout).profile()
    payload = result_with_checked_at(after)
    payload.update(
        {
            "status": renewed.status,
            "previous_token_validity": before.token_validity,
            "new_expiry_time": renewed.expiry_time,
            "hours_remaining_before_renewal": hours_remaining,
            "renew_before_hours": renew_before_hours,
            "env_updated": bool(write_env),
        }
    )
    return payload


def parse_dhan_token_validity(value: str | None) -> datetime | None:
    if not value:
        return None
    for fmt in ("%d/%m/%Y %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--renew-before-hours", type=float, default=18.0)
    parser.add_argument("--write-env", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--timeout", type=float, default=30.0)
    return parser.parse_args()


if __name__ == "__main__":
    main()
