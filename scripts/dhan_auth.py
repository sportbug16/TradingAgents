"""DhanHQ auth health, renew, and token generation CLI."""

from __future__ import annotations

import argparse
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
    client = DhanAuthClient(timeout=args.timeout)
    try:
        if args.command == "health":
            result = client.profile()
            print(json.dumps(result_with_checked_at(result), indent=2, default=str))
            return
        if args.command == "renew":
            result, token = client.renew_token()
            if args.write_env:
                update_env_file(args.env_file, {"DHAN_ACCESS_TOKEN": token})
            payload = result_with_checked_at(result)
            payload["env_updated"] = bool(args.write_env)
            print(json.dumps(payload, indent=2, default=str))
            return
        if args.command == "generate":
            result, token = client.generate_access_token()
            if args.write_env:
                update_env_file(args.env_file, {"DHAN_ACCESS_TOKEN": token})
            payload = result_with_checked_at(result)
            payload["env_updated"] = bool(args.write_env)
            print(json.dumps(payload, indent=2, default=str))
            return
    except DhanAuthError as exc:
        print(json.dumps({"status": "error", "message": str(exc)}, indent=2), file=sys.stderr)
        raise SystemExit(1) from exc


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["health", "renew", "generate"])
    parser.add_argument("--write-env", action="store_true")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--timeout", type=float, default=30.0)
    return parser.parse_args()


if __name__ == "__main__":
    main()
