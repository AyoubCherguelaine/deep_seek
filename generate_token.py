#!/usr/bin/env python3
import os
import sys
from datetime import datetime, timedelta, timezone

import jwt


def load_env_file(path: str) -> dict[str, str]:
    values: dict[str, str] = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip()
    except FileNotFoundError:
        raise SystemExit(f"Missing {path} file.")
    return values


def main() -> int:
    env_file = os.getenv("ENV_FILE", ".env")
    subject = sys.argv[1] if len(sys.argv) > 1 else "api-client"

    env = load_env_file(env_file)
    secret = env.get("AUTH_API_KEY", "").strip()
    if not secret:
        raise SystemExit(f"AUTH_API_KEY is missing in {env_file}.")

    ttl_days = int(env.get("AUTH_TOKEN_TTL_DAYS", "365"))
    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(days=ttl_days)

    payload = {
        "sub": subject,
        "iat": int(now.timestamp()),
        "exp": int(expires_at.timestamp()),
        "scope": "ocr",
    }

    token = jwt.encode(payload, secret, algorithm="HS256")
    print(token)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
