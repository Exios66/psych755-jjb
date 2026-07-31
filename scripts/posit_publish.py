#!/usr/bin/env python3
"""One-command publish to Posit Connect Cloud.

Usage:
    python scripts/posit_publish.py               # first time (OAuth)
    POSIT_TOKEN=xxx python scripts/posit_publish.py  # with API key

To get an API key:
    1. Go to https://connect.posit.cloud/jackjburleson
    2. Click your avatar → Profile → API Keys
    3. Create a new key and paste it below
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Check for API key first
token = os.environ.get("POSIT_CONNECT_CLOUD_ACCESS_TOKEN")
if not token:
    try:
        from dotenv import load_dotenv

        dotenv_path = ROOT / ".env"
        if dotenv_path.is_file():
            load_dotenv(dotenv_path)
            token = os.environ.get("POSIT_CONNECT_CLOUD_ACCESS_TOKEN")
    except ImportError:
        pass

if token:
    print("Using POSIT_CONNECT_CLOUD_ACCESS_TOKEN from environment")
    os.environ["POSIT_CONNECT_CLOUD_ACCESS_TOKEN"] = token
else:
    print("No API key found. Will use interactive OAuth.")
    print("  Step 1: A URL + code will appear below")
    print("  Step 2: Open the URL in your browser")
    print("  Step 3: Log in and enter the code")
    print("  Step 4: Come back here — it auto-publishes\n")

os.chdir(str(ROOT))
# Run the publish script
ret = os.system(
    f"{sys.executable} scripts/publish_posit_jackjburleson.py "
    "--skip-analysis --skip-render"
)
sys.exit(ret >> 8 if ret >= 0 else 1)