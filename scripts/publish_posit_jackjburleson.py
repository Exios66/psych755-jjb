#!/usr/bin/env python3
"""Render (optional full-cohort re-run) and publish to JackJBurleson Posit Connect Cloud.

Canonical target:
  account: jackjburleson
  content: 019f9a10-ebb9-d1d5-839f-97e794bfd0ca
  share:   https://019f9a10-ebb9-d1d5-839f-97e794bfd0ca.share.connect.posit.cloud/

Requires private File A/B/C (sibling_data or /tmp/sibling_data) unless
``--allow-excerpt`` is passed (not recommended for production publishes).

Auth:
  - env POSIT_CONNECT_CLOUD_ACCESS_TOKEN (+ REFRESH_TOKEN, ACCOUNT_ID), or
  - interactive device-code flow (prints URL + code; polls until approved)
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CONTENT_ID = "019f9a10-ebb9-d1d5-839f-97e794bfd0ca"
ACCOUNT_NAME = "jackjburleson"
SHARE_URL = f"https://{CONTENT_ID}.share.connect.posit.cloud/"
UI_URL = f"https://connect.posit.cloud/{ACCOUNT_NAME}/content/{CONTENT_ID}"
API = "https://api.connect.posit.cloud/v1"
AUTH_HOST = "login.posit.cloud"
CLIENT_ID = "quarto-cli"
SCOPE = "vivid"

SIBLING = (ROOT / ".." / "sibling_data").resolve()
TMP_SIBLING = Path("/tmp/sibling_data")
FILE_A = "PRCAProlificExport_FileA.csv"
FILE_B = "PRCAProlificExport_FileB.csv"
FILE_C = "PRCAQualtricsExport_FileC.csv"


def _log(msg: str) -> None:
    print(msg, flush=True)


def resolve_full_data(*, allow_excerpt: bool) -> tuple[list[Path], Path]:
    from ca_personas.paths import (
        DEFAULT_PROLIFIC_A,
        DEFAULT_PROLIFIC_B,
        DEFAULT_QUALTRICS_C,
        EXCERPT_PROLIFIC,
        EXCERPT_QUALTRICS,
        sibling_data_available,
    )

    if sibling_data_available():
        return [DEFAULT_PROLIFIC_A, DEFAULT_PROLIFIC_B], DEFAULT_QUALTRICS_C

    tmp_a, tmp_b, tmp_c = TMP_SIBLING / FILE_A, TMP_SIBLING / FILE_B, TMP_SIBLING / FILE_C
    if tmp_a.is_file() and tmp_b.is_file() and tmp_c.is_file():
        SIBLING.mkdir(parents=True, exist_ok=True)
        for src, name in ((tmp_a, FILE_A), (tmp_b, FILE_B), (tmp_c, FILE_C)):
            dest = SIBLING / name
            if not dest.is_file() or dest.stat().st_mtime < src.stat().st_mtime:
                shutil.copy2(src, dest)
                _log(f"Staged {src} → {dest}")
        return [SIBLING / FILE_A, SIBLING / FILE_B], SIBLING / FILE_C

    if allow_excerpt:
        _log("WARNING: --allow-excerpt in use; results are NOT full-cohort.")
        return [EXCERPT_PROLIFIC], EXCERPT_QUALTRICS

    raise SystemExit(
        "Full cohort File A/B/C not found. Place exports in ../sibling_data/ "
        "or /tmp/sibling_data/, or pass --allow-excerpt (discouraged)."
    )


def run(cmd: list[str], *, cwd: Path = ROOT) -> None:
    _log("$ " + " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def run_full_analyses(prolific: list[Path], qualtrics: Path, *, seed: int) -> None:
    common = [
        "--join",
        "inner",
        "--seed",
        str(seed),
        "--prolific",
        *[str(p) for p in prolific],
        "--qualtrics",
        str(qualtrics),
    ]
    run(["ca-personas", "prepare", "--join", "inner", "--prolific", *[str(p) for p in prolific], "--qualtrics", str(qualtrics)])
    for cmd in (
        ["ca-personas", "transit-ca", *common],
        ["ca-personas", "geo-transit-rf", *common],
        ["ca-personas", "ca-transit-rf", *common],
        [
            "ca-personas",
            "covariate-transit-rf",
            *common,
            "--specs",
            "q27_intensity",
            "q28_days",
            "q27_q28",
            "rideshare",
            "car_access",
            "employment",
            "mobility_bundle",
            "--figures-dir",
            "memos/figures",
        ],
    ):
        run(cmd)


def verify_analysis_artifacts(*, allow_excerpt: bool) -> dict[str, float]:
    """Read key result cards; enforce full-cohort N when not allowing excerpts."""
    checks: dict[str, float] = {}
    q28 = ROOT / "outputs/transit_covariate_rf/q28_days/q28_days_results_card.json"
    if q28.is_file():
        card = json.loads(q28.read_text())
        n = int(card["sample"]["n"])
        auc = float(card["cv_metrics"]["roc_auc"])
        checks["q28_n"] = n
        checks["q28_auc"] = auc
        _log(f"Verified q28_days artifacts: n={n}, auc={auc:.3f}")
        if not allow_excerpt and n < 200:
            raise SystemExit(f"Full-data gate failed: Q28 n={n} (<200). Refusing publish.")
    else:
        _log("WARNING: missing q28_days results card; run analyses before publish.")
    return checks


def ensure_quarto() -> None:
    if shutil.which("quarto") is None:
        raise SystemExit("quarto not on PATH; install Quarto ≥ 1.10")
    out = subprocess.check_output(["quarto", "--version"], text=True).strip()
    _log(f"quarto {out}")


def render_site() -> Path:
    ensure_quarto()
    run(["quarto", "render"])
    site = ROOT / "_site"
    index = site / "index.html"
    if not index.is_file():
        raise SystemExit("quarto render did not produce _site/index.html")
    return site


def post_form(url: str, data: dict[str, str]) -> dict:
    body = urllib.parse.urlencode(data).encode()
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"},
    )
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read().decode())


def device_auth() -> dict:
    auth = post_form(
        f"https://{AUTH_HOST}/oauth/device/authorize",
        {"scope": SCOPE, "client_id": CLIENT_ID},
    )
    _log("=" * 72)
    _log("AUTHORIZE NOW (Posit Connect Cloud / JackJBurleson)")
    _log("=" * 72)
    _log(f"URL:  {auth['verification_uri_complete']}")
    _log(f"CODE: {auth['user_code']}")
    _log("=" * 72)
    interval = max(int(auth.get("interval", 5)), 5)
    expires = int(auth.get("expires_in", 1800))
    start = time.time()
    while True:
        if time.time() - start > expires:
            raise SystemExit("Device authorization timed out.")
        try:
            tok = post_form(
                f"https://{AUTH_HOST}/oauth/token",
                {
                    "scope": SCOPE,
                    "client_id": CLIENT_ID,
                    "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                    "device_code": auth["device_code"],
                },
            )
            _log(f"Authorized after {time.time() - start:.0f}s")
            return tok
        except urllib.error.HTTPError as e:
            raw = e.read().decode()
            try:
                code = json.loads(raw).get("error", raw)
            except Exception:
                code = raw.strip()
            if code == "authorization_pending":
                time.sleep(interval)
                continue
            if code == "slow_down":
                interval += 5
                time.sleep(interval)
                continue
            raise SystemExit(f"OAuth error: {code}")


def load_tokens() -> tuple[str, str | None]:
    access = os.environ.get("POSIT_CONNECT_CLOUD_ACCESS_TOKEN")
    refresh = os.environ.get("POSIT_CONNECT_CLOUD_REFRESH_TOKEN")
    if access:
        _log("Using POSIT_CONNECT_CLOUD_* environment tokens")
        return access, refresh
    tok = device_auth()
    Path("/tmp/posit-tokens.json").write_text(json.dumps(tok, indent=2), encoding="utf-8")
    return tok["access_token"], tok.get("refresh_token")


def api(
    method: str,
    path: str,
    access: str,
    body: dict | None = None,
) -> dict | None:
    data = None if body is None else json.dumps(body).encode()
    headers = {"Accept": "application/json", "Authorization": f"Bearer {access}"}
    if body is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(f"{API}/{path}", data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req) as r:
            raw = r.read()
            return json.loads(raw.decode()) if raw else None
    except urllib.error.HTTPError as e:
        raise SystemExit(f"{method} {path} → {e.code}: {e.read().decode()[:800]}") from e


def assert_writable_account(access: str) -> str:
    accounts = api("GET", "accounts?has_user_role=true", access) or {}
    rows = accounts.get("data") or []
    names = [a.get("name") for a in rows]
    _log(f"Authorized accounts: {names}")
    # Prefer exact JackJBurleson account; else first writable.
    for a in rows:
        if a.get("name") == ACCOUNT_NAME:
            return a["id"]
    env_id = os.environ.get("POSIT_CONNECT_CLOUD_ACCOUNT_ID")
    if env_id:
        return env_id
    if not rows:
        raise SystemExit("No publishable Posit accounts for this login.")
    _log(f"WARNING: '{ACCOUNT_NAME}' not in account list; using {rows[0].get('name')}")
    return rows[0]["id"]


def make_bundle(site: Path) -> bytes:
    buf = io.BytesIO()
    files = sorted(p for p in site.rglob("*") if p.is_file())
    manifest = {
        "version": 1,
        "locale": "en_US",
        "platform": "4.0.0",
        "metadata": {"appmode": "static", "primary_rmd": None, "primary_html": "index.html"},
        "packages": None,
        "files": {p.relative_to(site).as_posix(): {"checksum": ""} for p in files},
        "users": None,
    }
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        man = json.dumps(manifest).encode()
        info = tarfile.TarInfo("manifest.json")
        info.size = len(man)
        tar.addfile(info, io.BytesIO(man))
        for p in files:
            tar.add(p, arcname=p.relative_to(site).as_posix())
    return buf.getvalue()


def publish_bundle(access: str, site: Path) -> dict:
    content = api("GET", f"contents/{CONTENT_ID}", access) or {}
    perms = content.get("permissions") or []
    if "content:read" in perms and not any(
        p.startswith("content:") and p != "content:read" for p in perms
    ):
        # still try PATCH; API will 403 if not writable
        _log(f"Content permissions: {perms}")

    updated = api(
        "PATCH",
        f"contents/{CONTENT_ID}?new_bundle=true",
        access,
        {
            "secrets": [],
            "revision_overrides": {"primary_file": "index.html", "app_mode": "static"},
        },
    ) or {}
    rev = updated.get("next_revision") or updated.get("current_revision") or {}
    upload_url = rev.get("source_bundle_upload_url")
    if not upload_url:
        raise SystemExit(f"No upload URL for content {CONTENT_ID}")

    bundle = make_bundle(site)
    _log(f"Uploading bundle ({len(bundle)} bytes)")
    req = urllib.request.Request(
        upload_url,
        data=bundle,
        method="POST",
        headers={"Content-Type": "application/gzip"},
    )
    with urllib.request.urlopen(req) as r:
        _log(f"upload_status {r.status}")

    req = urllib.request.Request(
        f"{API}/contents/{CONTENT_ID}/publish",
        method="POST",
        headers={"Accept": "application/json", "Authorization": f"Bearer {access}"},
    )
    with urllib.request.urlopen(req) as r:
        _log(f"publish_http {r.status}")
        r.read()

    for i in range(60):
        content = api("GET", f"contents/{CONTENT_ID}", access) or {}
        rev = content.get("current_revision") or {}
        result = rev.get("publish_result")
        status = rev.get("status") or rev.get("state")
        url = rev.get("url")
        _log(f"poll[{i}] status={status} result={result} url={url}")
        if result == "success" or status == "published":
            return {"content": content, "share_url": url or SHARE_URL, "ui_url": UI_URL}
        if result and result not in {"success", "running", None}:
            raise SystemExit(f"Publish failed: {rev.get('publish_error_code')} {rev.get('publish_error_args')}")
        time.sleep(3)
    raise SystemExit("Timed out waiting for publish success")


def verify_live(share_url: str, *, expect_substrings: list[str]) -> None:
    req = urllib.request.Request(share_url, headers={"User-Agent": "psych755-jjb-publish/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        html = r.read().decode("utf-8", "replace")
        code = r.status
    if code != 200:
        raise SystemExit(f"Live verify HTTP {code}")
    missing = [s for s in expect_substrings if s not in html]
    if missing:
        raise SystemExit(f"Live page missing expected strings: {missing}")
    _log(f"Live verification OK ({len(html)} bytes): {share_url}")
    try:
        from playwright.sync_api import sync_playwright

        art = Path("/opt/cursor/artifacts")
        art.mkdir(parents=True, exist_ok=True)
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(share_url, wait_until="networkidle", timeout=90000)
            page.screenshot(path=str(art / "connect-cloud-published.png"), full_page=False)
            browser.close()
        _log(f"Screenshot → {art / 'connect-cloud-published.png'}")
    except Exception as exc:  # noqa: BLE001
        _log(f"Screenshot skipped: {exc}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--skip-analysis", action="store_true", help="Skip full-cohort CLI re-runs")
    p.add_argument("--skip-render", action="store_true", help="Publish existing _site/")
    p.add_argument("--allow-excerpt", action="store_true", help="Allow excerpt fixtures (discouraged)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--expect",
        action="append",
        default=[],
        help="Substring that must appear on the live share page (repeatable)",
    )
    args = p.parse_args(argv)

    prolific, qualtrics = resolve_full_data(allow_excerpt=args.allow_excerpt)
    _log(f"Data: prolific={[str(x) for x in prolific]} qualtrics={qualtrics}")

    if not args.skip_analysis:
        run_full_analyses(prolific, qualtrics, seed=args.seed)
    checks = verify_analysis_artifacts(allow_excerpt=args.allow_excerpt)

    if not args.skip_render:
        site = render_site()
    else:
        site = ROOT / "_site"
        if not (site / "index.html").is_file():
            raise SystemExit("_site/index.html missing; refuse --skip-render")

    access, _refresh = load_tokens()
    account_id = assert_writable_account(access)
    _log(f"Using account_id={account_id}")
    result = publish_bundle(access, site)

    expect = list(args.expect) or ["Do LLMs Stereotype Communication Apprehension", "Q27"]
    if checks.get("q28_auc"):
        expect.append(f"{checks['q28_auc']:.3f}"[:5])  # e.g. 0.762
    verify_live(result["share_url"], expect_substrings=expect)

    out = {
        **result,
        "account": ACCOUNT_NAME,
        "content_id": CONTENT_ID,
        "checks": checks,
    }
    Path("/tmp/posit-publish-result.json").write_text(json.dumps(out, indent=2, default=str))
    _log("UI_URL " + result["ui_url"])
    _log("SHARE_URL " + result["share_url"])
    _log("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
