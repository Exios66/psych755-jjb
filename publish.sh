#!/usr/bin/env bash
# Publish the Quarto site to Posit Connect Cloud (JackJBurleson) from the
# project root. Sources .env (Posit tokens + LLM provider), then delegates to
# scripts/publish_posit_jackjburleson.py, passing all flags through.
#
# Usage (from anywhere in the repo):
#   ./publish.sh                              # full-cohort analyses + render + publish
#   ./publish.sh --skip-analysis              # skip full-cohort CLI re-runs
#   ./publish.sh --skip-analysis --skip-render # re-push the current _site/
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
fi

PYTHON="python3"
if [ -x "$ROOT/.venv/bin/python" ]; then
  PYTHON="$ROOT/.venv/bin/python"
fi

exec "$PYTHON" scripts/publish_posit_jackjburleson.py "$@"
